import os
import glob
import tqdm
import torch
import pickle
import requests
import argparse
import numpy as np

from torch.utils.data import Dataset

# This is a copy of MD17Dataset with minor changes to enable label rescaling
# Parent class of SMD17, Intended to used as a molecules-mixed dataset
class SMD17Dataset(Dataset):
    def __init__(self, style, task, split, root="./datas"):
        # Names
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"}
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12}

        # Download missing data
        missing = [v for v in self.Mdict.values() if not os.path.isfile(f"{root}/{v}_dft.npz")]
        if missing: self.download(root, missing) 
        
        datas_dict, labels_dict = self.load_dataset(root, "*", style, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)
    
    def __len__(self):
        return len(self.datas["R"])

    def __getitem__(self, idx):
        return self.datas["R"][idx], self.datas["z"], self.labels["E"][idx], self.labels["F"][idx]

    def collate(self, batch):
        data = {"R":[], "z":[], "batch":[], "n":[]}
        label = {"E":[], "F":[]}      

        for i, b in enumerate(batch):
            data["R"].append(b[0])
            data["z"].append(b[1])
            data["n"].append(len(b[1]))
            data["batch"].append(torch.ones((b[1].size()), dtype=torch.int64)*i)
            label["E"].append(b[2])
            label["F"].append(b[3])
        data["R"] = torch.cat(data["R"])
        data["z"] = torch.cat(data["z"])
        data["n"] = torch.Tensor(data["n"])
        data["batch"] = torch.cat(data["batch"])
        label["E"] = torch.stack(label["E"])
        label["F"] = torch.stack(label["F"])
        #print("collate", data["R"].shape, data["z"].shape, data["n"].shape, data["batch"].shape, label["E"].shape, label["F"].shape)

        return data, label

    def print_info(self):
        print("Data sizes:")
        for m in self.Mdict.values():
            R = "R"
            print(f"{m}: {len(self.datas[m][R])}")
        return

    def download(self, path, molecules):
        print("Downloading missing datas")
        for m in tqdm.tqdm(molecules):
            if m == "benzene":
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}2017.npz")
            else:
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}.npz")
            open(os.path.join(path, f"{m}_dft.npz") , "wb").write(r.content)
        return 

    def load_dataset(self, root, style, molecule, task, split):
        npzs = glob.glob(f"{root}/{molecule}_dft.npz")
        datas, labels = dict(), dict()

        for npz in npzs:
            z = np.load(npz, allow_pickle=True)
            name = npz.split("/")[-1].split("_")[0]
            datas[name], labels[name] = dict(), dict()

            with open(f"{root}/../{style}/{split}/{name}/{task}.pkl", "rb") as f:
                idx = pickle.load(f)

            # Keys are ['E', 'name', 'F', 'theory', 'R', 'z', 'type', 'md5']
            datas[name]["R"] = z["R"][idx]
            datas[name]["z"] = z["z"]
            labels[name]["E"] = z["E"][idx]
            labels[name]["F"] = z["F"][idx]
       
        return datas, labels
    
    def unroll_data(self, datas_dict, labels_dict):
        datas, labels = dict(), dict()
        #TBD

        return datas, labels

# Child class, only for one molecule. Mostly used now
class SMD17SingleDataset(SMD17Dataset):
    def __init__(self, style,  m, task, split, root="./datas"):
        # Initialize
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"} # full name of molecule
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12} # number of atom in each molecule
        self.identifier = f"SMD17SingleDataset_{style}_{m}{split}{task}" # name tag for this dataset configuration
        self.m = self.Mdict[m]
        self.natom = self.Ndict[m]            
        
        # Download missing data
        missing = [v for v in self.Mdict.values() if not os.path.isfile(f"{root}/{v}_dft.npz")]
        if missing: self.download(root, missing) 
        
        # Load and organized data
        datas_dict, labels_dict = self.load_dataset(root, style, self.m, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)
        self.C = 1000000.0
        self.labels["E"], self.labels["F"] = self.labels["E"]/self.C  , self.labels["F"]/self.C

    def print_info(self):
        print("Data sizes:")
        R = "R"
        print(f"{self.m}: {len(self.datas[R])}")
        return

    def unroll_data(self, datas_dict, labels_dict):
        datas = {"R":torch.Tensor(datas_dict[self.m]["R"]), "z":torch.LongTensor(datas_dict[self.m]["z"])} 
        labels = {"E":torch.Tensor(labels_dict[self.m]["E"]), "F":torch.Tensor(labels_dict[self.m]["F"])}

        return datas, labels


def parse_index(exps, l):
    indice = []
    
    for exp in exps.split(","):
        if exp == ":": indice += [i for i in range(l)]
        elif ":" not in exp:  indice += [int(exp)]
        else:
            if exp[0] == ":": indice += [i for i in range(int(exp[1:])+1)]
            elif exp[-1] == ":": indice += [i for i in range(int(exp[:-1])+1)]
            else: indice += [i for i in range(int(exp.split(":")[0]), int(exp.split(":")[1])+1)]

    return indice

def main(args):
    # As a script it could be used to display values in the dataset
    
    # Load dataset
    dataset = SMD17SingleDataset(args.style, args.molecule, args.task, args.split)
    if args.info: dataset.print_info()
    
    # Get numerical index
    try:
        indice = parse_index(args.index, len(dataset.labels["E"]))
    except BaseException as err:
        print(err)
        print("expression of index incorrect")

    # Print
    R, z, E, F = "R", "z", "E", "F" 
    #print(f"covariance_inv:{dataset.covariance_inv}")
    #print(len(dataset.datas[z]))
    for idx in indice:
        print(f"{idx:06d}:\n (R={dataset.datas[R][idx]},\t z={dataset.datas[z]}),\t E={dataset.labels[E][idx]},\t F={dataset.labels[F][idx]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", type=str, help="path to the root dir of dataset", default="./datas")
    parser.add_argument("-y", "--style", type=str, default="random", help="specify how to split the data")
    parser.add_argument("-m", "--molecule", type=str, help="lowercase initial of the molecule in the dataset", required=True)
    parser.add_argument("-t", "--task", type=str, help="train or valid or test", required=True)
    parser.add_argument("-s", "--split", type=int, help="the name of dataset subset, aka the number of train samples", required=True)
    parser.add_argument("-i", "--index", type=str, help="indice of the data to be shown", default=":")
    parser.add_argument("-f", "--info", type=bool, help="print the information about this dataset", default=False)

    main(parser.parse_args())
