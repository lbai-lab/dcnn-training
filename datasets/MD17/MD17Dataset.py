import os
import glob
import tqdm
import torch
import pickle
import requests
import argparse
import numpy as np

from torch.utils.data import Dataset

# Parent class of MD17, Intended to used as a molecules-mixed dataset
class MD17Dataset(Dataset):
    def __init__(self, task, split, root="./datas"):
        # Names
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"}
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12}

        # Download missing data
        missing = [v for v in self.Mdict.values() if not os.path.isfile(f"{root}/{v}_dft.npz")]
        if missing: self.download(root, missing) 
        
        datas_dict, labels_dict = self.load_dataset(root, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)

    def __len__(self):
        return len(self.datas["R"])

    def __getitem__(self, idx):
        return self.datas["R"][idx], self.datas["z"][idx], self.labels["E"][idx], self.labels["F"][idx]

    def collate(self, batch):
        data = {"R":[], "z":[], "batch":[], "n":[]}
        label = {"E":[], "F":[]}      
        
        for i, b in enumerate(batch):
            data["R"].append(b[0])
            data["z"].append(b[1])
            data["n"].append(len(b[1]))
            data["batch"].append(torch.ones((len(b[1])), dtype=torch.int64)*i)
            label["E"].append(b[2])
            label["F"].append(b[3])
        data["R"] = torch.cat(data["R"])
        data["z"] = torch.cat(data["z"])
        data["n"] = torch.Tensor(data["n"])
        data["batch"] = torch.cat(data["batch"])
        label["E"] = torch.cat(label["E"])
        label["F"] = torch.cat(label["F"])
        #print("collate", data["R"].shape, data["z"].shape, data["n"].shape, data["batch"].shape, label["E"].shape, label["F"].shape)

        return data, label

    def download(self, path, molecules):
        print("Downloading missing datas")
        for m in tqdm.tqdm(molecules):
            if m == "benzene":
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}2017.npz")
            else:
                r = requests.get(f"http://www.quantum-machine.org/gdml/data/npz/md17_{m}.npz")
            open(os.path.join(path, f"{m}_dft.npz") , "wb").write(r.content)
        return 

    def load_dataset(self, root, task, split):
        npzs = glob.glob(f"{root}/*_dft.npz")
        datas, labels = dict(), dict()

        for npz in npzs:
            z = np.load(npz, allow_pickle=True)
            name = npz.split("/")[-1].split("_")[0]
            datas[name], labels[name] = dict(), dict()

            with open(f"{root}/../splits/all/random/{split}/{name}/{task}.pkl", "rb") as f:
                idx = pickle.load(f)

            # Keys are ['E', 'name', 'F', 'theory', 'R', 'z', 'type', 'md5']
            datas[name]["R"] = torch.Tensor(z["R"][idx])
            datas[name]["z"] = torch.LongTensor(z["z"])
            labels[name]["E"] = torch.Tensor(z["E"][idx])
            labels[name]["F"] = torch.Tensor(z["F"][idx])
       
        return datas, labels
    
    def unroll_data(self, datas_dict, labels_dict):
        datas = {"R":[], "z":[]}
        labels = {"E":[], "F":[]}

        for m in self.Mdict.values():
            for r in datas_dict[m]["R"]: 
                datas["R"].append(r)
                datas["z"].append(datas_dict[m]["z"])
            for e in labels_dict[m]["E"]: labels["E"].append(e)
            for f in labels_dict[m]["F"]: labels["F"].append(f)

        return datas, labels
    

# Child class, only for one molecule. 
class MD17SingleDataset(MD17Dataset):
    def __init__(self, style,  m, task, split, root="./datas"):
        # Initialize
        self.Mdict = {"a":"aspirin", "b":"benzene", "e":"ethanol", "m":"malonaldehyde", "n":"naphthalene", "s":"salicylic", "t":"toluene", "u":"uracil"} # full name of molecule
        self.Ndict = {"a":21, "b":12, "e":9, "m":9, "n":18, "s":16, "t":15, "u":12} # number of atom in each molecule
        self.identifier = f"MD17SingleDataset_{style}_{m}{split}{task}" # name tag for this dataset configuration
        self.m = self.Mdict[m]
        self.natom = self.Ndict[m]
        
        # Download missing data
        missing = [v for v in self.Mdict.values() if not os.path.isfile(f"{root}/{v}_dft.npz")]
        if missing: self.download(root, missing) 
        
        # Load and organized data
        datas_dict, labels_dict = self.load_dataset(root, style, self.m, task, split)
        self.datas, self.labels = self.unroll_data(datas_dict, labels_dict)
    
    def collate(self, batch):
        data = {"R":[], "z":[], "batch":[], "n":[]}
        label = {"E":[], "F":[]}      

        for i, b in enumerate(batch):
            data["R"].append(b[0])
            data["z"].append(b[1])
            data["n"].append(len(b[1]))
            data["batch"].append(torch.ones((1,len(b[1])), dtype=torch.int64)*i)
            label["E"].append(b[2])
            label["F"].append(b[3])
        data["R"] = torch.cat(data["R"])
        data["z"] = torch.cat(data["z"])
        data["n"] = torch.Tensor(data["n"])
        data["batch"] = torch.cat(data["batch"])
        label["E"] = torch.cat(label["E"])
        label["F"] = torch.cat(label["F"])
        #print("collate", data["R"].shape, data["z"].shape, data["n"].shape, data["batch"].shape, label["E"].shape, label["F"].shape)

        return data, label

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
    if args.molecule == "all": dataset = MD17Dataset(args.task, args.split)
    else: dataset = MD17SingleDataset(args.style, args.molecule, args.task, args.split)

    # Print
    # R, z, E, F = "R", "z", "E", "F" 
    # print(f"R={dataset.datas[R][0]},\t z={dataset.datas[z][0]},\t E={dataset.labels[E][0]},\t F={dataset.labels[F][0]}")
    print(len(dataset))

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=17, shuffle=True, collate_fn=dataset.collate, num_workers=16)
    for data, label in dataloader:
        print(data["R"].shape, data["z"].shape, data["n"].shape, data["batch"].shape)
        print(label["E"].shape, label["F"].shape)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", type=str, help="path to the root dir of dataset", default="./datas")
    parser.add_argument("-y", "--style", type=str, default="random", help="specify how to split the data")
    parser.add_argument("-m", "--molecule", type=str, help="lowercase initial of the molecule in the dataset or all", required=True)
    parser.add_argument("-t", "--task", type=str, help="train or valid or test", required=True)
    parser.add_argument("-s", "--split", type=int, help="the name of dataset subset, aka the number of train samples", required=True)
    parser.add_argument("-f", "--info", type=bool, help="print the information about this dataset", default=False)

    main(parser.parse_args())
