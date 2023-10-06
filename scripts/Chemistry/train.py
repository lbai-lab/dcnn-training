import os
import csv
import dill
import pytz
import torch
import wandb
import numpy
import random
import shutil
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from statistics import mean
from datetime import datetime
from torch.utils.data import DataLoader

from util import * 
from IAF_replace import replace_activation
from losses import EnergyLoss, AtomForceLoss, PosForceLoss
from scripts.commom_util import *

def setup_dir(args, model_path):
    os.mkdir(model_path)
    # store run data for save point loading when predicting
    with open(f"{model_path}/info.txt", 'w') as fp:
        fp.write(f"model {args.model}\n")
        fp.write(f"dataset {args.dataset}\n")
        fp.write(f"style {args.style}\n")
        fp.write(f"split {args.split}\n")
        fp.write(f"molecule {args.molecule}\n")
        fp.write(f"loss_fn {args.loss_fn}\n")
        fp.write(f"optimizer {args.optimizer}\n")
        fp.write(f"batch_size {args.batch_size}\n")
        fp.write(f"learning_rate {args.learning_rate}\n")
        fp.write(f"activation {args.activation}\n")
        #fp.write(f" {args.}\n")
    
    return  

def print_info(args):
    # print run info on screen
    if torch.cuda.is_available():
        print(f"GPU num detected: {torch.cuda.device_count()}")
        print(f"Using GPU {torch.cuda.current_device()}")
    else:
        print(f"Using CPU")


    print("INFO:")
    print(f"\tmodel: {args.model}")
    print(f"\tdataset: {args.dataset}")
    print(f"\tstyle: {args.style}")
    print(f"\tsplit: {args.split}")
    print(f"\tmolecule {args.molecule}")
    print(f"\tloss_fn: {args.loss_fn}")
    print(f"\toptimizer: {args.optimizer}")
    print(f"\tbatch_size: {args.batch_size}")
    print(f"\tlearning_rate: {args.learning_rate}")
    print(f"\tactivation: {args.activation}")
    print("**************************************************************")
    return

def wandb_setup(args, name):
    if args.wandb:
        # use wandb to monitor training, setting up its configuration 
        wandb.init(project="IAF-QC", entity="bogp", group=args.model, name=name, id=name)
        wandb.run.summary["model"] = args.model
        wandb.run.summary["DS/Sp"] = args.dataset+str(args.split)
        wandb.run.summary["molecule"] = args.molecule
        wandb.run.summary["activation"] = args.activation
    return 

def wandb_finish(args, name):
    if args.wandb:
        wandb.alert(title=f"Train finish", text=f"Run {name} traiing finished\nmodel:{args.model}\nmolecule:{args.molecule}\nactivation:{args.activation}") 
    return

def gradient_on(freeze_epoch, model, e):
    # unfreeze layers in network
    if freeze_epoch >= 0 and e >= freeze_epoch:
                for name, param in model.named_parameters():
                    if name in model.freeze_layers: param.requires_grad = True

def gradient_off(freeze_epoch, model, e):
    # freeze layers in network
    if freeze_epoch >= 0 and e >= freeze_epoch:
                for name, param in model.named_parameters():
                    if name in model.freeze_layers: param.requires_grad = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args, model_path, device):
    # initialize
    torch.manual_seed(0)
    model = get_pretrain_model(args.model).to(device) if args.pre_train else get_model(args.model).to(device)
    dataset = get_dataset(args.dataset, {"style":args.style, "molecule":args.molecule, "split":args.split}, "train", args.root)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate, num_workers=16, worker_init_fn=seed_worker)
    loss_fn = get_loss_fn(args.loss_fn)
    optimizer = get_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    scheduler = get_scheduler(optimizer, args.scheduler)
    if args.activation: replace_activation(model, args.activation, args.gamma)

    # train
    tq = tqdm(range(args.epoch))
    for e in tq:
        # epoch train
        losses, lossesE, lossesF = [], [], []
        for data, label in train_dataloader:
            data = {i:v.to(device) for i, v in data.items()}
            label = {i:v.to(device) for i, v in label.items()}
            
            # (if needed) turn gradient back on for force prediction
            gradient_on(args.freeze_epoch, model, e)

            pred = model(data)
            loss = loss_fn(pred, label, train_stat={"cur_epoch":e, "alpha":args.alpha})
            lossE = EnergyLoss(pred, label)
            lossF = AtomForceLoss(pred, label)

            # (if needed) turn gradient off for loss update
            gradient_off(args.freeze_epoch, model, e)

            # record for wandb monitoring
            losses.append(loss.to("cpu").item())
            lossesE.append(lossE.to("cpu").item())
            lossesF.append(lossF.to("cpu").item())
            tq.set_postfix({'mean_loss': mean(losses)})

            # update
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
        if args.scheduler:  scheduler.step()

        # record for wandb monitoring
        epo_loss = mean(losses)
        epo_lossE = mean(lossesE)
        epo_lossF = mean(lossesF)
        
        # save model
        if e%args.save_freqency==0 and e>args.epoch-10:
            torch.save(model, f"{model_path}/{e:03d}_{epo_loss}.pth", pickle_module=dill)  

        # log
        if args.wandb:
            log_dict = {f"{i:03d}_"+name:torch.mean(param.data) for i, (name, param) in enumerate(model.named_parameters())}
            log_dict2 = {f"{i:03d}_"+name+"_grad":torch.mean(param.grad) for i, (name, param) in enumerate(model.named_parameters()) \
                         if param.grad != None}
            log_dict.update(log_dict2)
            log_dict.update({"loss": epo_loss, "lossE": epo_lossE, 
                             "lossF": epo_lossF, "lr":optimizer.param_groups[0]["lr"]})
            wandb.log(log_dict, commit=True)    

        # break if model becomes nan
        if epo_loss != epo_loss: 
            torch.cuda.empty_cache()
            raise RuntimeError("model parameters become NaN") 

    return

def predict(args, model_path, device):
    # initialize
    dataset = get_dataset(args.dataset, {"style":args.style, "molecule":args.molecule, "split":args.split}, "test", args.root)
    identifier = dataset.identifier
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate)
    
    path = glob.glob(f"{model_path}/{args.epoch-1:03d}_*.pth")
    model = torch.load(path[0], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)
    print(f"Using model {path[0]}")
    model.eval()
    
    # test
    preds = []
    losses = []
    tq = tqdm(test_dataloader)
    for data, label in tq:
        data = {i:v.to(device) for i, v in data.items()}
        label = {i:v.to(device) for i, v in label.items()}
        pred = model(data)

        # scale the predictions back
        if args.dataset == "SMD17SingleDataset":
            pred["E"] *= dataset.C
            pred["F"] *= dataset.C
            label["E"] *= dataset.C
            label["F"] *= dataset.C
            
        preds.append( ((torch.cat((pred["E"], pred["F"].reshape(1, -1)), axis=1)).squeeze()).tolist() )
     
        loss_E = (EnergyLoss(pred, label).to("cpu").item())
        loss_F = PosForceLoss(pred, label)
        loss_F = [ (l.to("cpu").item()) for l in loss_F]
        losses.append([loss_E]+loss_F)
    
    # save result
    save_reult(model_path, identifier, preds, losses)

def metric(args, model_path):
    # For EFwT calculation, transform units to -> Energy:eV and Force: eV/A
    toEv = {"MD17SingleDataset":0.0433634, "SMD17SingleDataset":0.0433634, "RMD17SingleDataset":0.0433634}
    scale = toEv[args.dataset]

    # load and calculate
    with open(f"{model_path}/loss_{args.dataset}_{args.style}_{args.molecule}{args.split}test.csv", newline='') as fp:
        cdata = list(csv.reader(fp, quoting=csv.QUOTE_NONNUMERIC))
        cdata = [[c for c in row] for row in cdata]
        loss_Emole , loss_Fmole= [], []
        hit, total = 0, len(cdata)
        failE, failF = 0, 0

        for row in tqdm(cdata):
            loss_Emole.append(row[0]) 
            loss_Fmole.append(row[1:])
            m = max(row[1:])
            if row[0]<=0.02/scale and m<=0.03/scale: hit += 1
            else:
                if row[0]>0.02/scale: failE += 1
                if m>0.03/scale: failF += 1
        loss_Emole , loss_Fmole= np.array(loss_Emole), np.array(loss_Fmole)

        print(f"Energy MAE: {np.mean(loss_Emole):.3f},\tForce MAE: {np.mean(loss_Fmole):.3f},\tEFwT: {hit/total:.3f} (failE:{(failE/total):.3f}, failF:{(failF/total):.3f})\n")    
        if args.wandb:
            wandb.run.summary["Emae"] = f"{np.mean(loss_Emole):.3f}"
            wandb.run.summary["Fmae"] = f"{np.mean(loss_Fmole):.3f}"
            wandb.run.summary["EFwT"] = f"{hit/total:.3f}"

def main(args):
    # get strating datetime
    timezone = pytz.timezone("America/Los_Angeles")
    start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"start traing at {start_dt}")
    
    run_name = f"IAF-QC-{args.model}({start_dt})"
    model_path = f"../../models/Chemistry/{args.model}/checkpoints/{run_name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # script flow contorl
    try:
        setup_dir(args, model_path)
        print_info(args)
        wandb_setup(args, run_name)
        train(args, model_path, device)
        predict(args, model_path, device)
        metric(args, model_path)
        wandb_finish(args, run_name)
    # error handle
    except BaseException as e:
        print(e)
        traceback.print_exc()

        if args.wandb:
            wandb.alert(title=f"Train crashed", text=f"Run {run_name} traiing finished\nmodel:{args.model}\nmolecule:{args.molecule}\nactivation:{args.activation}") 

        # Decide if want to delete this run both on local and on wandb
        ans = input("Ceased. Do you want to delete checkpoint directory? [Y/N]:")
        if ans in ["Yes", "yes", "Y", "y"]:   
            if args.wandb:
                wandb.finish()          
                api = wandb.Api()
                r = api.run(f"bogp/IAF-QC/{run_name}")
                r.delete()
            shutil.rmtree(model_path)
            print(f"run {run_name} deleted")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # env
    parser.add_argument("--root", type=str, default="../../datasets/MD17/datas", help="path to the root dir of dataset")
    parser.add_argument("--seed", type=str, default="0", help="the seed for randomness")
    parser.add_argument("--pre_train", type=str, default="", help="full training with the given model path with the latest checkpoint")
    parser.add_argument("--freeze_epoch", type=int, default=-1, help="freeze backbone afterwards")
    parser.add_argument("--wandb", default=True, help="If want to use wandb")
    parser.add_argument("--no_wandb", action='store_false', dest="wandb", help="If don't want to use wandb")

    
    # configuration
    parser.add_argument("-M", "--model", type=str, default="schnet", help="the model to be trained")
    parser.add_argument("-D", "--dataset", type=str, default="MD17SingleDataset", help="the dataset to be used")
    parser.add_argument("-Y", "--style", type=str, default="random", help="specify how to split the data")
    parser.add_argument("-P", "--split", type=int, default=50000, help="the name of dataset subset, aka the number of train samples")
    parser.add_argument("-m", "--molecule", default="a", type=str, help="lowercase initial of the molecule in the dataset")
    parser.add_argument("-L", "--loss_fn", type=str, default="EnergyForceLoss", help="the loss fn to be used")
    parser.add_argument("-O", "--optimizer", type=str, default="Adam", help="the optimizer to be used")
    parser.add_argument("-S", "--scheduler", type=str, default="", help="the scheduler to be used")
    parser.add_argument("-A", "--activation", type=str, default="", help="replace all activation layers of model by this")

    # training para
    parser.add_argument("-e", "--epoch", type=int, default=50, help="number of epoch to train")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="batch size to train")
    parser.add_argument("-s", "--save_freqency", type=int, default=1, help="freqency to save result")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001, help="learning rate to train")
    parser.add_argument("-a", "--alpha", type=float, default=30, help="alpha for force in loss function")
    parser.add_argument("-g", "--gamma", type=float, default=0.005, help="gamma for the scalar of Leaky RELU and ELU")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay to train")

    main(parser.parse_args())