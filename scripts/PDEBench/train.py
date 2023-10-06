import os
import pytz
import wandb
import torch
import shutil
import argparse
import traceback
import numpy as np
import deepxde as dde

from models.PDEBench.pde_metric import metric_func
from models.PDEBench.setup_pde import *
from scripts.PDEBench.util import get_activation
from datetime import datetime

# Callback class to record training losses
class WandbRecord(dde.callbacks.Callback):
    def __init__(self, loss_terms, wandb_freq):
        super().__init__()
        self.loss_terms = loss_terms
        self.wandb_freq = wandb_freq
        self.epoch = 0
        return
    
    def on_epoch_end(self):
        if (self.epoch+1) % self.wandb_freq == 0:
            loss_dict = {"train_"+lt:self.model.train_state.loss_train[i] for i, lt in enumerate(self.loss_terms)}
            loss_dict2 =  {"test_"+lt:self.model.train_state.loss_test[i] for i, lt in enumerate(self.loss_terms)}
            loss_dict.update(loss_dict2)
            wandb.log(loss_dict, commit=True)  
        self.epoch += 1
        return

def print_info(args):
    # print run info on screen
    if torch.cuda.is_available():
        print(f"GPU num detected: {torch.cuda.device_count()}")
        print(f"Using GPU {torch.cuda.current_device()}")
    else:
        print(f"Using CPU")

    print("INFO:")
    print(f"\tpinn: {args.model}")
    print(f"\tactivation: {args.activation}")
    print("**************************************************************")
    return
    
def wandb_setup(args, name):
    if args.wandb:
        wandb.init(project="PDEB", entity="bogp", group="IR-"+args.model, name=name, id=name, resume="allow")    
        wandb.run.summary["activation"] = args.activation
    return 

def wandb_finish(args, name):
    if args.wandb:
        wandb.alert(title=f"Train finish", text=f"Run {name} traiing finished\nmodel:{args.model}\nact:{args.activation}") 
    return

def _run_training(model_name, scenario, epochs, learning_rate, model_update, flnm,
                  input_ch, output_ch, root_path, val_batch_idx, if_periodic_bc, 
                  aux_params, seed, 
                  run_name, loss_terms, act, wandb_log, wandb_freq, backbone, batchnorm):
    
    if scenario == "swe2d":
        model, dataset = setup_swe_2d(filename=flnm, seed=seed, act=act, backbone=backbone, batchnorm=batchnorm, root_path=root_path)
        n_components = 1
    elif scenario == "diff-react":
        model, dataset = setup_diffusion_reaction(filename=flnm, seed=seed, act=act, backbone=backbone, batchnorm=batchnorm, root_path=root_path)
        n_components = 2
    elif scenario == "diff-sorp":
        model, dataset = setup_diffusion_sorption(filename=flnm, seed=seed, act=act, backbone=backbone, batchnorm=batchnorm, root_path=root_path)
        n_components = 1
    elif scenario == "pde1D":
        model, dataset = setup_pde1D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     if_periodic_bc=if_periodic_bc,
                                     aux_params=aux_params,
                                     act=act, 
                                     backbone=backbone,
                                     batchnorm=batchnorm)
        
        if flnm.split('_')[1][0] == 'C': n_components = 3
        else: n_components = 1
    elif scenario == "CFD2D":
        model, dataset = setup_CFD2D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params, 
                                     backbone=backbone)
        n_components = 4
    elif  scenario == "CFD3D":
        model, dataset = setup_CFD3D(filename=flnm,
                                     root_path=root_path,
                                     input_ch=input_ch,
                                     output_ch=output_ch,
                                     val_batch_idx=val_batch_idx,
                                     aux_params=aux_params, 
                                     backbone=backbone)
        n_components = 5
    else:
        raise NotImplementedError(f"PINN training not implemented for {scenario}")

    checker = dde.callbacks.ModelCheckpoint(
        f"../../models/PDEBench/{model_name}/checkpoints/{run_name}/{run_name}.pt", save_better_only=True, period=5000
    )

    # add orignial callback and custom callbacks
    callbacks =  [checker]
    if wandb_log: callbacks += [WandbRecord(loss_terms, wandb_freq)]

 
    optimizer ="adam"
    model.compile(optimizer, lr=learning_rate)
    losshistory, train_state = model.train(
        epochs=epochs, display_every=model_update, callbacks=callbacks
    )

    test_input, test_gt = dataset.get_test_data(
        n_last_time_steps=20, n_components=n_components
    )
    print("test_input",test_input.shape)
    print("test_gt",test_gt.shape)

    # select only n_components of output
    # dirty hack for swe2d where we predict more components than we have data on
    test_pred = model.predict(test_input.cpu())
    test_pred = torch.tensor(test_pred[:, :n_components])
    print("test_pred",test_pred.shape)

    # prepare data for metrics eval
    test_pred = dataset.unravel_tensor(
        test_pred, n_last_time_steps=20, n_components=n_components
    )
    test_gt = dataset.unravel_tensor(
        test_gt, n_last_time_steps=20, n_components=n_components
    )
    errs = metric_func(test_pred, test_gt)
    errors = [np.array(err.cpu()) for err in errs]

    metrics = ["err_RMSE", "err_nRMSE", "err_CSV", "err_Max", "err_BD", "err_F"]
    for i in range(2):
        print(f"{metrics[i]}:{errors[i].item()}")
    if wandb_log: wandb.log({"RMSE":errors[i].item(), "NN":backbone}, commit=True) 
    return train_state.loss_test

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    
    # env
    parser.add_argument("--wandb", default=True, help="If want to use wandb")
    parser.add_argument("--no_wandb", action='store_false', dest="wandb", help="If don't want to use wandb")
    
    # configuration
    parser.add_argument("-M", "--model", type=str, default="Advection1D", help="the pinn (scenario) to train")
    parser.add_argument("-B", "--backbone", type=str, default="40-6", help="the size of backbone NN to use")
    parser.add_argument("-S", "--scheduler", type=str, default="", help="the scheduler to be used")
    parser.add_argument("-A", "--activation", type=str, default="", help="replace all activation layers of model by this")
    parser.add_argument("-R", "--repeat", type=int, default=1, help="repeat training several times to get error bar")
    parser.add_argument("--no_batchnorm", default=True, help="if we do not add batch normailzation layer in backbone NN")
    parser.add_argument("--batchnorm", action='store_false', dest="no_batchnorm", help="if we add batch normailzation layer in backbone NN")

    # training para
    parser.add_argument("-n", "--npow", type=float, default=2, help="power of activation")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay to train")

    args = parser.parse_args()
    #########################################################################################################################################

    # get strating datetime
    timezone = pytz.timezone("America/Los_Angeles")
    start_dt = (datetime.now().astimezone(timezone)).strftime("%Y-%m-%d_%H-%M-%S")
    print(f"start traing at {start_dt}")
    
    # set up 
    run_name = f"PDEB-{args.model}({start_dt})"
    model_path = f"../../models/PDEBench/{args.model}/checkpoints/{run_name}"
    os.mkdir(model_path)

    # get IAF if needed
    if args.activation:
        act  = get_activation(args.activation, args.npow)
    else:
        act = None

    # setting configurations for each PINN
    if args.model == "Advection1D":
        model = "Advection1D"
        scenario = "pde1D"
        epochs = 15000
        learning_rate = 1.e-3
        model_update = 500
        flnm = "1D_Advection_Sols_beta0.1.hdf5"
        input_ch = 2
        output_ch = 1
        root_path = "/usr/data1/PDEBench/pdebench/data/1D/Advection"
        val_num = 10
        if_periodic_bc = True
        aux_params = [0.1]
        loss_terms = ["f", "ic", "bc"]
        wandb_freq = 100
        
    elif args.model == "CFD1D":
        model = "CFD1D"
        scenario = "pde1D"
        epochs = 15000
        learning_rate = 1.e-3
        model_update = 500
        flnm = "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5"
        input_ch = 2
        output_ch = 3
        root_path = "/usr/data1/PDEBench/pdebench/data/1D/CDF"
        val_num = 10
        if_periodic_bc = True
        aux_params = [1.6666666667]
        loss_terms = ["f", "ic_data_d", "ic_data_v", "ic_data_p", "bc_D", "bc_V", "bc_P"]
        wandb_freq = 100

    elif args.model == "DiffReact2D":
        model = "DiffReact2D"
        scenario = "diff-react"
        epochs = 100
        learning_rate = 1.e-3
        model_update = 500
        flnm = "2D_diff-react_NA_NA.h5"
        input_ch = 0
        output_ch = 1
        root_path = "/usr/data1/PDEBench/pdebench/data/2D/ReactionDiffusion"
        val_num = 1
        if_periodic_bc = False
        aux_params = 0
        loss_terms = ["f", "bc", "ic_data_u", "ic_data_v", "bc_data_u", "bc_data_v"]
        wandb_freq = 1

    elif args.model == "DiffSorp1D":
        model = "DiffSorp1D"
        scenario = "diff-sorp"
        epochs = 15000
        learning_rate = 1.e-3
        model_update = 500
        flnm = "1D_diff-sorp_NA_NA.h5"
        input_ch = 0
        output_ch = 1
        root_path = "/usr/data1/PDEBench/pdebench/data/1D/diffusion-sorption"
        val_num = 1
        if_periodic_bc = False
        aux_params = 0
        loss_terms = ["f", "ic", "bc_d", "bc_d2", "bc_data"]  
        wandb_freq = 100
    
    elif args.model == "Swe2D":
        model = "Swe2D"
        scenario = "swe2d"
        epochs = 15000
        learning_rate = 1.e-3
        model_update = 500
        flnm = "2D_rdb_NA_NA.h5"
        input_ch = 0
        output_ch = 1
        root_path = "/usr/data1/PDEBench/pdebench/data/2D/SWE"
        val_num = 1
        if_periodic_bc = False
        aux_params = 0
        loss_terms = ["f", "bc", "ic_h", "ic_u", "ic_v", "bc_data"]  
        wandb_freq = 100
    
    else:
        raise NotImplementedError(f"Unexpected behaiver in {os.path.basename(__file__)}: args.model switch case error")

    # Start of script
    try:
        losses = []
        print_info(args)
        wandb_setup(args, run_name)
        
        # repeat several time to get error bar, default only 1 time
        for i in range(args.repeat):
            test_loss = _run_training(model, scenario, epochs, learning_rate, model_update, flnm,
                                    input_ch, output_ch,
                                    root_path, -val_num, if_periodic_bc, aux_params, f"{i:04d}", 
                                    run_name, loss_terms, act, args.wandb, wandb_freq, args.backbone, not args.no_batchnorm)
            losses.append(test_loss)

        # calculate mean and std
        if args.repeat > 1:
            losses = np.stack(losses)
            losses_mean = np.mean(losses, axis=0)
            losses_std = np.std(losses, axis=0)
            print(f"mean: {losses_mean}")
            print(f"std:  {losses_std}")
            if args.wandb:
                wandb.run.summary["bar_mean"] = losses_mean
                wandb.run.summary["bar_std"] = losses_std
        wandb_finish(args, run_name)
    
    # error handle
    except BaseException as e:
        print(e)
        traceback.print_exc()

        if args.wandb:
            wandb.alert(title=f"Train crashed", text=f"Run {run_name} traiing finished\nmodel:{args.model}\nact:{args.activation}")

        # Decide if want to delete this run both on local and on wandb
        ans = input("Ceased. Do you want to delete checkpoint directory? [Y/N]:")
        if ans in ["Yes", "yes", "Y", "y"]:   
            if args.wandb:
                wandb.finish()          
                api = wandb.Api()
                r = api.run(f"bogp/IAF-PDEB/IR-{run_name}")
                r.delete()
            shutil.rmtree(model_path)
            print(f"run {run_name} deleted")