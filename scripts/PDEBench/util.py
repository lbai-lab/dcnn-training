import csv
import glob
import dill
import torch
import argparse

# import IAFs
from scripts.IAFs import *

def get_scheduler(optimizer, args):
    if args.scheduler == "": return None
    if args.scheduler == "L0530": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch//30), last_epoch=-1)
    if args.scheduler == "L0630": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.6**(epoch//30), last_epoch=-1)
    if args.scheduler == "L0730": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.7**(epoch//30), last_epoch=-1)
    if args.scheduler == "L0830": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.8**(epoch//30), last_epoch=-1)

    raise ValueError("scheduler name incorrect")

def get_activation(activation, n):
    # get IAF, only relu_int is used for now
    if activation == "int_relu": return relu_int(n)
    if activation == "int_leaky_relu": return leaky_relu_int(n)
    if activation == "int_elu": return elu_int(n)

    raise ValueError("activation name incorrect")

if __name__ == "__main__":
    # Cloud edit this part to test something
    print("testing")
