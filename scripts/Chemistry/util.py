import torch

# import loss_fn
from scripts.Chemistry.losses import *

# import models
from models.Chemistry.schnet.schnet import SchNetWrap
from models.Chemistry.cgcnn.cgcnn import CGCNN
from models.Chemistry.cgcnn_no_bn.cgcnn_no_bn import CGCNN_NO_BN
from models.Chemistry.dimenet_plus_plus.dimenet_plus_plus import DimeNetPlusPlusWrap
from models.Chemistry.gemnet.gemnet import GemNetT
from models.Chemistry.forcenet.forcenet import ForceNet
from models.Chemistry.forcenet_no_bn.forcenet_no_bn import ForceNet_NO_BN 

# import datasets
from datasets.MD17.MD17Dataset import MD17Dataset, MD17SingleDataset
from datasets.SMD17.SMD17Dataset import SMD17SingleDataset
from datasets.OC22.OC22Dataset import OC22LmdbDataset
from datasets.SOC22.SOC22Dataset import SOC22LmdbDataset

def get_model(model):
    if model == "schnet": return SchNetWrap(regress_forces=True)
    if model == "cgcnn": return CGCNN(regress_forces=True)
    if model == "cgcnn_no_bn": return CGCNN_NO_BN(regress_forces=True)
    if model == "dimenet_plus_plus": return DimeNetPlusPlusWrap(regress_forces=True)
    if model == "gemnet": return GemNetT(regress_forces=True)
    if model == "forcenet": return ForceNet(regress_forces=True)
    if model == "forcenet_no_bn": return ForceNet_NO_BN(regress_forces=True)

    raise ValueError("model name incorrect")

def get_dataset(dataset, dataset_stat, task, root):
    if dataset == "MD17SingleDataset": return MD17SingleDataset(dataset_stat["style"], dataset_stat["molecule"], task, dataset_stat["split"], root)
    if dataset == "MD17Dataset": return MD17Dataset(dataset_stat["style"], dataset_stat["molecule"], task, dataset_stat["split"], root)
    if dataset == "SMD17SingleDataset": return SMD17SingleDataset(dataset_stat["style"], dataset_stat["molecule"], task, dataset_stat["split"], root)
    if dataset == "OC22LmdbDataset": return OC22LmdbDataset(task, root)
    if dataset == "SOC22LmdbDataset": return SOC22LmdbDataset(task, root)

    raise ValueError("dataset name incorrect")

def get_loss_fn(loss_fn):
    # baseline losses
    if loss_fn == "EnergyForceLoss": return EnergyForceLoss
    if loss_fn == "EnergyLoss": return EnergyLoss
    if loss_fn == "AtomForceLoss": return AtomForceLoss

    raise ValueError("loss function name incorrect")

def get_optimizer(model, optimizer, lr, wd):
    parameters = model.parameters()
    if optimizer == "Adam": return torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    if optimizer == "SGD": return torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)

    raise ValueError("optimizer name incorrect")

def get_scheduler(optimizer, scheduler):
    if scheduler == "": return None
    if scheduler == "L0530": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch//5), last_epoch=-1)
    if scheduler == "L0630": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.6**(epoch//5), last_epoch=-1)
    if scheduler == "L0730": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.7**(epoch//5), last_epoch=-1)
    if scheduler == "L0830": return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.8**(epoch//5), last_epoch=-1)

    raise ValueError("scheduler name incorrect")



