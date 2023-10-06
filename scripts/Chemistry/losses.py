import torch

# Total Loss (E+F) ########################################################################################################
def EnergyForceLoss(pred, label, train_stat):
    alpha = train_stat["alpha"]
    E = EnergyLoss(pred, label)
    F = AtomForceLoss(pred, label)
    return E + alpha*F

# Enery Loss ########################################################################################################
def EnergyLoss(pred, label):
    p, l = pred["E"].squeeze(), label["E"].squeeze()
    mae = torch.nn.L1Loss()
    return mae(p, l)

# Force Loss ########################################################################################################
# not for training, only for predicting.
def OcpForceLoss(pred, label):
    p, l = pred["F"], label["F"]
    p = p.reshape(-1, 1)
    l = l.reshape(-1, 1)

    mae = torch.nn.L1Loss()
    loss = [mae(p[i], l[i]) for i in range(len(p))]
    return loss

# not for training, only for predicting.
def PosForceLoss(pred, label):
    p, l = pred["F"], label["F"]
    batch_size = l.shape[0]   
    p = p.reshape(batch_size, -1).t()
    l = l.reshape(batch_size, -1).t()

    mae = torch.nn.L1Loss()
    loss = [mae(p[i], l[i]) for i in range(len(p))]
    return loss

# force loss for training
def AtomForceLoss(pred, label):
    p, l = pred["F"].squeeze(), label["F"].squeeze()
    p = p.reshape((-1, 3))
    l = l.reshape((-1, 3))
    mae = torch.nn.L1Loss()
    f = mae(p, l)
    return f


