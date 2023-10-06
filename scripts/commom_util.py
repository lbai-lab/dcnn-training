import torch
import glob
import dill
import csv

def get_pretrain_model(args):
    # load pretained checkpoint
    models = glob.glob(f"{args.pre_train}/*.pth")
    models = sorted(models, key=lambda x: float(x.split("/")[-1][4:-4]))
    args.start_epoch = 0
    print(f"Using model {models[0]}")

    return torch.load(models[0], map_location=torch.device(torch.cuda.current_device()), pickle_module=dill)

def save_reult(model_path, identifier, preds, losses):
    # save predictions
    if preds:
        csvfile = open(f"{model_path}/pred_{identifier}.csv", "w", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(preds)

    if losses:
        csvfile = open(f"{model_path}/loss_{identifier}.csv", "w", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(losses)
    return