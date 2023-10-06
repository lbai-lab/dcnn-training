import os
import random
import pickle
import shutil
import argparse

def random_idx(train_num, valid_id_num, valid_ood_num, train, valid):
    pool = [i for i in range(train_num)]
    train_set = random.sample(pool, train)
    
    pool = [i for i in range(valid_id_num)]
    valid_id_set = random.sample(pool, valid)

    pool = [i for i in range(valid_ood_num)]
    valid_ood_set = random.sample(pool, valid)
    
    return train_set, valid_id_set, valid_ood_set

def main(args):
    train_num = 8225293
    valid_ood_num = 450669
    valid_id_num =  394727

    train = args.train
    valid = train//8

    if train > train_num:
        print(f"train smaple not enough")
        return
    if valid > valid_id_num or valid > valid_ood_num:
        print(f"valid smaple not enough")
        return

    if not os.path.isdir(f"random"): os.mkdir(f"random")
    if os.path.isdir(f"random/{train}"): 
        print("split exist, remove and try again")
        return
    os.mkdir(f"random/{train}")

    train_set, valid_id_set, valid_ood_set = random_idx(train_num, valid_id_num, valid_ood_num, train, valid)
     
    with open(f"random/{train}/info.txt", "a+") as f:
        f.write(f"train:{len(train_set)} valid_id:{len(valid_id_set)} valid_ood:{len(valid_ood_set)} total:{len(train_set)+len(valid_id_set)+len(valid_ood_set)}\n")

    with open(f"random/{train}/train.pkl", "wb") as f:
        pickle.dump(train_set, f)
    with open(f"random/{train}/val_id.pkl", "wb") as f:
        pickle.dump(valid_id_set, f)
    with open(f"random/{train}/val_ood.pkl", "wb") as f:
        pickle.dump(valid_ood_set, f)

    print("split generated")
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200000, help="number of sample in training set (in percent if <100, in real number otherwise)")


    main(parser.parse_args())