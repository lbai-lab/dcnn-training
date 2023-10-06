import os
import random
import pickle
import shutil
import argparse

def random_idx(all, train):
    pool = [i for i in range(all)]
    train_set = random.sample(pool, train)
    pool = list(set(pool)-set(train_set))
    valid_set = random.sample(pool, train//8)
    pool = list(set(pool)-set(valid_set))
    test_set = random.sample(pool, train//8)

    return train_set, valid_set, test_set

def random_percent_idx(all, train):
    train_num = int(all*(train/100))

    pool = [i for i in range(all)]
    train_set = random.sample(pool, train_num)
    pool = list(set(pool)-set(train_set))
    valid_set = random.sample(pool, len(pool)//2)
    test_set = list(set(pool)-set(valid_set))

    return train_set, valid_set, test_set

def trajectory_idx(all, train):
    assert False
    train_set = [i for i in range(train)]
    valid_set = [i for i in range(train, train+valid)]
    test_set =  [i for i in range(train+valid, all)]

    return train_set, valid_set, test_set

def trajectory_percent_idx(all, train):
    assert False
    train_num = int(all*(train/100))
    valid_num = int(all*(valid/100))

    train_set = [i for i in range(train_num)]
    valid_set = [i for i in range(train_num, train_num+valid_num)]
    test_set =  [i for i in range(train_num+valid_num, all)]

    return train_set, valid_set, test_set

def main(args):
    sample_num = {"benzene":627983, "uracil":133770, "naphthalene":326250, "aspirin":211762, "salicylic":320231, "malonaldehyde":993237, "ethanol":555092, "toluene":442790}

    if not os.path.isdir(f"{args.style}"): os.mkdir(f"{args.style}")
    if os.path.isdir(f"{args.style}/{args.train}"): 
        print("split exist, remove and try again")
        return
    os.mkdir(f"{args.style}/{args.train}")

    for m in sample_num.keys():
        if args.train*1.25 > sample_num[m]:
            print(f"{m} smaple not enough")
            continue

        os.mkdir(f"{args.style}/{args.train}/{m}")
        if args.train >= 100:
            if args.style == "random": train_set, valid_set, test_set = random_idx(sample_num[m], args.train)
            elif args.style == "trajectory": train_set, valid_set, test_set = trajectory_idx(sample_num[m], args.train)
            else: train_set, valid_set, test_set = [], [], []
        else:
            if args.style == "random": train_set, valid_set, test_set = random_percent_idx(sample_num[m], args.train)
            elif args.style == "trajectory": train_set, valid_set, test_set = trajectory_percent_idx(sample_num[m], args.train)
            else: train_set, valid_set, test_set = [], [], []

        with open(f"{args.style}/{args.train}/info.txt", "a+") as f:
            f.write(f"{m} - train:{len(train_set)} valid:{len(valid_set)} test:{len(test_set)} total:{len(train_set)+len(valid_set)+len(test_set)}\n")

        with open(f"{args.style}/{args.train}/{m}/train.pkl", "wb") as f:
            pickle.dump(train_set, f)
        with open(f"{args.style}/{args.train}/{m}/valid.pkl", "wb") as f:
            pickle.dump(valid_set, f)
        with open(f"{args.style}/{args.train}/{m}/test.pkl", "wb") as f:
            pickle.dump(test_set, f)

    print("split generated")
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="random", help="specify how to split the data")
    parser.add_argument("--train", type=int, default=80, help="number of sample in training set (in percent if <100, in real number otherwise)")


    main(parser.parse_args())