import argparse
import pickle
import time
import os

from torch.utils.data import DataLoader
import torchvision

from steps import train
from models import DevModel, VanillaLAINet
from dataloaders import GenomeDataset

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="exp/default_exp")

parser.add_argument("--train-data", type=str, default="data/chm20/test.npz")
parser.add_argument("--valid-data", type=str, default="data/chm20/test.npz")

parser.add_argument("--model", type=str, choices=["kk"],
                    default="kk")

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--loss", type=str, default="BCE", choices=["BCE"])

parser.add_argument("--resume", dest="resume", action='store_true')

if __name__ == '__main__':

    args = parser.parse_args()

    print(args)
    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True

    transforms = []

    transforms = torchvision.transforms.Compose(transforms)


    train_dataset = GenomeDataset(data=args.train_data, transforms=transforms)
    valid_dataset = GenomeDataset(data=args.valid_data, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # t = time.time()
    # for x in train_loader:
    #     pass
    # print("loop time", time.time() - t)

    if args.model == "kk":
        model = VanillaLAINet(7)


    if not args.resume:
        if os.path.isdir(args.exp):
            raise Exception("Experiment name " + args.exp +" already exists.")
        os.mkdir(args.exp)
        os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    train(model, train_loader, valid_loader, args)