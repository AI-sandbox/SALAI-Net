import argparse
import pickle
import time
import os

from torch.utils.data import DataLoader
import torchvision

from stepsagnostic import train
from models import AgnosticConvModel
from dataloaders import ReferencePanelDataset, reference_panel_collate
from stepsagnostic import build_transforms

from models.lainet import LAINetOriginal

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="exp/default_exp")

parser.add_argument("--train-mixed", type=str, default="data/benet_generations/train_8gens1k/vcf_and_labels.h5")
parser.add_argument("--valid-mixed", type=str, default="data/benet_generations/val_8gen100/vcf_and_labels.h5")
parser.add_argument("--ref-panel", type=str, default="data/benet_generations/train2_0gen/vcf_and_labels.h5")

parser.add_argument("--model", type=str, choices=["VanillaConvNet",
                                                  "LAINet"],
                    default="VanillaConvNet")

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--smoother", type=str, choices=["1conv",
                                                     "2conv",
                                                     "1TransfEnc"],
                    default="1conv")
parser.add_argument("--pos-emb", type=str, choices=["linpos",
                                                    "trained1",
                                                    "trained2",
                                                    "trained3",
                                                    "trained1dim4"],
                    default=None)
parser.add_argument("--transf-emb", dest="transf_emb", action='store_true')

parser.add_argument("--win-size", type=int, default=400)

parser.add_argument("--loss", type=str, default="BCE", choices=["BCE"])

parser.add_argument("--resume", dest="resume", action='store_true')


parser.add_argument("--seq-len", type=int, default=317408)
parser.add_argument("--n-classes", type=int, default=7)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True
    print(args)

    # if args.model == "VanillaConvNet":
    #     model = VanillaConvNet(args)
    #     # model = DevModel(7)
    #
    # elif args.model == "LAINet":
    #     model = LAINetOriginal(args.seq_len, args.n_classes,
    #                            window_size=args.win_size, is_haploid=True)
    # else:
    #     raise ValueError()

    model = AgnosticConvModel(args)

    transforms = build_transforms(args)

    train_dataset = ReferencePanelDataset(mixed_h5=args.train_mixed,
                                          reference_panel_h5=args.ref_panel,
                                          transforms=transforms)
    valid_dataset = ReferencePanelDataset(mixed_h5=args.valid_mixed,
                                          reference_panel_h5=args.ref_panel,
                                          transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)

    # t = time.time()
    # for x in train_loader:
    #     pass
    # print("loop time", time.time() - t)



    if not args.resume:
        if os.path.isdir(args.exp):
            raise Exception("Experiment name " + args.exp +" already exists.")
        os.mkdir(args.exp)
        os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)

    train(model, train_loader, valid_loader, args)

