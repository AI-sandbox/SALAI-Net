import argparse
import pickle
import time
import os

from torch.utils.data import DataLoader
import torchvision

from stepsagnostic import train
from models import AgnosticModel
from dataloaders import ReferencePanelDataset, reference_panel_collate
from stepsagnostic import build_transforms

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="exp/default_exp")

parser.add_argument("--train-mixed", type=str, default="data/benet_generations/4classes/chm22/train1_128gen20/vcf_and_labels.h5")
parser.add_argument("--valid-mixed", type=str, default="data/benet_generations/4classes/chm22/val_128gen5/vcf_and_labels.h5")
parser.add_argument("--train-ref-panel", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")
parser.add_argument("--valid-ref-panel", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")

parser.add_argument("--query", '-q', default=False)
parser.add_argument("--reference", '-r', default=False)
parser.add_argument("--map", '-m', default=False)

# parser.add_argument("--train-mixed", type=str, default="data/benet_generations/chm22/train_8gens1k/vcf_and_labels.h5")
# parser.add_argument("--valid-mixed", type=str, default="data/benet_generations/chm22/val_8gens100/vcf_and_labels.h5")
# parser.add_argument("--ref-panel", type=str, default="data/benet_generations/chm22/train2_0gens/vcf_and_labels.h5")
#
# parser.add_argument("--model", type=str, choices=["VanillaConvNet",
#                                                   "LAINet"],
#                     default="VanillaConvNet")
#                     default="VanillaConvNet")

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--update-every", type=int, default=1)

parser.add_argument("--smoother", type=str, choices=["anc1conv", "none"],
                    default="anc1conv")

parser.add_argument("--ref-pooling", type=str, choices=["maxpool", "topk", "average",
                                                        "baggingmaxpool"],
                    default="topk")

parser.add_argument("--topk-k", type=int, default=1)
# parser.add_argument("--bagging-k", type=int, default=10)
# parser.add_argument("--bagging-split", type=float, default=0.2)


parser.add_argument("--win-size", type=int, default=200)
parser.add_argument("--win-stride", type=int, default=-1)

parser.add_argument("--dropout", type=float, default=-1)

parser.add_argument("--loss", type=str, default="BCE", choices=["BCE"])

parser.add_argument("--resume", dest="resume", action='store_true')

parser.add_argument("--n-refs", type=int, default=99999)

parser.add_argument("--comment", type=str, default=None)


if __name__ == '__main__':

    args = parser.parse_args()

    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True
    print(args)

    model = AgnosticModel(args)

    transforms = build_transforms(args)
    print("Loading train data")
    train_dataset = ReferencePanelDataset(mixed_file_path=args.train_mixed,
                                          reference_panel_h5=args.train_ref_panel,
                                          reference_panel_vcf=args.reference,
                                          reference_panel_map=args.map,
                                          n_refs_per_class=args.n_refs,
                                          transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)

    print("Loading validation data")
    valid_dataset = ReferencePanelDataset(mixed_file_path=args.valid_mixed,
                                          reference_panel_h5=args.valid_ref_panel,
                                          reference_panel_vcf=args.reference,
                                          reference_panel_map=args.map,
                                          n_refs_per_class=args.n_refs,
                                          transforms=transforms)

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)

    if not args.resume:
        if os.path.isdir(args.exp):
            raise Exception("Experiment name " + args.exp +" already exists.")
        os.mkdir(args.exp)
        os.mkdir(args.exp + "/models")

    with open(args.exp + "/args.pckl", "wb") as f:
        pickle.dump(args, f)
    train(model, train_loader, valid_loader, args)

