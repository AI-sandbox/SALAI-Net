import argparse
import pickle
import time
import os

from torch.utils.data import DataLoader
import torchvision
import torch
from stepsagnostic import validate
from models import AgnosticModel
from dataloaders import ReferencePanelDataset, reference_panel_collate
from stepsagnostic import build_transforms, ReshapedCrossEntropyLoss

from models.lainet import LAINetOriginal

parser = argparse.ArgumentParser()

parser.add_argument("--model-dir", type=str, default=None)

parser.add_argument("--test-mixed", type=str, default="data/benet_generations/4classes/chm22/val_128gen5/vcf_and_labels.h5")
parser.add_argument("--ref-panel", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")
parser.add_argument("--ref-panel-map", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")

parser.add_argument("-b", "--batch-size", type=int, default=16)

parser.add_argument("--smoother", type=str, choices=["anc1conv",
                                                     "none"],
                    default="anc1conv")

parser.add_argument("--win-size", type=int, default=200)
parser.add_argument("--win-stride", type=int, default=-1)
parser.add_argument("--dropout", type=float, default=-1)

parser.add_argument("--ref-pooling", type=str, choices=["maxpool", "topk"],
                    default="topk")
parser.add_argument("--topk-k", type=int, default=1)

parser.add_argument("--loss", type=str, default="BCE", choices=["BCE"])

parser.add_argument("--n-classes", type=int, default=4)

parser.add_argument("--n-refs", type=int, default=99999)


if __name__ == '__main__':

    args = parser.parse_args()

    assert (bool(args.exp))
    print("Loading args from", args.exp)
    with open("%s/args.pckl" % args.exp, "rb") as f:
        args = pickle.load(f)

    print(args)

    model = AgnosticModel(args)

    if args.model_cp:
        model.load_state_dict(torch.load(args.model_cp))

    transforms = build_transforms(args)

    test_dataset = ReferencePanelDataset(mixed_h5=args.test_mixed,
                                         reference_panel_h5=args.ref_panel,
                                         n_classes=args.n_classes,
                                         n_refs_per_class=args.n_refs,
                                         transforms=transforms)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)

    criterion = ReshapedCrossEntropyLoss()
    val_acc, val_loss = validate(model, test_loader, criterion, args)

    print("Accuracy: ", val_acc)
    print("Loss: ", val_loss)

