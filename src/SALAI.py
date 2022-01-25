import argparse

from torch.utils.data import DataLoader
import torch
from stepsagnostic import inference
from models import AgnosticModel
from dataloaders import ReferencePanelDataset, reference_panel_collate
from stepsagnostic import build_transforms, ReshapedCrossEntropyLoss

parser = argparse.ArgumentParser()

parser.add_argument("--model-cp", type=str, default=None)

parser.add_argument("--test-mixed", type=str, default=False)
parser.add_argument("--ref-panel", type=str, default=False)

parser.add_argument("--query", '-q', default=False)
parser.add_argument("--reference", '-r', default=False)
parser.add_argument("--map", '-m', default=False)

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
    print(args)

    model = AgnosticModel(args)

    if args.model_cp:
        model.load_state_dict(torch.load(args.model_cp))

    transforms = build_transforms(args)

    test_dataset = ReferencePanelDataset(mixed_h5=args.test_mixed,
                                         query=args.query,
                                         reference_panel_h5=args.ref_panel,
                                         reference_panel_vcf=args.reference,
                                         reference_panel_map=args.map,
                                         n_classes=args.n_classes,
                                         n_refs_per_class=args.n_refs,
                                         transforms=transforms)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)

    criterion = ReshapedCrossEntropyLoss()
    val_acc, val_loss = inference(model, test_loader, args)

    print("Accuracy: ", val_acc)
    print("Loss: ", val_loss)
