import argparse
import pickle
import time
import os

import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torch
from stepsagnostic import inference
from models import AgnosticModel
from dataloaders import ReferencePanelDataset, reference_panel_collate
from stepsagnostic import build_transforms, ReshapedCrossEntropyLoss

parser = argparse.ArgumentParser()

parser.add_argument("--model-cp", type=str, default=None)
parser.add_argument("--model-args", type=str, default=None)

parser.add_argument("--test-mixed", type=str, default=False)
parser.add_argument("--ref-panel", type=str, default=False)

parser.add_argument("--query", '-q', default=False)
parser.add_argument("--reference", '-r', default=False)
parser.add_argument("--map", '-m', default=False)

parser.add_argument('--out-folder', '-o', default='outputs/default')

parser.add_argument("-b", "--batch-size", type=int, default=16)

if __name__ == '__main__':

    args = parser.parse_args()

    if os.path.isdir(args.out_folder):
        raise Exception("Experiment name " + args.out_folder + " already exists.")


    print(args)
    if not args.model_args:
        args.model_args = args.model_cp.replace('models/best_model.pth', 'args.pckl')
    with open(args.model_args, "rb") as f:
        model_args = pickle.load(f)

    model = AgnosticModel(model_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_cp, map_location=device))

    transforms = build_transforms(model_args)

    if args.test_mixed:
        mixed_file_path = args.test_mixed
    elif args.query:
        mixed_file_path = args.query

    test_dataset = ReferencePanelDataset(mixed_file_path=mixed_file_path,
                                         reference_panel_h5=args.ref_panel,
                                         reference_panel_vcf=args.reference,
                                         reference_panel_map=args.map,
                                         n_refs_per_class=model_args.n_refs,
                                         transforms=transforms)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate, shuffle=False)

    criterion = ReshapedCrossEntropyLoss()
    predicted_classes, ibd = inference(model, test_loader, args)

    os.mkdir(args.out_folder)

    np.save(args.out_folder + '/ancestry_prediction', predicted_classes.cpu().numpy())
    np.save(args.out_folder + '/descendant_prediction', ibd.cpu())
