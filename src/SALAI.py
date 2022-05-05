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
from stepsagnostic import build_transforms, ReshapedCrossEntropyLoss, get_meta_data, write_msp_tsv, msp_to_lai

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

    info = test_dataset.info

    criterion = ReshapedCrossEntropyLoss()
    predicted_classes, predicted_classes_window, ibd = inference(model, test_loader, args)

    os.mkdir(args.out_folder)

    np.save(args.out_folder + '/ancestry_prediction', predicted_classes.cpu().numpy().astype(int))
    np.save(args.out_folder + '/descendant_prediction', ibd.cpu())

    chm = info['chm'][0]
    pos = info['pos']
    n_seq, n_wind = predicted_classes_window.shape
    wind_size = model.base_model.window_size

    predicted_classes_window = predicted_classes_window.cpu().numpy()

    query_samples = np.array(test_dataset.samples_list)
    populations = np.array(test_dataset.ancestry_names)

    np.save(args.out_folder + '/population_ids', populations)
    np.save(args.out_folder + '/founder_ids', query_samples)

    meta = get_meta_data(chm, pos, pos, n_wind, wind_size)
    write_msp_tsv(args.out_folder, meta, predicted_classes_window, populations, query_samples)
    msp_to_lai(args.out_folder + "/predictions.msp.tsv", pos, lai_file=args.out_folder + "/predictions.lai")