import argparse
import pickle
import time
import os

from torch.utils.data import DataLoader
import torchvision

from stepsagnostic import train
from models import AgnosticModel, MultisizeAgnosticModel
from dataloaders import ReferencePanelDataset, reference_panel_collate, ReferencePanelMultiChmDataset, get_num_samples_per_chromosome, SameChmSampler
from stepsagnostic import build_transforms

from models.lainet import LAINetOriginal

parser = argparse.ArgumentParser()

parser.add_argument("--exp", type=str, default="exp/default_exp")


parser.add_argument("--train-mixed", type=str, default="data/benet_generations/4classes/chm22/train1_8gens600/vcf_and_labels.h5")
parser.add_argument("--valid-mixed", type=str, default="data/benet_generations/4classes/chm22/val_8gen200/vcf_and_labels.h5")
parser.add_argument("--train-ref-panel", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")
parser.add_argument("--valid-ref-panel", type=str, default="data/benet_generations/4classes/chm22/train2_0gen/vcf_and_labels.h5")

# parser.add_argument("--train-mixed", type=str, default="data/benet_generations/chm22/train_8gens1k/vcf_and_labels.h5")
# parser.add_argument("--valid-mixed", type=str, default="data/benet_generations/chm22/val_8gens100/vcf_and_labels.h5")
# parser.add_argument("--ref-panel", type=str, default="data/benet_generations/chm22/train2_0gens/vcf_and_labels.h5")

parser.add_argument("--model", type=str, choices=["VanillaConvNet",
                                                  "LAINet"],
                    default="VanillaConvNet")

parser.add_argument("--num-epochs", type=int, default=99999999)
parser.add_argument("-b", "--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr-decay", type=int, default=-1)

parser.add_argument("--update-every", type=int, default=1)

parser.add_argument("--inpref-oper", type=str, choices=["XOR",
                                                        "AND"],
                    default="XOR")
parser.add_argument("--fst", dest="fst", action='store_true', default=True)


parser.add_argument("--smoother", type=str, choices=["1conv",
                                                     "2conv",
                                                     "3convdil",
                                                     "1TransfEnc",
                                                     "anc1conv",
                                                     "anc2conv"],
                    default="anc1conv")
parser.add_argument("--base-model", type=str, choices=["SFC", "SCS", "SCC",
                                                       "SCSMultisize"],
                    default="SCS")
parser.add_argument("--ref-pooling", type=str, choices=["maxpool", "topk"],
                    default="maxpool")
parser.add_argument("--topk-k", type=int, default=2)


parser.add_argument("--pos-emb", type=str, choices=["linpos",
                                                    "trained1",
                                                    "trained2",
                                                    "trained3",
                                                    "trained1dim4"],
                    default=None)
parser.add_argument("--transf-emb", dest="transf_emb", action='store_true')

parser.add_argument("--win-size", type=int, default=1200)
parser.add_argument("--win-stride", type=int, default=-1)
parser.add_argument("--dropout", type=float, default=-1)


parser.add_argument("--loss", type=str, default="BCE", choices=["BCE"])

parser.add_argument("--resume", dest="resume", action='store_true')

# parser.add_argument("--seq-len", type=int, default=317408)
parser.add_argument("--n-classes", type=int, default=4)

parser.add_argument("--multi-chm-train", dest="multi_chm_train", action='store_true')

parser.add_argument("--n-refs", type=int, default=60)

parser.add_argument("--comment", type=str, default=None)


if __name__ == '__main__':

    args = parser.parse_args()

    if args.resume:
        assert (bool(args.exp))
        with open("%s/args.pckl" % args.exp, "rb") as f:
            args = pickle.load(f)
            args.resume = True
    print(args)

    # model = AgnosticConvModel(args)
    model = AgnosticModel(args)
    # model = MultisizeAgnosticModel(args)

    transforms = build_transforms(args)
    if not args.multi_chm_train:
        train_dataset = ReferencePanelDataset(mixed_h5=args.train_mixed,
                                              reference_panel_h5=args.train_ref_panel,
                                              n_classes=args.n_classes,
                                              n_refs=args.n_refs,
                                              transforms=transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=reference_panel_collate)
    else:
        samples_per_chromosome = get_num_samples_per_chromosome(args.train_mixed)
        sampler = SameChmSampler(samples_per_chromosome, batch_size=args.batch_size)
        train_dataset = ReferencePanelMultiChmDataset(mixed_h5=args.train_mixed,
                                              reference_panel_h5=args.train_ref_panel,
                                              n_classes=args.n_classes,
                                              n_refs=args.n_refs,
                                              samples_per_chm=samples_per_chromosome,
                                              transforms=transforms)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                            collate_fn=reference_panel_collate)
    valid_dataset = ReferencePanelDataset(mixed_h5=args.valid_mixed,
                                          reference_panel_h5=args.valid_ref_panel,
                                          n_classes=args.n_classes,
                                          n_refs=args.n_refs,
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

