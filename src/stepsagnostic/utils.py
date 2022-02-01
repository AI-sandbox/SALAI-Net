import torch
import pickle
import numpy as np
from collections import Counter
import pandas as pd

import torch.nn as nn

from torchvision import transforms

def ancestry_accuracy(prediction, target):
    b, l, c = prediction.shape
    prediction = prediction.reshape(b*l, c)
    target = target.reshape(b*l)

    prediction = prediction.max(dim=1)[1]
    accuracy = (prediction == target).sum()

    return accuracy / l


class AverageMeter():
    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def get_average(self):
        return self.total / self.count

class ProgressSaver():

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.progress = {
            "epoch":[],
            "train_loss":[],
            "val_loss":[],
            "val_acc": [],
            "time":[],
            "best_epoch":[],
            "best_val_loss":[],
            "lr": []
            }

    def update_epoch_progess(self, epoch_data):

        for key in epoch_data.keys():
            self.progress[key].append(epoch_data[key])

        with open("%s/progress.pckl" % self.exp_dir, "wb") as f:
            pickle.dump(self.progress, f)

    def load_progress(self):
        with open("%s/progress.pckl" % self.exp_dir, "rb") as f:
            self.progress = pickle.load(f)

    def get_resume_stats(self):
        return self.progress["epoch"][-1], self.progress["best_val_loss"][-1], self.progress["time"][-1]



class ReshapedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ReshapedCrossEntropyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
    def forward(self, prediction, target):

        bs, seq_len, n_classes = prediction.shape

        # print(prediction.shape)
        # print(target.shape)
        # quit()
        prediction = prediction.reshape(bs * seq_len, n_classes)
        target = target.reshape(bs * seq_len)
        loss = self.CELoss(prediction, target)

        return loss

def adjust_learning_rate(base_lr, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay epochs"""
    lr = base_lr * (0.1 ** (epoch / lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr




class EncodeBinary:

    def __call__(self, inp):

        # 0 -> -1
        # 1 -> 1
        inp["mixed_vcf"] = inp["mixed_vcf"] * 2 - 1
        for anc in inp["ref_panel"]:
            inp["ref_panel"][anc] = inp["ref_panel"][anc] * 2 - 1

        return inp


def build_transforms(args):

    transforms_list = []

    transforms_list.append(EncodeBinary())

    transforms_list = transforms.Compose(transforms_list)

    return transforms_list


def to_device(item, device):

    item["mixed_vcf"] = item["mixed_vcf"].to(device)

    if "mixed_labels" in item.keys():
        item["mixed_labels"] = item["mixed_labels"].to(device)

    for i, panel in enumerate(item["ref_panel"]):
        for anc in panel.keys():
            item["ref_panel"][i][anc] = item["ref_panel"][i][anc].to(device)

    return item


def correct_max_indices(max_indices_batch, ref_panel_idx_batch):

    '''
    for each element of a batch, the dataloader samples randomly a set of founders in random order. For this reason,
    the argmax values output by the base model will represent different associations of founders, depending on how they have been
    sampled and ordered. By storing the sampling information during the data loading, we can then correct the argmax outputs
    into a shared meaning between batches and elements within the batch.
    '''

    for n in range(len(max_indices_batch)):
        max_indices = max_indices_batch[n]
        ref_panel_idx = ref_panel_idx_batch[n]
        max_indices_ordered = [None] * len(ref_panel_idx.keys())
        for i, c in enumerate(ref_panel_idx.keys()):
            max_indices_ordered[i] = max_indices[c]
        max_indices_ordered = torch.stack(max_indices_ordered)

        for i in range(max_indices.shape[0]):
            max_indices_ordered[i] = torch.take(torch.tensor(ref_panel_idx[i]), max_indices_ordered[i].cpu())
        max_indices_batch[n] = max_indices_ordered

    return max_indices_batch

def compute_ibd(output):

    all_ibd = []
    for n in range(output['out_basemodel'].shape[0]):

        classes_basemodel = torch.argmax(output['out_basemodel'][n], dim=0)
        # classes_smoother = torch.argmax(output['out_smoother'][n], dim=0)
        ibd = torch.gather(output['max_indices'][n].t(), index=classes_basemodel.unsqueeze(1), dim=1)
        ibd = ibd.squeeze(1)

        all_ibd.append(ibd)

    all_ibd = torch.stack(all_ibd)

    return all_ibd


def get_meta_data(chm, model_pos, query_pos, n_wind, wind_size, gen_map_df=None):
    """
    from LAI-Net code
    Transforms the predictions on a window level to a .msp file format.
        - chm: chromosome number
        - model_pos: physical positions of the model input SNPs in basepair units
        - query_pos: physical positions of the query input SNPs in basepair units
        - n_wind: number of windows in model
        - wind_size: size of each window in the model
        - genetic_map_file: the input genetic map file
    """

    model_chm_len = len(model_pos)

    # chm
    chm_array = [chm] * n_wind

    # start and end pyshical positions
    if model_chm_len % wind_size == 0:
        spos_idx = np.arange(0, model_chm_len, wind_size)  # [:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:], np.array([model_chm_len])]) - 1
    else:
        spos_idx = np.arange(0, model_chm_len, wind_size)[:-1]
        epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:-1], np.array([model_chm_len])]) - 1

    spos = model_pos[spos_idx]
    epos = model_pos[epos_idx]

    sgpos = [1] * len(spos)
    egpos = [1] * len(epos)

    # number of query snps in interval
    wind_index = [min(n_wind - 1, np.where(q == sorted(np.concatenate([epos, [q]])))[0][0]) for q in query_pos]
    window_count = Counter(wind_index)
    n_snps = [window_count[w] for w in range(n_wind)]

    # print(len(chm_array), len(spos), len(epos), len(sgpos), len(egpos), len(n_snps))
    # Concat with prediction table
    meta_data = np.array([chm_array, spos, epos, sgpos, egpos, n_snps]).T
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.columns = ["chm", "spos", "epos", "sgpos", "egpos", "n snps"]

    return meta_data_df


def write_msp_tsv(output_folder, meta_data, pred_labels, populations, query_samples, write_population_code=False):

    msp_data = np.concatenate([np.array(meta_data), pred_labels.T], axis=1).astype(str)

    with open(output_folder + "/predictions.msp.tsv", 'w') as f:
        if write_population_code:
            # first line (comment)
            f.write("#Subpopulation order/codes: ")
            f.write("\t".join([str(pop) + "=" + str(i) for i, pop in enumerate(populations)]) + "\n")
        # second line (comment/header)
        f.write("#" + "\t".join(meta_data.columns) + "\t")
        f.write("\t".join([str(s) for s in np.concatenate([[s + ".0", s + ".1"] for s in query_samples])]) + "\n")
        # rest of the lines (data)
        for l in range(msp_data.shape[0]):
            f.write("\t".join(msp_data[l, :]))
            f.write("\n")

    return


def msp_to_lai(msp_file, positions, lai_file=None):
    msp_df = pd.read_csv(msp_file, sep="\t", comment="#", header=None)
    data_window = np.array(msp_df.iloc[:, 6:])
    n_reps = msp_df.iloc[:, 5].to_numpy()
    assert np.sum(n_reps) == len(positions)
    data_snp = np.concatenate([np.repeat([row], repeats=n_reps[i], axis=0) for i, row in enumerate(data_window)])

    with open(msp_file) as f:
        first_line = f.readline()
        second_line = f.readline()

    header = second_line[:-1].split("\t")
    samples = header[6:]
    df = pd.DataFrame(data_snp, columns=samples, index=positions)

    if lai_file is not None:
        with open(lai_file, "w") as f:
            f.write(first_line)
        df.to_csv(lai_file, sep="\t", mode='a', index_label="position")
