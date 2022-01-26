import torch
import pickle

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


