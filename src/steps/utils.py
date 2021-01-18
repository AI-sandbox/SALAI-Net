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
        inp["vcf"] = inp["vcf"] * 2 - 1

        return inp


def build_transforms(args):

    transforms_list = []

    transforms_list.append(EncodeBinary())

    transforms_list = transforms.Compose(transforms_list)

    return transforms_list