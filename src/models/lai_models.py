
import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

import math

import time

class SlidingWindowSum(nn.Module):

    def __init__(self, win_size, stride):
        super(SlidingWindowSum, self).__init__()
        self.kernel = torch.ones(1, 1, win_size).float() / win_size
        # We pass is as parameter but freeze it
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        self.stride = stride

    def forward(self, inp):
        inp = inp.unsqueeze(1)
        inp = f.conv1d(inp, self.kernel, stride=(self.stride))
        inp = inp.squeeze(1)
        return inp

class AncestryLevelConvSmoother(nn.Module):
    def __init__(self, kernel_size, padding, init="rand"):
        super(AncestryLevelConvSmoother, self).__init__()
        self.conv = nn.Conv2d(1, 1, (1, kernel_size), padding=(0, padding))

        if init == "gauss_filter":
            var = 0.2
            x = np.arange(kernel_size) / kernel_size - 0.5
            kernel = np.exp(- (x/var)**2)
            kernel = torch.tensor(kernel).unsqueeze(0).unsqueeze(0).unsqueeze(0).float() * 7

            self.conv.weight = nn.Parameter(kernel)

        elif init == "rand":
            pass
        else:
            raise ValueError()

    def forward(self, inp):
        inp = inp.unsqueeze(1)
        inp = self.conv(inp)
        inp = inp.squeeze(1)
        return inp


class RefMaxPool(nn.Module):
    def __init__(self):
        super(RefMaxPool, self).__init__()

    def forward(self, inp):
        maximums, indices = torch.max(inp, dim=0)
        return maximums.unsqueeze(0)

class BaggingMaxPool(nn.Module):
    def __init__(self, k=20, split=0.25):
        super(BaggingMaxPool, self).__init__()
        self.k = k
        self.split = split

        self.maxpool = RefMaxPool()
        self.averagepool = AvgPool()

    def forward(self, inp):
        pooled_refs = []

        total_n = inp.shape[0]
        select_n = int(total_n * self.split)

        for _ in range(self.k):

            indices = torch.randint(low=0, high=int(total_n), size=(select_n,))
            selected = inp[indices, :]
            maxpooled = self.maxpool(selected)

            pooled_refs.append(maxpooled)

        pooled_refs = torch.cat(pooled_refs, dim=0)
        return self.averagepool(pooled_refs)

class TopKPool(nn.Module):
    def __init__(self, k):
        super(TopKPool, self).__init__()
        self.k = k
    def forward(self, inp):
        k = self.k
        if inp.shape[0] < k:
            k=inp.shape[0]
        maximums, indices = torch.topk(inp, k=k, dim=0)
        return maximums

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
    def forward(self, inp):

        inp = inp.mean(dim=0, keepdim=True)
        return inp


def stack_ancestries(inp):
    out = []
    for i, x in enumerate(inp):

        out_sample = []
        for ancestry in x.keys():
            out_sample.append(x[ancestry])
        out_sample = torch.cat(out_sample)
        out.append(out_sample)
    out = torch.stack(out)

    return out

class AddPoolings(nn.Module):
    def __init__(self, max_n=2):
        self.max_n = max_n
        super(AddPoolings, self).__init__()
        #self.weights=nn.Parameter(torch.ones(max_n).unsqueeze(1))
        self.weights=nn.Parameter(torch.rand(max_n).unsqueeze(1), requires_grad=True)
        # self.bias = nn.Parameter(torch.rand(max_n).unsqueeze(1), requires_grad=True)
    def forward(self, inp):

        # inp = inp + self.bias[:min(inp.shape[0], self.max_n)]
        out = inp * self.weights[:min(inp.shape[0], self.max_n)]
        out = torch.sum(out, dim=0, keepdim=True)

        return out

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

        self.inpref_oper = XOR()
        # old base_model

        self.sliding_window_sum = SlidingWindowSum(win_size=args.win_size, stride=args.win_stride)

        if args.ref_pooling == "maxpool":
            self.ref_pooling = RefMaxPool()
        elif args.ref_pooling == "topk":
            self.ref_pooling = TopKPool(args.topk_k)
            self.add_poolings = AddPoolings(max_n=args.topk_k)
        elif args.ref_pooling == "average":
            self.ref_pooling = AvgPool()

        else:
            raise ValueError('Wrong type of ref pooling')


    def forward(self, input_mixed, ref_panel):

        with torch.no_grad():
            out = self.inpref_oper(input_mixed, ref_panel)
        out_ = []
        for x in out:
            x_ = {}
            for c in x.keys():
                x_[c] = self.sliding_window_sum(x[c])
                x_[c] = self.ref_pooling(x_[c])
                if self.args.ref_pooling == 'topk':
                    x_[c] = self.add_poolings(x_[c])
            out_.append(x_)

        out = out_
        del out_
        return out


class AgnosticModel(nn.Module):

    def __init__(self, args):
        super(AgnosticModel, self).__init__()
        if args.win_stride == -1:
            args.win_stride = args.win_size
        self.args = args

        self.base_model = BaseModel(args=args)

        if args.smoother == "anc1conv":
            self.smoother = AncestryLevelConvSmoother(kernel_size=75, padding=37)
        elif args.smoother == "none":
            self.smoother = nn.Sequential()
        else:
            raise ValueError()

        if args.dropout > 0:
            print("Dropout=", args.dropout)
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            print("No dropout")
            self.dropout = nn.Sequential()

    def forward(self, input_mixed, ref_panel):

        seq_len = input_mixed.shape[-1]


        out = self.base_model(input_mixed, ref_panel)


        out = stack_ancestries(out).to(next(self.parameters()).device)
        out = self.dropout(out)

        out = self.smoother(out)

        out = interpolate_and_pad(out, self.args.win_stride, seq_len)

        out = out.permute(0, 2, 1)

        return out


def multiply_ref_panel_stack_ancestries(mixed, ref_panel):
    all_refs = [ref_panel[ancestry] for ancestry in ref_panel.keys()]
    all_refs = torch.cat(all_refs, dim=0)

    return all_refs * mixed.unsqueeze(0)

def multiply_ref_panel(mixed, ref_panel):
    out = {
        ancestry: mixed.unsqueeze(0) * ref_panel[ancestry] for ancestry in ref_panel.keys()
    }
    return out


# SNP-wise similrity
class XOR(nn.Module):

    def __init__(self):
        super(XOR, self).__init__()

    def forward(self, input_mixed, ref_panel):
        with torch.no_grad():
            out = []
            for inp, ref in zip(input_mixed, ref_panel):
                out.append(multiply_ref_panel(inp, ref))
        return out

def interpolate_and_pad(inp, upsample_factor, target_len):

    bs, n_chann, original_len = inp.shape
    non_padded_upsampled_len = original_len * upsample_factor
    inp = f.interpolate(inp, size=non_padded_upsampled_len)

    left_pad = (target_len - non_padded_upsampled_len) // 2
    right_pad = target_len - non_padded_upsampled_len - left_pad
    inp = f.pad(inp, (left_pad, right_pad), mode="replicate")

    return inp
