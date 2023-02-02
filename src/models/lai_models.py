
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
        return maximums.unsqueeze(0), indices

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

    def shared_refpanel(self, inp):
        k = min(self.k, inp.shape[1])
        maximums, indices = torch.topk(inp, k=k, dim=1)

        return maximums, indices

    def samplewise_refpanel(self, inp):
        k = self.k
        if inp.shape[0] < k:
            k = inp.shape[0]
        maximums, indices = torch.topk(inp, k=k, dim=0)
        assert indices.max() < inp.shape[0]
        return maximums, indices[0]

    def forward(self, inp, shared_refpanel):

        if shared_refpanel:
            return self.shared_refpanel(inp)
        else:
            return self.samplewise_refpanel(inp)

class AvgPool(nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
    def forward(self, inp):

        inp = inp.mean(dim=0, keepdim=True)
        return inp


def stack_ancestries(inp):
    out = []

    # inp is a batch (16)
    for i, x in enumerate(inp):
        out_sample = [None] * len(x.keys())
        for ancestry in x.keys():
            out_sample[ancestry] = x[ancestry]
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

    def shared_refpanel(self, inp):
        inp = inp * self.weights[:min(inp.shape[1], self.max_n)].unsqueeze(0)
        inp = torch.sum(inp, dim=1, keepdim=False)

        return inp

    def samplewise_refpanel(self, inp):
        # inp = inp + self.bias[:min(inp.shape[0], self.max_n)]
        out = inp * self.weights[:min(inp.shape[0], self.max_n)]
        out = torch.sum(out, dim=0, keepdim=True)
        return out

    def forward(self, inp, shared_refpanel):
        if shared_refpanel:
            return self.shared_refpanel(inp)
        else:
            return self.samplewise_refpanel(inp)

class BaseModel(nn.Module):
    def __init__(self, args, shared_refpanel=True):
        super(BaseModel, self).__init__()
        self.args = args
        self.window_size = args.win_size
        self.shared_refpanel = shared_refpanel

        self.batch_size_refpanel = args.batch_size_refpanel

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

        out = fast_windowed_distances(input_mixed, dict(ref_panel), window_size=self.window_size, ref_panel_bs=self.batch_size_refpanel)
        indices = {}
        for ancestry in out.keys():
            out[ancestry], indices[ancestry] = self.ref_pooling(out[ancestry], shared_refpanel=True)
            out[ancestry] = self.add_poolings(out[ancestry], shared_refpanel=True)
        return out, indices

        '''
        out_ = []
        max_indices_batch = []
        for x in out:
            x_ = {}
            max_indices_element = []
            for c in x.keys():
                x_[c] = self.sliding_window_sum(x[c])
                x_[c], max_indices = self.ref_pooling(x_[c])
                if self.args.ref_pooling == 'topk':
                    x_[c] = self.add_poolings(x_[c])
                max_indices_element.append(max_indices)

            out_.append(x_)
            max_indices_element = torch.stack(max_indices_element, dim=0)
            max_indices_batch.append(max_indices_element)

        max_indices_batch = torch.stack(max_indices_batch, dim=0)
        
        return out_, max_indices_batch
        '''


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

        out, max_indices = self.base_model(input_mixed, ref_panel)

        out = [out[anc] for anc in sorted(out.keys())]
        out_basemodel = out = torch.stack(out, axis=1)

        out = self.dropout(out)

        out_smoother = out = self.smoother(out)

        out = interpolate_and_pad(out, self.args.win_stride, seq_len)

        out = out.permute(0, 2, 1)

        output = {
            'predictions':out,
            'out_basemodel': out_basemodel,
            'out_smoother': out_smoother,
            'max_indices': max_indices
        }

        return output

def fast_windowed_distances(mixed, ref_panel, window_size, ref_panel_bs):
    bs, n_snps = mixed.shape
    n_windows = n_snps // window_size
    pad = n_snps % window_size
    if pad > 0:
        n_windows += 1
        pad = window_size - pad
        #pad = (pad // 2, pad - pad // 2)
        pad = (0, int(pad))
    for ancestry in ref_panel.keys():
        out = []

        # print(ref_panel[ancestry].shape)
        # print(mixed.shape)
        # ref_panel[ancestry] = f.pad(ref_panel[ancestry], pad)
        # mixed = f.pad(mixed, pad)
        # print(ref_panel[ancestry].shape)
        # print(mixed.shape)
        # quit()

        ref_panel[ancestry] = ref_panel[ancestry].unsqueeze(0) * mixed.unsqueeze(1)
        if pad != 0:
            ref_panel[ancestry] = f.pad(ref_panel[ancestry], pad)
        ref_panel[ancestry] = ref_panel[ancestry].reshape(bs, -1, n_windows, window_size)
        ref_panel[ancestry] = ref_panel[ancestry].mean(dim=3)



    return ref_panel


def fast_window_sum(inp, window_size):

    bs, n_ref, n_snps = inp.shap
    n_windows = n_snps // window_size

    pad = n_snps % window_size
    if pad != 0:
        n_windows += 1
        pad = window_size - pad
        pad = (pad//2, pad - pad//2)
        inp = f.pad(inp, pad)
    inp = inp.reshape(bs, n_ref, n_windows, window_size)
    inp = inp.sum(dim=3)
    return inp

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
