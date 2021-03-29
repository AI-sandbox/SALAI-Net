
import torch
import torch.nn as nn
import torch.nn.functional as f

import math

class DevModel(nn.Module):
    def __init__(self, n_classes):
        super(DevModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 20, 201, padding=100)
        self.conv2 = nn.Conv1d(20, 20, 201, padding=100)
        self.conv3 = nn.Conv1d(20, n_classes, 201, padding=100)

    def forward(self, inp):
        out = inp.unsqueeze(1)
        out = f.relu(self.conv1(out))
        out = f.relu(self.conv2(out))
        out = self.conv3(out)
        # B x C x L -> B x L x C
        out = out.permute(0, 2, 1)

        return out


class VanillaConvNet(nn.Module):
    def __init__(self, args):

        self.args = args

        super(VanillaConvNet, self).__init__()
        ninp = 1
        fchid = 30
        fcout = smooth_in = 30
        smooth_hidd = 30

        if args.pos_emb is None:
            self.pos_emb = None
        # Concatenate a fixed embedding with linear dependency with the position
        # index: emb(n) = n / N
        elif args.pos_emb == "linpos":
            ninp += 1
            self.pos_emb = LinearPositionalEmbedding()
        # Concatenate on the input a trainable vector
        elif args.pos_emb == "trained1":
            ninp += 1
            self.pos_emb = TrainedPositionalEmbedding(args.seq_len)

        elif args.pos_emb == "trained1dim4":
            ninp += 4
            self.pos_emb = TrainedPositionalEmbedding(args.seq_len, dim=4)

        # Concatenate a trainable vector after the sliding fully connected
        elif args.pos_emb == "trained2":
            self.pos_emb = TrainedPositionalEmbedding(args.seq_len // args.win_size)
            smooth_in += 1

        elif args.pos_emb == "trained3":
            self.pos_emb1 = TrainedPositionalEmbedding(args.seq_len)
            ninp += 1
            self.pos_emb2 = TrainedPositionalEmbedding(args.seq_len // args.win_size)
            smooth_in += 1
        else:
            raise ValueError()

        if self.args.transf_emb == True:
            self.transf_emb = PositionalEncoding(smooth_in, max_len=args.seq_len // args.win_size, dropout=0.1)

        if args.win_stride == -1:
            self.win_stride = args.win_size
        else:
            self.win_stride = args.win_stride

        self.base_model = SlidingFullyConnected(args.win_size, self.win_stride, ninp=ninp, nhid=fchid, nout=fcout)

        self.hid2classes = nn.Conv1d(fcout, args.n_classes, kernel_size=1)

        if args.smoother == "1conv":
            kernel_size = 75
            dilation = args.conv1_dilation
            dilated_kernelsize = kernel_size + (kernel_size-1) * (dilation-1)
            padding = dilated_kernelsize // 2
            self.smoother = nn.Conv1d(smooth_in, args.n_classes, kernel_size=kernel_size, padding=padding, dilation=dilation)
        elif args.smoother == "2conv":
            self.smoother = TwoConvSmoother(ninp=smooth_in, nhid=smooth_hidd, nout=args.n_classes)
        elif args.smoother == "2convDil":
            self.smoother = TwoConvDilSmoother(ninp=smooth_in, nhid=smooth_hidd, nout=args.n_classes)
        elif args.smoother == "1TransfEnc":
            self.smoother = TransformerEncoderConv(fcout, 3, smooth_hidd, args.n_classes, dropout=0.1)
        else:
            raise ValueError()
        # self.last_conv = nn.Conv1d(convhidd, args.n_classes, kernel_size=75, padding=37)
        # self.bn1 = nn.BatchNorm1d(30)


    def forward(self, inp, need_weights=False):

        h1 = None
        out = inp.unsqueeze(1)

        # Pos Embeddings before baseNet
        if self.args.pos_emb == "linpos":
            out = self.pos_emb.apply_embedding(out)
        elif self.args.pos_emb in ["trained1", "trained1dim4", "trained3"]:
            out = self.pos_emb(out)
        # print(out.shape)
        out = self.base_model(out)
        # print(out.shape)

        if self.args.pos_emb == "trained2":
            out = self.pos_emb(out)
        if self.args.pos_emb == "trained3":
            out = self.pos_emb2(out)
        if self.args.transf_emb:
            # B x C x L --> L x B x C
            out = out.permute(2, 0, 1)
            out = self.transf_emb(out)
            out = out.permute(1, 2, 0)

        attention = None
        if self.args.smoother == "1TransfEnc":
            hid, out, attention = self.smoother(out, need_weights=need_weights)
        else:
            hid, out = self.smoother(out)

        # print(out.shape)
        # h1 = self.hid2classes(h1)

        # h1 = interpolate_and_pad(h1, self.win_stride, self.args.seq_len)
        out = interpolate_and_pad(out, self.win_stride, self.args.seq_len)
        # print(out.shape)


        # h1 = h1.permute(0, 2, 1)
        out = out.permute(0, 2, 1)

        return h1, out, attention


class SlidingFullyConnected(nn.Module):
    def __init__(self, win_size, stride, ninp, nhid, nout):
        super(SlidingFullyConnected, self).__init__()
        self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=win_size, stride=stride, padding=0)

        self.conv2 = nn.Conv1d(nhid, nout, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(nhid)

    def forward(self, inp):
        # print("1", inp.shape)
        out = self.conv1(inp)
        # print("2", out.shape)

        out = self.bn(out)
        out = f.relu(out)
        out = f.relu(self.conv2(out))
        # print("3", out.shape)

        return out

    class SlidingFullyConnected(nn.Module):
        def __init__(self, win_size, stride, ninp, nhid, nout):
            super(SlidingFullyConnected, self).__init__()
            self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=win_size,
                                   stride=stride, padding=0)

            self.conv2 = nn.Conv1d(nhid, nout, kernel_size=1, stride=1)
            self.bn = nn.BatchNorm1d(nhid)

        def forward(self, inp):
            # print("1", inp.shape)
            out = self.conv1(inp)
            # print("2", out.shape)

            out = self.bn(out)
            out = f.relu(out)
            out = f.relu(self.conv2(out))
            # print("3", out.shape)

            return out


class SlidingChannelConnected(nn.Module):
    def __init__(self, win_size, stride):
        super(SlidingChannelConnected, self).__init__()

        self.kernel = torch.randn((1, 1, 1, win_size))
        # We pass is as parameter but freeze it
        self.kernel = nn.Parameter(self.kernel)
        self.stride = stride

        # self.batchnorm = nn.BatchNorm1d(num_features=32)


    def forward(self, inp):

        inp = inp.unsqueeze(1)
        inp = f.conv2d(inp, self.kernel, stride=(1, self.stride))
        inp = inp.squeeze(1)
        # inp = self.batchnorm(inp)

        return inp


class SlidingChannelSum(nn.Module):
    def __init__(self, win_size, stride):
        super(SlidingChannelSum, self).__init__()

        self.kernel = torch.ones(1, 1, 1, win_size).float() / win_size
        # We pass is as parameter but freeze it
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        self.stride = stride


    def forward(self, inp):

        inp = inp.unsqueeze(1)
        inp = f.conv2d(inp, self.kernel, stride=(1, self.stride))
        inp = inp.squeeze(1)

        return inp


class TransformerEncoderConv(nn.Module):
    def __init__(self, ninp, nhead, nhid, nout, dropout):
        super(TransformerEncoderConv, self).__init__()
        self.pos_emb = PositionalEncoding(ninp, dropout=dropout)

        # Started using
        self.transf_enc = TransformerEncoderLayer(ninp, nhead, nhid, dropout=dropout)
        # self.transf_enc = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout=dropout)

        self.out_conv = nn.Conv1d(nhid, nout, kernel_size=1, stride=1)

    def forward(self, inp, need_weights=False):

        # B x C x L --> L x B x C
        out = inp.permute(2, 0, 1)

        out = self.pos_emb(out)
        out, attention_mat = self.transf_enc(out, need_weights=need_weights)

        # L x B x C --> B x C x L
        out = out.permute(1, 2, 0)

        out = self.out_conv(out)
        return (None, out, attention_mat)


class TwoConvSmoother(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super(TwoConvSmoother, self).__init__()

        self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=75, padding=37)
        self.conv2 = nn.Conv1d(nhid, nout, kernel_size=75, padding=37)

    def forward(self, inp):
        out = hidd = self.conv1(inp)

        # out = f.relu(out)
        out = self.conv2(out)

        return hidd, out


class TwoConvDilSmoother(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super(TwoConvDilSmoother, self).__init__()

        self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=75, padding=37)
        self.conv2 = nn.Conv1d(nhid, nout, kernel_size=75, padding=74, dilation=2)

    def forward(self, inp):
        hidd = self.conv1(inp)
        out = hidd
        # out = f.relu(out)
        out = self.conv2(out)

        return hidd, out


class ConvMaxPoolSmoother(nn.Module):
    def __init__(self, ninp, nout):
        super(ConvMaxPoolSmoother, self).__init__()
        self.maxpool = nn.MaxPool2d((1, 75), padding=(0, 37))
        self.conv1 = nn.Conv1d(ninp, nout, kernel_size=75, padding=37)

    def forward(self, inp):
        print(inp.shape)
        out = inp.unsqueeze(1)
        print(out.shape)
        out = self.maxpool(out)
        print(out.shape)
        out = out.squeeze(1)
        print(out.shape)
        out = self.conv2(f.relu(out))
        print(out.shape)
        quit()

        return None, out








class AgnosticConvModel(nn.Module):

    def __init__(self, args):
        super(AgnosticConvModel, self).__init__()
        self.args = args

        ninp = args.n_refs
        fchid = 30
        fcout = smooth_in = 30

        if args.win_stride == -1:
            self.win_stride = args.win_size
        else:
            self.win_stride = args.win_stride

        # Which input representation wrt the templates:
        if args.inpref_oper == "XOR":
            self.inpref_oper = XOR()
        elif args.inpref_oper == "AND":
            self.inpref_oper = AND()
        else:
            raise ValueError()

        # Which base model
        if args.base_model == "SFC":
            self.base_model = SlidingFullyConnected(win_size=args.win_size, stride=self.win_stride, ninp=ninp,
                                             nhid=fchid, nout=fcout)
        elif args.base_model == "SCS":
            self.base_model = SlidingChannelSum(win_size=args.win_size, stride=self.win_stride)
            smooth_in = args.n_refs

        elif args.base_model == "SCC":
            self.base_model = nn.Sequential(
                SlidingChannelConnected(win_size=args.win_size, stride=self.win_stride),
                nn.BatchNorm1d(args.n_refs))
            smooth_in = args.n_refs

        dilation = 1
        dilated_kernel_size = 75 + 74 * (dilation - 1)
        #
        padding = dilated_kernel_size // 2
        padding = 37

        if args.smoother == "1conv":
            # kernel_size = 150
            smooth_in = 4
            kernel_size = 75
            self.smoother = nn.Conv1d(smooth_in, args.n_classes, kernel_size=kernel_size,
                                  padding=padding, dilation=dilation)

        elif args.smoother == "2conv":
            self.smoother = nn.Sequential(
                nn.Conv1d(smooth_in, 30, kernel_size=75, padding=padding, dilation=dilation),
                # nn.ReLU(),
                # nn.BatchNorm1d(30),
                nn.Softmax(dim=1),
                nn.Conv1d(30, args.n_classes, kernel_size=75, padding=padding,dilation=dilation),
            )
        elif args.smoother == "3convdil":
            self.smoother = nn.Sequential(
                nn.Conv1d(smooth_in, 30, kernel_size=75, padding=padding,dilation=dilation),
                # nn.ReLU(),
                nn.BatchNorm1d(30),
                nn.Conv1d(30, 30, kernel_size=75, padding=padding,dilation=dilation),
                nn.BatchNorm1d(30),
                nn.Conv1d(30, args.n_classes, kernel_size=75, padding=74,dilation=2),
            )
        else:
            raise ValueError()

        if args.dropout > 0:
            print("Dropout=", args.dropout)
            self.dropout = nn.Dropout(p=args.dropout)
        else:
            print("No dropout")
            self.dropout = nn.Sequential()

        refs_per_class = args.n_refs // args.n_classes
        self.maxpool = nn.MaxPool2d(kernel_size=(refs_per_class, 1),
                                    stride=(refs_per_class, 1))



    def forward(self, input_mixed, ref_panel):

        seq_len = input_mixed.shape[-1]

        out = self.inpref_oper(input_mixed, ref_panel)

        out = self.base_model(out)

        out = out.unsqueeze(1)
        out = self.maxpool(out)
        out = out.squeeze(1)

        out = self.dropout(out)

        out = self.smoother(out)

        out = interpolate_and_pad(out, self.win_stride, seq_len)


        out = out.permute(0, 2, 1)
        # print("9", out.shape)
        # quit()

        return out


def multiply_ref_panel(mixed, ref_panel):
    all_refs = [ref_panel[ancestry] for ancestry in ref_panel.keys()]
    all_refs = torch.cat(all_refs, dim=0)

    # print(all_refs.shape, torch.unique(all_refs, return_counts=True))
    # print(mixed.shape, torch.unique(mixed, return_counts=True))
    # quit()

    return all_refs * mixed.unsqueeze(0)

# Enconde whether they are equal or diferent
class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
    def forward(self, input_mixed, ref_panel):
        with torch.no_grad():
            out = []
            for inp, ref in zip(input_mixed, ref_panel):
                out.append(multiply_ref_panel(inp, ref))
            out = torch.stack(out)

        return out

class AND(nn.Module):
    def __init__(self):
        super(AND, self).__init__()

    def forward(self, input_mixed, ref_panel):
        with torch.no_grad():
            out = []

            input_mixed = (input_mixed + 1) / 2

            for inp, ref in zip(input_mixed, ref_panel):
                for chm in ref.keys():
                    ref[chm] = (ref[chm] + 1) / 2

                out.append(multiply_ref_panel(inp, ref))
            out = torch.stack(out)

        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=516800//400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LinearPositionalEmbedding():
    def __init__(self, operation="concat"):
        self.emb = None
        self.operation = operation

    def concatenate_embedding(self, inp):

        bs, n_channels, seq_len = inp.shape
        if self.emb is None:
            self.emb = torch.range(0, seq_len - 1) / seq_len
            self.emb = self.emb.repeat(bs, 1)
            self.emb = self.emb.unsqueeze(1)
            self.emb = self.emb.to(inp.device)

        inp = torch.cat((inp, self.emb[:bs]), dim=1)
        return inp

    def apply_embedding(self, inp):
        if self.operation == "concat":
            return self.concatenate_embedding(inp)


class TrainedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, operation="concat", dim=1):
        super(TrainedPositionalEmbedding, self).__init__()

        self.emb = nn.Parameter(torch.randn(1, dim, seq_len), requires_grad=True)
        self.operation = operation

    def concatenate_embedding(self, inp):

        bs, n_channels, seq_len = inp.shape

        # print(inp.shape)
        # print(self.emb.shape)
        # print(self.emb.repeat(bs, 1, 1).shape)
        inp = torch.cat((inp, self.emb.repeat(bs, 1, 1)), dim=1)
        return inp

    def add_embedding(self, inp):
        return inp + self.emb

    def apply_embedding(self, inp):
        if self.operation == "concat":
            return self.concatenate_embedding(inp)
        if self.operation == "add":
            return self.add_embedding(inp)

    def forward(self, inp):
        return self.apply_embedding(inp)


def interpolate_and_pad(inp, upsample_factor, target_len):

    bs, n_chann, original_len = inp.shape
    non_padded_upsampled_len = original_len * upsample_factor
    inp = f.interpolate(inp, size=non_padded_upsampled_len)

    left_pad = (target_len - non_padded_upsampled_len) // 2
    right_pad = target_len - non_padded_upsampled_len - left_pad
    inp = f.pad(inp, (left_pad, right_pad), mode="replicate")

    return inp

# Below this, there is only overwritten pytorch code
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch import Tensor
from typing import Optional
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, need_weights=False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attention = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=need_weights)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
