
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


        self.sfc_net = SlidingFullyConnected(args.win_size, ninp=ninp, nhid=fchid, nout=fcout)
        self.hid2classes = nn.Conv1d(fcout, args.n_classes, kernel_size=1)

        if args.smoother == "1conv":
            self.smoother = nn.Conv1d(smooth_in, args.n_classes, kernel_size=75, padding=37)
        elif args.smoother == "2conv":
            self.smoother = nn.Sequential(
                nn.Conv1d(smooth_in, smooth_hidd, kernel_size=75, padding=37),
                nn.ReLU(),
                nn.Conv1d(smooth_hidd, args.n_classes, kernel_size=75, padding=37),
            )
        elif args.smoother == "1TransfEnc":
            self.smoother = TransformerEncoderConv(fcout, 3, smooth_hidd, args.n_classes, dropout=0.1)
        else:
            raise ValueError()
        # self.last_conv = nn.Conv1d(convhidd, args.n_classes, kernel_size=75, padding=37)
        # self.bn1 = nn.BatchNorm1d(30)


    def forward(self, inp):
        out = inp.unsqueeze(1)

        # Pos Embeddings before baseNet
        if self.args.pos_emb == "linpos":
            out = self.pos_emb.apply_embedding(out)
        elif self.args.pos_emb in ["trained1", "trained1dim4", "trained3"]:
            out = self.pos_emb(out)

        out = h1 = self.sfc_net(out)

        if self.args.pos_emb == "trained2":
            out = self.pos_emb(out)
        if self.args.pos_emb == "trained3":
            out = self.pos_emb2(out)
        if self.args.transf_emb:
            # B x C x L --> L x B x C
            out = out.permute(2, 0, 1)
            out = self.transf_emb(out)
            out = out.permute(1, 2, 0)

        out = self.smoother(out)

        non_padded_length = self.args.seq_len // self.args.win_size * self.args.win_size


        h1 = self.hid2classes(h1)
        h1 = f.interpolate(h1, size=non_padded_length)
        h1 = f.pad(h1, (0, self.args.seq_len - non_padded_length),
                    mode="replicate")
        h1 = h1.permute(0, 2, 1)

        out = f.interpolate(out, size=non_padded_length)
        out = f.pad(out, (0, self.args.seq_len - non_padded_length),
                    mode="replicate")

        out = out.permute(0, 2, 1)
        return h1, out


class SlidingFullyConnected(nn.Module):
    def __init__(self, win_size, ninp, nhid, nout):
        super(SlidingFullyConnected, self).__init__()
        self.conv1 = nn.Conv1d(ninp, nhid, kernel_size=win_size, stride=win_size, padding=0)
        self.conv2 = nn.Conv1d(nhid, nout, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm1d(nhid)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.bn(out)
        out = f.relu(out)
        out = f.relu(self.conv2(out))
        return out


class TransformerEncoderConv(nn.Module):
    def __init__(self, ninp, nhead, nhid, nout, dropout):
        super(TransformerEncoderConv, self).__init__()
        self.pos_emb = PositionalEncoding(ninp, dropout=dropout)

        self.transf_enc = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout=dropout)
        self.out_conv = nn.Conv1d(nhid, nout, kernel_size=1, stride=1)

    def forward(self, inp):

        # B x C x L --> L x B x C
        out = inp.permute(2, 0, 1)

        out = self.pos_emb(out)
        out = self.transf_enc(out)

        # L x B x C --> B x C x L
        out = out.permute(1, 2, 0)

        out = self.out_conv(out)
        return out


class AgnosticConvModel(nn.Module):

    def __init__(self, args):
        super(AgnosticConvModel, self).__init__()
        self.args = args

        ninp = 28
        fchid = 30
        fcout = smooth_in = 30

        self.sfc_net = SlidingFullyConnected(args.win_size, ninp=ninp,
                                             nhid=fchid, nout=fcout)

        self.smoother = nn.Conv1d(smooth_in, args.n_classes, kernel_size=75,
                                  padding=37)


    def multiply_ref_panel(self, mixed, ref_panel):
        all_refs = [ref_panel[ancestry] for ancestry in ref_panel.keys()]
        all_refs = torch.cat(all_refs, dim=0)

        return all_refs * mixed.unsqueeze(0)

    def forward(self, input_mixed, ref_panel):
        out = []
        for inp, ref in zip(input_mixed, ref_panel):
            out.append(self.multiply_ref_panel(inp, ref))
        out = torch.stack(out)

        out = self.sfc_net(out)

        out = self.smoother(out)

        non_padded_length = self.args.seq_len // self.args.win_size * self.args.win_size
        out = f.interpolate(out, size=non_padded_length)
        out = f.pad(out, (0, self.args.seq_len - non_padded_length),
                    mode="replicate")

        out = out.permute(0, 2, 1)

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

