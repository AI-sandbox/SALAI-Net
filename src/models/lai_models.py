

import torch
import torch.nn as nn
import torch.nn.functional as f

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

        print(out.shape)
        quit()

        return out

class VanillaConvNet(nn.Module):
    def __init__(self, args):

        self.args = args

        super(VanillaConvNet, self).__init__()
        ninp = 1
        fchid = 30
        fcout = convin = 30
        if args.pos_emb is None:
            self.pos_emb = None
        # Concatenate a fixed embedding with linear dependency with the positon index
        # emb(n) = n / N
        elif args.pos_emb == "linpos":
            ninp += 1
            self.pos_emb = LinearPositionalEmbedding()
        # Concatenate on the input a trainable vector
        elif args.pos_emb == "trained1":
            ninp += 1
            self.pos_emb = TrainedPositionalEmbedding(args.seq_len)
        # Concatenate a trainable vector after the sliding fully connected
        elif args.pos_emb == "trained2":
            self.pos_emb = TrainedPositionalEmbedding(args.seq_len // 400)
            convin += 1

        else:
            raise ValueError()

        self.conv1 = nn.Conv1d(ninp, fchid, kernel_size=400, stride=400, padding=0)
        self.conv2 = nn.Conv1d(fchid, fcout, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(convin, args.n_classes, kernel_size=75, padding=37)
        self.bn1 = nn.BatchNorm1d(30)


    def forward(self, inp):
        out = inp.unsqueeze(1)

        if self.args.pos_emb == "linpos":
            out = self.pos_emb.apply_embedding(out)
        elif self.args.pos_emb == "trained1":
            out = self.pos_emb(out)

        out = self.conv1(out)
        out = f.relu(self.bn1(out))

        out = f.relu(self.conv2(out))
        if self.args.pos_emb == "trained2":
            out = self.pos_emb(out)

        # removed the relu from the last layer
        out = (self.conv3(out))
        out = f.interpolate(out, size=self.args.seq_len)
        out = out.permute(0, 2, 1)
        return out

class LinearPositionalEmbedding():
    def __init__(self, operation="concat"):
        self.emb = None
        self.operation = operation

    def concatenate_embedding(self, inp):

        bs, n_channels, seq_len = inp.shape
        if self.emb is None:
            self.emb = torch.range(0, seq_len-1) / seq_len
            self.emb = self.emb.repeat(bs, 1)
            self.emb = self.emb.unsqueeze(1)
            self.emb = self.emb.to(inp.device)

        inp = torch.cat((inp, self.emb[:bs]), dim=1)
        return inp

    def apply_embedding(self, inp):
        if self.operation =="concat":
            return self.concatenate_embedding(inp)


class TrainedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, operation="concat"):
        super(TrainedPositionalEmbedding, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = nn.Parameter(torch.randn(1, 1, seq_len), requires_grad=True).to(device)
        self.operation = operation

    def concatenate_embedding(self, inp):

        bs, n_channels, seq_len = inp.shape
        inp = torch.cat((inp, self.emb.repeat(bs, 1, 1)), dim=1)
        return inp

    def apply_embedding(self, inp):
        if self.operation == "concat":
            return self.concatenate_embedding(inp)

    def forward(self, inp):
        return self.apply_embedding(inp)


