

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
    def __init__(self, n_classes, posemb=None):
        super(VanillaConvNet, self).__init__()

        if posemb is None:
            self.posemb = None
            ninp = 1
        elif posemb == "linpos":
            ninp = 2
            self.posemb = LinearPositionalEmbedding()
        else:
            raise ValueError()

        self.conv1 = nn.Conv1d(ninp, 30, kernel_size=400, stride=400, padding=0)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(30, n_classes, kernel_size=75, padding=37)

        self.bn1 = nn.BatchNorm1d(30)


    def forward(self, inp):
        out = inp.unsqueeze(1)

        if self.posemb is not None:
            out = self.posemb.concatenate_embedding(out)

        out = self.conv1(out)
        out = f.relu(self.bn1(out))
        out = f.relu(self.conv2(out))
        out = f.relu(self.conv3(out))
        out = f.interpolate(out, size=516800)
        out = out.permute(0, 2, 1)
        return out

class LinearPositionalEmbedding():
    def __init__(self):
        self.emb = None

    def concatenate_embedding(self, inp):

        bs, n_channels, seq_len = inp.shape
        if self.emb is None:
            self.emb = torch.range(0, seq_len-1) / seq_len
            self.emb = self.emb.repeat(bs, 1)
            self.emb = self.emb.unsqueeze(1)
            self.emb = self.emb.to(inp.device)

        inp = torch.cat((inp, self.emb[:bs]), dim=1)
        return inp

