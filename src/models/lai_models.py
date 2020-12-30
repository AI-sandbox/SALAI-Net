

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

class VanillaLAINet(nn.Module):
    def __init__(self, n_classes):
        super(VanillaLAINet, self).__init__()
        self.conv1 = nn.Conv1d(1, 30, kernel_size=400, stride=400, padding=0)
        self.conv2 = nn.Conv1d(30, 30, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(30, n_classes, kernel_size=75, padding=37)

        self.bn1 = nn.BatchNorm1d(30)

    def forward(self, inp):
        out = inp.unsqueeze(1)
        out = self.conv1(out)
        out = f.relu(self.bn1(out))
        out = f.relu(self.conv2(out))
        out = f.relu(self.conv3(out))
        out = f.interpolate(out, size=516800)
        out = out.permute(0, 2, 1)
        return out

