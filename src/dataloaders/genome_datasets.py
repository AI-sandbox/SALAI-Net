import numpy as np

import torch
from torch.utils.data import Dataset


def to_tensor(item):
    for k in item.keys():
        item[k] = torch.tensor(item[k])

    item["vcf"] = item["vcf"].float()
    item["labels"] = item["labels"].long()
    return item

class GenomeDataset(Dataset):
    def __init__(self, data, transforms):
        data = np.load(data)
        self.vcf_data = data["vcf"].astype(np.float)
        self.labels = data["labels"]
        self.transforms = transforms

    def __len__(self):
        return self.vcf_data.shape[0]

    def __getitem__(self, item):
        item = {
            "vcf": self.vcf_data[item],
            "labels": self.labels[item]
        }

        item = to_tensor(item)

        item = self.transforms(item)
        return item

