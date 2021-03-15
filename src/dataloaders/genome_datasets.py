import numpy as np

import torch
from torch.utils.data import Dataset
import random
import h5py
from torch.utils.data.dataloader import default_collate


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


class ReferencePanel:

    def __init__(self, reference_panel_h5, n_refs, n_classes):

        self.reference_vcf = reference_panel_h5["vcf"]
        reference_labels = reference_panel_h5["labels"]

        reference_panel = {}

        for i, ref in enumerate(reference_labels):
            ancestry = np.unique(ref)
            # For now, it is only supported single ancestry references
            assert len(ancestry) == 1
            ancestry = int(ancestry)

            if ancestry in reference_panel.keys():
                reference_panel[ancestry].append(i)
            else:
                reference_panel[ancestry] = [i]

        for ancestry in reference_panel.keys():
            random.shuffle(reference_panel[ancestry])
            print(ancestry, len(reference_panel[ancestry]))

        self.reference_panel_index_dict = reference_panel

        self.n_classes = n_classes
        self.n_refs = n_refs

    def sample_uniform_all_classes(self, n_sample_per_class):

        reference_samples = {}

        for ancestry in self.reference_panel_index_dict.keys():
            indexes = random.sample(self.reference_panel_index_dict[ancestry],
                                    n_sample_per_class)
            reference_samples[ancestry] = []
            for i in indexes:
                reference_samples[ancestry].append(self.reference_vcf[i])

            reference_samples = {x: np.array(reference_samples[x]) for x in
                                 reference_samples.keys()}

        return (reference_samples)

    def sample_reference_panel(self):
        return self.sample_uniform_all_classes(n_sample_per_class=self.n_refs // self.n_classes)


def ref_pan_to_tensor(item):
    item["mixed_vcf"] = torch.tensor(item["mixed_vcf"]).float()
    item["mixed_labels"] = torch.tensor(item["mixed_labels"]).long()

    for c in item["ref_panel"]:
        item["ref_panel"][c] = torch.tensor(item["ref_panel"][c])

    return item


class ReferencePanelDataset(Dataset):

    def __init__(self, mixed_h5, reference_panel_h5, n_refs, n_classes, transforms):
        reference_panel_file = h5py.File(reference_panel_h5, "r")
        self.reference_panel = ReferencePanel(reference_panel_file, n_refs, n_classes)

        mixed_file = h5py.File(mixed_h5)
        self.mixed_vcf = mixed_file["vcf"]
        self.mixed_labels = mixed_file["labels"]
        self.transforms = transforms

        self.n_refs = n_refs
        self.n_classes = n_classes

    def __len__(self):
        return self.mixed_vcf.shape[0]

    def __getitem__(self, item):

        item = {
            "mixed_vcf": self.mixed_vcf[item].astype(float),
            "mixed_labels": self.mixed_labels[item]
        }

        item["ref_panel"] = self.reference_panel.sample_reference_panel()

        item = ref_pan_to_tensor(item)

        if self.transforms is not None:
            item = self.transforms(item)
        return item


def reference_panel_collate(batch):
    ref_panel = []
    for x in batch:
        ref_panel.append(x["ref_panel"])
        del x["ref_panel"]

    batch = default_collate(batch)
    batch["ref_panel"] = ref_panel

    return batch
