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
            n_samples = min(n_sample_per_class, len(self.reference_panel_index_dict[ancestry]))
            indexes = random.sample(self.reference_panel_index_dict[ancestry],
                                    n_samples)
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


class ReferencePanelMultiChmDataset(Dataset):

    def __init__(self, mixed_h5, reference_panel_h5, n_refs, n_classes,
                 samples_per_chm, transforms):

        # This dict tells which indices are associated to which chromosomes
        index_range_per_chm = {}
        last_chm = None
        for chm in samples_per_chm:
            if last_chm is None:
                index_range_per_chm[chm] = [0, samples_per_chm[chm] - 1]
            else:
                index_range_per_chm[chm] = [
                    index_range_per_chm[last_chm][-1] + 1,
                    index_range_per_chm[last_chm][-1] + samples_per_chm[chm]]
            last_chm = chm
        self.index_range_per_chm = index_range_per_chm

        reference_panel_file = h5py.File(reference_panel_h5, "r")
        mixed_file = h5py.File(mixed_h5)

        self.mixed_vcf = {}
        self.mixed_labels = {}
        for chm in mixed_file:
            self.mixed_vcf[chm] = mixed_file[chm]["vcf"]
            self.mixed_labels[chm] = mixed_file[chm]["labels"]

        self.reference_panel = {}
        for chm in reference_panel_file:
            self.reference_panel[chm] = ReferencePanel(
                reference_panel_file[chm], n_refs, n_classes)

        self.n_refs = n_refs
        self.n_classes = n_classes

        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError()
        return self.mixed_vcf.shape[0]

    def which_chm_and_index(self, index):
        for chm in self.index_range_per_chm.keys():
            if index >= self.index_range_per_chm[chm][0] and index <= \
                    self.index_range_per_chm[chm][1]:
                return chm, index - self.index_range_per_chm[chm][0]
        raise ValueError()

    def __getitem__(self, item):

        chm, item = self.which_chm_and_index(item)

        item = {
            "mixed_vcf": self.mixed_vcf[chm][item].astype(float),
            "mixed_labels": self.mixed_labels[chm][item]
        }

        item["ref_panel"] = self.reference_panel[chm].sample_reference_panel()

        item = ref_pan_to_tensor(item)

        if self.transforms is not None:
            item = self.transforms(item)

        return item


class SameChmSampler:

    def __init__(self, chm_samples, batch_size):

        self.batch_size = batch_size

        chm_samples = [chm_samples[x] for x in chm_samples.keys()]

        self.chm_start_indices = [0]
        for i, x in enumerate(chm_samples):
            self.chm_start_indices.append(self.chm_start_indices[i] + x)
        # print(self.chm_start_indices)
        self.n_chms = len(chm_samples)

        all_indices = list(range(self.chm_start_indices[-1]))

        self.chm_ranges = {}

        # Each element of the dict is a chromosome
        for i in range(self.n_chms):
            self.chm_ranges[i] = all_indices[self.chm_start_indices[i]:
                                             self.chm_start_indices[i + 1]]

    def __iter__(self):

        for i in range(self.n_chms):
            random.shuffle(self.chm_ranges[i])

        chm_ranges_split = {}
        for chm in range(self.n_chms):
            n_batches = len(self.chm_ranges[chm]) // self.batch_size
            chm_ranges_split[chm] = [self.chm_ranges[chm][i * self.batch_size:(i + 1) * self.batch_size]
                                     for i in range(n_batches)]

        # print(chm_ranges_split)

        all_slices = []
        for i in range(self.n_chms):
            all_slices += chm_ranges_split[i]
        random.shuffle(all_slices)
        for chm_slice in all_slices:
            yield chm_slice

def get_num_samples_per_chromosome(h5_file):
    h5_file = h5py.File(h5_file)
    samples_count = {}
    for chm in h5_file:
        samples_count[chm] = h5_file[chm]["vcf"].shape[0]
    print("samples per chromosome:", samples_count)
    return samples_count


def reference_panel_collate(batch):
    ref_panel = []
    for x in batch:
        ref_panel.append(x["ref_panel"])
        del x["ref_panel"]

    batch = default_collate(batch)
    batch["ref_panel"] = ref_panel

    return batch
