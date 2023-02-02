import numpy
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import h5py
from torch.utils.data.dataloader import default_collate
import pandas as pd
import allel


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


def load_refpanel_from_h5py(reference_panel_h5):

    reference_panel_file = h5py.File(reference_panel_h5, "r")
    return reference_panel_file["vcf"], reference_panel_file["labels"]


def load_map_file(map_file):
    sample_map = pd.read_csv(map_file, sep="\t", header=None)
    sample_map.columns = ["sample", "ancestry"]
    ancestry_names, ancestry_labels = np.unique(sample_map['ancestry'], return_inverse=True)
    samples_list = np.array(sample_map['sample'])
    return samples_list, ancestry_labels, ancestry_names


def load_vcf_samples_in_map(vcf_file, samples_list):
    # Reading VCF
    vcf_data = allel.read_vcf(vcf_file)

    # Intersection between samples from VCF and samples from .map
    inter = np.intersect1d(vcf_data['samples'], samples_list, assume_unique=False, return_indices=True)
    samp, idx = inter[0], inter[1]

    # Filter only interecting samples
    snps = vcf_data['calldata/GT'].transpose(1, 2, 0)[idx, ...]
    samples = vcf_data['samples'][idx]

    # Save header info of VCF file
    info = {
        'chm': vcf_data['variants/CHROM'],
        'pos': vcf_data['variants/POS'],
        'id': vcf_data['variants/ID'],
        'ref': vcf_data['variants/REF'],
        'alt': vcf_data['variants/ALT'],
    }

    return samples, snps, info

def load_refpanel_from_vcfmap(reference_panel_vcf, reference_panel_samplemap):
    samples_list, ancestry_labels, ancestry_names = load_map_file(reference_panel_samplemap)
    samples_vcf, snps, info = load_vcf_samples_in_map(reference_panel_vcf, samples_list)

    argidx = np.argsort(samples_vcf)
    samples_vcf = samples_vcf[argidx]
    snps = snps[argidx, ...]

    argidx = np.argsort(samples_list)
    samples_list = samples_list[argidx]

    ancestry_labels = ancestry_labels[argidx, ...]

    # Upsample for maternal and paternal sequences
    ancestry_labels = np.expand_dims(ancestry_labels, axis=1)
    ancestry_labels = np.repeat(ancestry_labels, 2, axis=1)
    ancestry_labels = ancestry_labels.reshape(-1)

    samples_list_upsampled = []
    for sample_id in samples_list:
        for _ in range(2): samples_list_upsampled.append(sample_id)

    snps = snps.reshape(snps.shape[0] * 2, -1)

    return snps, ancestry_labels, samples_list_upsampled, ancestry_names, info

def vcf_to_npy(vcf_file):
    vcf_data = allel.read_vcf(vcf_file)
    snps = vcf_data['calldata/GT'].transpose(1, 2, 0)
    samples = vcf_data['samples']

    return snps, samples


class ReferencePanel:

    def __init__(self, reference_panel_vcf, reference_panel_labels, n_refs_per_class, samples_list=None):

        self.reference_vcf = reference_panel_vcf

        self.samples_list = samples_list
        self.shared_refpanel_cache = None

        reference_labels = reference_panel_labels
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
            # random.shuffle(reference_panel[ancestry])
            print(ancestry, len(reference_panel[ancestry]))

        self.reference_panel_index_dict = reference_panel

        self.n_refs_per_class = n_refs_per_class

    def sample_uniform_all_classes(self, n_sample_per_class):

        reference_samples = {}
        reference_samples_names = {}
        reference_samples_idx = {}

        for ancestry in self.reference_panel_index_dict.keys():
            ## If n_sample_per_class is smaller than zero we take them all.
            if 0 < n_sample_per_class < len(self.reference_panel_index_dict[ancestry]):
                n_samples = n_sample_per_class
            else:
                n_samples = len(self.reference_panel_index_dict[ancestry])
            indexes = random.sample(self.reference_panel_index_dict[ancestry],
                                    n_samples)
            reference_samples_idx[ancestry] = indexes
            reference_samples[ancestry] = []
            reference_samples_names[ancestry] = []
            for i in indexes:
                reference_samples[ancestry].append(self.reference_vcf[i])
                if self.samples_list is not None:
                    reference_samples_names[ancestry].append(self.samples_list[i])
                else:
                    reference_samples_names[ancestry].append(None)
            reference_samples = {x: np.array(reference_samples[x]) for x in
                                 reference_samples.keys()}

        return reference_samples, reference_samples_names, reference_samples_idx

    def take_all_refs(self):
        if self.shared_refpanel_cache is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            reference_samples = {}
            reference_samples_names = {}
            reference_samples_idx = {}

            for ancestry in self.reference_panel_index_dict.keys():

                n_samples = len(self.reference_panel_index_dict[ancestry])

                indexes = self.reference_panel_index_dict[ancestry]
                reference_samples_idx[ancestry] = indexes
                reference_samples[ancestry] = torch.tensor(self.reference_vcf[indexes]).to(device)
                reference_samples_names[ancestry] = [self.samples_list[i] for i in indexes]


                if self.samples_list is not None:
                    reference_samples_names[ancestry] = [self.samples_list[i] for i in indexes]
                else:
                    reference_samples_names[ancestry] = [None for i in indexes]

            self.reference_vcf = None
            self.shared_refpanel_cache = (reference_samples, reference_samples_names, reference_samples_idx)

        return self.shared_refpanel_cache

    def sample_reference_panel(self, shared_refpanel=False):
        if shared_refpanel:
            return self.take_all_refs()
        return self.sample_uniform_all_classes(n_sample_per_class=self.n_refs_per_class)

def ref_pan_to_tensor(item):

    item["mixed_vcf"] = torch.tensor(item["mixed_vcf"]).float()

    if "mixed_labels" in item.keys():
        item["mixed_labels"] = torch.tensor(item["mixed_labels"]).long()

    if "ref_panel" in item.keys():
        for c in item["ref_panel"]:
            item["ref_panel"][c] = torch.tensor(item["ref_panel"][c])

    return item


class ReferencePanelDataset(Dataset):
    def __init__(self, mixed_file_path, reference_panel_h5,
                 reference_panel_vcf, reference_panel_map,
                 n_refs_per_class, transforms, shared_refpanel=False):

        # if reference_panel_h5:
        if reference_panel_h5:
            print("Loading data from .h5 file", reference_panel_h5)
            reference_panel_snps, reference_panel_labels = load_refpanel_from_h5py(reference_panel_h5)
            ancestry_names = samples_list = info = None

        else:
            print("Loading data from .vcf file", reference_panel_vcf)
            reference_panel_snps, reference_panel_labels, samples_list, ancestry_names, info = load_refpanel_from_vcfmap(reference_panel_vcf, reference_panel_map)

        reference_panel_snps = reference_panel_snps * 2 - 1

        self.samples_list = samples_list
        self.ancestry_names = ancestry_names

        self.shared_refpanel = shared_refpanel
        if shared_refpanel: n_refs_per_class = -1
        self.reference_panel = ReferencePanel(reference_panel_snps, reference_panel_labels, n_refs_per_class, samples_list=samples_list)

        try:
            mixed_file = h5py.File(mixed_file_path)
            self.mixed_vcf = mixed_file["vcf"]
            self.mixed_labels = mixed_file["labels"]

        except:
            self.mixed_vcf, self.query_sample_names = vcf_to_npy(mixed_file_path)
            n_seq, n_chann, n_snps = self.mixed_vcf.shape
            self.mixed_vcf = self.mixed_vcf.reshape(n_seq * n_chann, n_snps)
            self.mixed_labels = None

        self.mixed_vcf = self.mixed_vcf * 2 - 1
        self.transforms = transforms
        self.info = info

    def __len__(self):
        return self.mixed_vcf.shape[0]

    def __getitem__(self, index):

        item = {
            "mixed_vcf": self.mixed_vcf[index].astype(float),
        }
        if self.mixed_labels is not None:
            item["mixed_labels"] = self.mixed_labels[index]

        if not self.shared_refpanel:
            item["ref_panel"], item['reference_names'], item['reference_idx'] = self.reference_panel.sample_reference_panel()

        item = ref_pan_to_tensor(item)

        if self.transforms is not None:
            item = self.transforms(item)

        return item

def reference_panel_collate(batch):
    if "ref_panel" in batch[0].keys():
        ref_panel = []
        reference_names = []
        reference_idx = []
        for x in batch:
            ref_panel.append(x["ref_panel"])
            reference_names.append(x["reference_names"])
            reference_idx.append(x['reference_idx'])
            del x["ref_panel"]
            del x["reference_names"]
            del x['reference_idx']
        batch["ref_panel"] = ref_panel


    batch = default_collate(batch)
    if "ref_panel" in batch.keys():
        batch["reference_names"] = reference_names
        batch["reference_idx"] = reference_idx

    return batch
