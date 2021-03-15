import h5py
import numpy as np
import sys

if __name__ == '__main__':

    folder_name = sys.argv[1]

    outfile = h5py.File(folder_name + "/vcf_and_labels.h5", "w")

    vcf_data = np.load(folder_name + "/mat_vcf_2d.npy")

    outfile.create_dataset("vcf", data=vcf_data)
    del vcf_data

    labels = np.load(folder_name + "/mat_map.npy")

    outfile.create_dataset("labels", data=labels)

    outfile.close()



