import h5py
import numpy as np
import sys
import glob

if __name__ == '__main__':

    folder_name = sys.argv[1]

    outfile = h5py.File(folder_name + "/vcf_and_labels.h5", "w")
    # outfile = h5py.File("kk", "w")


    gen_folder = glob.glob(folder_name + "/gen_*")
    gen_folder.sort()

    all_data = []
    for gen in gen_folder:
        print(gen)
        all_data.append(np.load(gen + "/mat_vcf_2d.npy"))

    all_data = np.concatenate(all_data)
    outfile.create_dataset("vcf", data=all_data)

    all_data = []
    for gen in gen_folder:
        print(gen)
        all_data.append(np.load(gen + "/mat_map.npy"))

    all_data = np.concatenate(all_data)
    outfile.create_dataset("labels", data=all_data)
    del all_data

    outfile.close()

