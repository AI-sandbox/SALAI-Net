import h5py
import numpy as np
import sys
import glob

if __name__ == '__main__':

    folder_name = sys.argv[1]

    # outfile = h5py.File(folder_name + "/vcf_and_labels.h5", "w")

    gen_folder = glob.glob(folder_name + "/gen_*")
    gen_folder.sort()

    Ngens_per_dataset = 2

    gen_split = [gen_folder[Ngens_per_dataset*i:Ngens_per_dataset*(i+1)] for i in range(len(gen_folder) // Ngens_per_dataset)]

    for split in gen_split:

        lastgen_name = split[-1]
        lastgen_name = lastgen_name.split("/")[-1].replace("_", "")


        outfile = h5py.File(folder_name + "/vcf_and_labels_" + lastgen_name + ".h5", "w")

        all_data = []
        for gen in split:
            print(gen)
            all_data.append(np.load(gen + "/mat_vcf_2d.npy"))

        all_data = np.concatenate(all_data)
        outfile.create_dataset("vcf", data=all_data)

        all_data = []
        for gen in split:
            print(gen)
            all_data.append(np.load(gen + "/mat_map.npy"))

        all_data = np.concatenate(all_data)
        outfile.create_dataset("labels", data=all_data)
        del all_data

        outfile.close()

