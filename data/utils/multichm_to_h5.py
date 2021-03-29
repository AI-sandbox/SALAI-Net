import h5py
import numpy as np
import sys
import glob

if __name__ == '__main__':

    folder_name = sys.argv[1]

    outfile = h5py.File(folder_name + "/multigen_vcf_and_labels.h5", "w")
    # outfile = h5py.File("kk", "w")

    chm_folders = glob.glob(folder_name + "/chm*")
    chm_folders.sort()

    print(chm_folders)

    for chm_folder in chm_folders:

        chm = chm_folder.split("/")[-1]

        gen_folders = glob.glob(chm_folder + "/gen_*")
        gen_folders.sort()

        chm_data = []
        print(chm + " vcf")
        print(gen_folders)
        for gen in gen_folders:
            chm_data.append(np.load(gen + "/mat_vcf_2d.npy"))

        chm_data = np.concatenate(chm_data)
        group = outfile.create_group(chm)
        group.create_dataset("vcf", data=chm_data)

        chm_data = []
        print(chm + " labels")
        for gen_folder in gen_folders:
            chm_data.append(np.load(gen + "/mat_map.npy"))

        chm_data = np.concatenate(chm_data)
        group.create_dataset("labels", data=chm_data)
        del chm_data

    outfile.close()

