import string
import os
import time

seeds_start = 0
num_jobs = 12

line_list = []
line_list.append("#!/bin/sh"+"\n")
line_list.append("nvidia-smi"+"\n")
line_list.append("python --version"+"\n")

for i in range(num_jobs):
    line_list.append("python3 -u operative_run_09b_obtain_flipped_label_indices.py --seed=" + str(i) + " --maximum_translation=0.1 --folder_name_extension=_reduced_maxtranslation_0.1_labelflip_40\n")

with open("job_labelflip_exp1_5.sh", "w+") as file:
    file.writelines(line_list)

os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_labelflip_exp1_5.sh")