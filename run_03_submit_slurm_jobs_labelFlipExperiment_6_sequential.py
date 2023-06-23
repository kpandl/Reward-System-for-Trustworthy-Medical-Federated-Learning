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
    line_list.append("python3 -u operative_run_10_analyze_SVs_multilabel_federated_6clients.py --folder_path=default" + str(i) + "_reduced_maxtranslation_0.1_labelflip_40 --largest_coalition_name=mimic_0_mimic_1_mimic_2_mimic_3_mimic_4_mimic_5 --first_client_name=mimic_0\n")
    line_list.append("python3 -u operative_run_10_analyze_SVs_multilabel_federated_flipped.py --folder_path=default" + str(i) + "_reduced_maxtranslation_0.1_labelflip_40 --largest_coalition_name=mimic_0_mimic_1_mimic_2_mimic_3_mimic_4_mimic_5 --first_client_name=mimic_0\n")

with open("job_labelflip_exp1_6.sh", "w+") as file:
    file.writelines(line_list)

os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_labelflip_exp1_6.sh")