import string
import os
import time

seeds = list(range(0,40))

alphabet = string.ascii_lowercase

str_1="#!/bin/sh"+"\n"
str_2="nvidia-smi"+"\n"
str_3="python --version"+"\n"

for i, value in enumerate(seeds):
    str_4="python -u operative_run_01_compute_shapley_values.py --seed=" + str(value) + " --start_environment=" + str(62) + " --end_environment=" + str(63) + " --maximum_translation=0.1 --combine_weighted_and_unweighted_aggregation --folder_name_extension=final_50_50 --dscmode=14"+"\n"

    with open("job_f24_50_50_b_"+str(value)+".sh", "w+") as file:
        file.writelines([str_1, str_2, str_3, str_4])

    time.sleep(0.8)
    os.system("sbatch -p gpu-single --cpus-per-gpu=20 --time=48:00:00 --mem=180000 --gres=gpu:1 " + "job_f24_50_50_b_"+str(value)+".sh")