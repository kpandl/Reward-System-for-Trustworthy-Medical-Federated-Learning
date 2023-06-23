import string
import os
import time

seeds = list(range(0,40))
#seeds = [16, 17, 18, 19]
seeds_per_job = 8

str_1="#!/bin/sh"+"\n"
str_2="nvidia-smi"+"\n"
str_3="python --version"+"\n"

splitted_seed_list = [seeds[i * seeds_per_job:(i + 1) * seeds_per_job] for i in range((len(seeds) + seeds_per_job - 1) // seeds_per_job )]

for i, value in enumerate(splitted_seed_list):

    str_7 = ""
    for j, seed in enumerate(value):
        str_7 += "python3 -u operative_run_07_compute_dataset_array_one_inst.py --seed=" + str(seed) + " --name_of_test_folder=default" + str(seed) + "_reduced_maxtranslation_0.1final_age_as_is --dscmode=28\n"

    with open("job_f_age_quantile_as_is_5_"+str(i)+".sh", "w+") as file:
        file.writelines([str_1, str_2, str_3, str_7])

    time.sleep(0.8)
    os.system("sbatch -p single --gres=gpu:1 --time=48:00:00 --mem=180gb --ntasks=32 job_f_age_quantile_as_is_5_" + str(i) + ".sh")
