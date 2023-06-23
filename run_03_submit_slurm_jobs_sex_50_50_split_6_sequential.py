import string
import os
import time

seeds = list(range(0,40))
seeds_per_job = 2

str_1="#!/bin/sh"+"\n"
str_2="nvidia-smi"+"\n"
str_3="python --version"+"\n"

splitted_seed_list = [seeds[i * seeds_per_job:(i + 1) * seeds_per_job] for i in range((len(seeds) + seeds_per_job - 1) // seeds_per_job )]

for i, value in enumerate(splitted_seed_list):

    str_7 = ""
    for j, seed in enumerate(value):
        str_7 += "python3 -u operative_run_08_compute_LR_models_federated_ensembling.py --seed=" + str(seed) + " --name_of_test_folder=default" + str(seed) + "_reduced_maxtranslation_0.1final_50_50 --dscmode=15\n"
        
        str_7 += "python3 -u operative_run_09_compute_SV_LR_coalitions_federated_ensembling_3c.py --start_environment=0 --end_environment=63 --name_of_test_folder=default" + str(seed) + "_reduced_maxtranslation_0.1final_50_50 --dscmode=15\n"
        str_7 += "python3 -u operative_run_10_LR_compute_SV.py --name_of_test_folder=default" + str(seed) + "_reduced_maxtranslation_0.1final_50_50 --measure=predictive_performance\n"
        str_7 += "python3 -u operative_run_10_LR_compute_SV.py --name_of_test_folder=default" + str(seed) + "_reduced_maxtranslation_0.1final_50_50 --measure=bias_gender\n"

    with open("job_f24_50_50_d_"+str(i)+".sh", "w+") as file:
        file.writelines([str_1, str_2, str_3, str_7])

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=248gb job_f24_50_50_d_" + str(i) + ".sh")
