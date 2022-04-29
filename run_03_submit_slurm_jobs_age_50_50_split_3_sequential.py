import string
import os
import time

seeds = list(range(0,40))

alphabet = string.ascii_lowercase

str_1="#!/bin/sh"+"\n"
str_2="nvidia-smi"+"\n"
str_3="python --version"+"\n"

for i, value in enumerate(seeds):
    
    str_6="python -u operative_run_02_test_models.py --seed=" + str(value) + " --start_environment=" + str(62) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_age_quantile_50_50 --dscmode=26\n"
    str_7="python -u operative_run_04_test_approximated_models_optimized.py --seed=" + str(value) + " --start_environment=" + str(0) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_age_quantile_50_50 --dscmode=26 --age_clients\n"

    with open("job_f_age_quantile_50_50_3_"+str(value)+".sh", "w+") as file:
        file.writelines([str_1, str_2, str_3, str_6, str_7])

    time.sleep(0.8)
    os.system("sbatch -p gpu-single --cpus-per-gpu=16 --time=48:00:00 --mem=64000 --gres=gpu:1 " + "job_f_age_quantile_50_50_3_"+str(value)+".sh")