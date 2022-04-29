import string
import os
import time

seeds = list(range(0,40))

alphabet = string.ascii_lowercase

str_1="#!/bin/sh"+"\n"
str_2="nvidia-smi"+"\n"
str_3="python --version"+"\n"

for i, value in enumerate(seeds):

    str_5="python -u operative_run_03_create_approximated_models.py --seed=" + str(value) + " --start_environment=" + str(0) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_75_25 --dscmode=5\n"


    with open("job_f24_"+str(value)+".sh", "w+") as file:
        file.writelines([str_1, str_2, str_3, str_5])

    time.sleep(0.8)
    os.system("sbatch -p single --cpus-per-task=2 --time=72:00:00 --mem=128000 " + "job_f24_"+str(value)+".sh")
