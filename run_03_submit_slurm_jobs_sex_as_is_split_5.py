import string
import os
import time

seeds = list(range(0,40))

alphabet = string.ascii_lowercase

str_1="#!/bin/sh"+"\n"
str_3="python --version"+"\n"

for i, value in enumerate(seeds):

    str_8="python -u operative_run_05_compute_bias_metrics_sex_based_as_is_split_age.py --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_as_is\n"
    str_9="python -u operative_run_06_compute_shapley_bias_age.py --seed=" + str(value) + " --start_environment=" + str(0) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_as_is --sex_based_split --dscmode=21\n"

    with open("job_f_as_is_4_"+str(value)+".sh", "w+") as file:
        file.writelines([str_1, str_3, str_8, str_9])

    time.sleep(0.8)
    os.system("sbatch -p single --cpus-per-task=2 --time=24:00:00 --mem=128000 " + "job_f_as_is_4_"+str(value)+".sh")