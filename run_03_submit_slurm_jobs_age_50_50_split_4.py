import string
import os
import time

seeds = list(range(0,40))

alphabet = string.ascii_lowercase

str_1="#!/bin/sh"+"\n"
str_3="python --version"+"\n"

for i, value in enumerate(seeds):

    str_8="python -u operative_run_05_compute_bias_metrics_age.py --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_age_quantile_50_50\n"
    str_9="python -u operative_run_06_compute_shapley_bias_age.py --seed=" + str(value) + " --start_environment=" + str(0) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_age_quantile_50_50 --dscmode=27\n"
    str_10="python -u operative_run_06_compute_shapley_predictive_performance.py --seed=" + str(value) + " --start_environment=" + str(0) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(value) + "_reduced_maxtranslation_0.1final_age_quantile_50_50 --dscmode=27\n"


    with open("job_f_age_quantile_50_50_4_"+str(value)+".sh", "w+") as file:
        file.writelines([str_1, str_3, str_8, str_9, str_10])

    time.sleep(0.8)
    os.system("sbatch -p single --cpus-per-task=2 --time=24:00:00 --mem=128000 " + "job_f_age_quantile_50_50_4_"+str(value)+".sh")