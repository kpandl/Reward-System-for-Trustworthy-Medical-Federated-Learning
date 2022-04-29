import string
import os
import time

seeds_start = 0
num_jobs = 10

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(4):
        line_list.append("python -u operative_run_01_compute_shapley_values.py --seed=" + str(seeds_start+i*4+j) + " --start_environment=" + str(62) + " --end_environment=" + str(63) + " --maximum_translation=0.1 --combine_weighted_and_unweighted_aggregation --folder_name_extension=final_age_quantile_50_50 --dscmode=26 --use_specific_gpu="+str(j)+" &\n")

    line_list.append("wait\n")

    with open("job_f_age_50_50_1_parallel_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p gpu-single --cpus-per-gpu=16 --time=48:00:00 --mem=384000 --gres=gpu:4 " + "job_f_age_50_50_1_parallel_"+str(seeds_start+i*4)+".sh")