import string
import os
import time

seeds_start = 0
max_job_seed = 61

job_seed_start = 0
job_seed_end = 0

while job_seed_start < max_job_seed:
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    lines = 0
    while job_seed_start < max_job_seed and lines < 4:
        line_list.append("python -u operative_run_01_compute_shapley_values.py --seed=" + str(0) + " --start_environment=" + str(job_seed_start) + " --end_environment=" + str(min(job_seed_start+3, max_job_seed+1)) + " --maximum_translation=0.1 --combine_weighted_and_unweighted_aggregation --folder_name_extension=final_100_0 --dscmode=6 --use_specific_gpu="+str(lines)+" &\n")
        lines += 1
        job_seed_start = min(job_seed_start+3, max_job_seed+1)
    line_list.append("wait\n")

    with open("job_allcoalitions_"+str(job_seed_start)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --gres=gpu:" + str(lines) + " --time=48:00:00 --mem=180gb --ntasks=32 " + "job_allcoalitions_"+str(job_seed_start)+".sh")