import string
import os
import time

seeds_start = 0
num_jobs = 1

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(12):
        line_list.append("python3 -u operative_run_10_LR_compute_SV.py --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.075 --measure=predictive_performance\n")

    with open("job_rewardsystems_labelflip_exp1_4_7.5p_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --ntasks=1 --time=24:00:00 --mem=100gb " + "job_rewardsystems_labelflip_exp1_4_7.5p_"+str(seeds_start+i*4)+".sh")