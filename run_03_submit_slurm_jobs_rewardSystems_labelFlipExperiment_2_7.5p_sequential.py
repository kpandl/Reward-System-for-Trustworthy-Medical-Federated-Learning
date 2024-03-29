import string
import os
import time

seeds_start = 0
num_jobs = 2

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(4):
        line_list.append("python -u operative_run_07_compute_dataset_array_one_inst_knn_labelflip.py --seed=" + str(seeds_start+i*4+j) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.075 --dscmode=20 --nolabel_flip_experiment --rewardsystemslabelflip=0.075 \n")

    with open("job_rewardsystems_labelflip_exp1_2_7.5p_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    os.system("sbatch -p single --cpus-per-gpu=64 --time=48:00:00 --mem=236000 --gres=gpu:1 " + "job_rewardsystems_labelflip_exp1_2_7.5p_"+str(seeds_start+i*4)+".sh")