import string
import os
import time

seeds_start = 0
num_jobs = 3

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    for j in range(4):
        line_list.append("python -u operative_run_01_compute_shapley_values_knn_labelflip.py --seed=" + str(seeds_start+i*4+j) + " --maximum_translation=0.1 --combine_weighted_and_unweighted_aggregation --folder_name_extension=_labelflip_40 --dscmode=40 --use_specific_gpu="+str(j)+" &\n")
    line_list.append("wait\n")

    for j in range(4):
        line_list.append("python -u operative_run_02_test_models.py --seed=" + str(seeds_start+i*4+j) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_labelflip_40 --dscmode=40 --use_specific_gpu="+str(j)+" --copy_only_nih_test &\n")
    line_list.append("wait\n")

    with open("job_labelflip_exp1_1_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #os.system("sbatch -p single --cpus-per-gpu=16 --time=48:00:00 --mem=236000 --gres=gpu:4 " + "job_labelflip_exp1_1_"+str(seeds_start+i*4)+".sh")