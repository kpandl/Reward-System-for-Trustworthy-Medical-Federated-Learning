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
        # first one normally uncomment
        line_list.append("python3 -u operative_run_08_compute_LR_models_federated_ensembling.py --seed=" + str(seeds_start+i*4+j) + " --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.05 --dscmode=21\n")
        line_list.append("python3 -u operative_run_09_compute_SV_LR_coalitions_federated_ensembling_3c.py --start_environment=0 --end_environment=63 --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.05 --dscmode=21 --skip_bias_gender\n")
        line_list.append("python3 -u operative_run_10_LR_compute_SV.py --name_of_test_folder=default" + str(seeds_start+i*4+j) + "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.05 --measure=predictive_performance\n")

    with open("job_rewardsystems_labelflip_exp1_3_5p_"+str(seeds_start+i*4)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236000 job_rewardsystems_labelflip_exp1_3_5p_" + str(seeds_start+i*4) + ".sh")