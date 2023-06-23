import string
import os
import time

seeds_start = 0
num_jobs = 12

for i in range(num_jobs):
    line_list = []
    line_list.append("#!/bin/sh"+"\n")
    line_list.append("nvidia-smi"+"\n")
    line_list.append("python --version"+"\n")

    line_list.append("python -u operative_run_08_compute_SVs_federated_knn_labelflip.py --seed=" + str(i) + " --start_environment=" + str(62) + " --end_environment=" + str(63) + " --combine_weighted_and_unweighted_aggregation --name_of_test_folder=default" + str(i) + "_reduced_maxtranslation_0.1_labelflip_40 --dscmode=41\n")
    
    with open("job_labelflip_exp1_3_"+str(i)+".sh", "w+") as file:
        file.writelines(line_list)

    time.sleep(0.8)
    #if(i != 0):
    os.system("sbatch -p single --ntasks=1 --time=72:00:00 --mem=236gb " + "job_labelflip_exp1_3_"+str(i)+".sh")