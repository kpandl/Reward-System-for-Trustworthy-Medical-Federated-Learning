# Reward System for Trustworthy Medical Federated Learning - code for the experiments
 
This repository contains the Python code for the research article entitled "Reward Systems for Trustworthy Medical Federated Learning" currently under review. The experiments were conducted within a Python 3.7.11 environment on a high performance computing cluster, running on a Unix operating systems and running with the Slurm Workload Manager.

## Preparing to run the experiments

For the experiments, you need to download the three datasets NIH ChestX-ray8, CheXpert, and MIMIC-CXR. Then, you need to specify the storage location of these datasets in the "config.json" file, and run the "run_01_index_data.py" file to index the datasets.

The medical image scans have a higher resolution than the convolutional neural network input requires (256x256 pixels). To prevent data loading being a bottleneck, we re-scaled and re-stored the images using the "run_02_resize_scans.py" file. To do this, you first should specify the storage location of the rescaled images in the "config_resized.json" file before. After running the Python file, you need to replace the the "config.json" content file with the content of the "config_resized.json" file and re-run the "run_01_index_data.py" file to re-index the datasets with the resized storage paths.

## Running the experiments on a high performance computing cluster

To run the experiments efficiently on a SLURM high performance computing cluster, we created Python programs that automatically submit multiple compute jobs. We used the "bwForCluster MLS&WISO" cluster, but the following files can be easily adapted for other SLURM and non-SLURM high performance computing clusters. In the standard setting we run each of the 8 data splits with 40 different random seeds.

Per data split out of the 8 data splits, we have different files, and the jobs each file submits need to be finished executing, before the next file can be run. For example, for the age-based 50/50 split, first run the 'run_03_submit_slurm_jobs_age_50_50_split_1_parallel.py' file. This parallel file submits jobs using multiple GPUs, alternatively, you can also use a sequential file. The file submits the computing jobs. After all of the jobs are finished successfully, you can run the corresponding next file 'run_03_submit_slurm_jobs_age_50_50_split_2', which creates the approximated federated learning models of different federated learning client coalitions. The next file 3 tests these models. Afterward, the 4th file computes bias metrics and Shapley values. The optional 5th file computes age-related bias and Shapley values for the sex-based 'as is' splits. 

## Running the experiments on a local computer

Alternatively, you can also run these files on your local computer, instead of a high performance computing cluster. For this, execute the files starting with 'operative_run' in the same order as stated in the code of the above mentioned files.

## Analyzing the results

You can then run "run_04_copy_files_into_folder.py" to copy the relevant result files into another folder entitled "compact_files". Then, you can download this smaller folder from the high performance computing cluster for further local results analysis, instead of the much larger folders otherwise. The following files ('run_05' and onwards) are not computationally intensive, and can be run on a local machine to compute and visualize final results.