import numpy as np
import os
from pathlib import Path
import random
from DataSet import *
from ShapleyCompute import *
import pickle
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import  models
from torch import nn
import time
from batchiterator import *
import csv
from sklearn.metrics import roc_auc_score
import pandas as pd
import shutil
import time
from itertools import chain, combinations
from DataSetCollection import *
import argparse, sys

print("starting, testing GPUs")

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  
a = torch.zeros(4,3)    
a = a.to(device)

print("succesfully tested GPUs")

parser=argparse.ArgumentParser()

parser.add_argument('--seed', nargs='?', default=0, type=int, help='Specify the random seed')
parser.add_argument('--start_environment', nargs='?', default=2**18-2, type=int, help='Specify the start federated learning environment')
parser.add_argument('--end_environment', nargs='?', default=2**18-1, type=int, help='Specify the end federated learning environment')
parser.add_argument('--maximum_translation', nargs='?', default=0.1, type=float, help='Set maximum translation')
parser.add_argument('--gender_filter', nargs='?', default="none", type=str, help='Set gender filter')
parser.add_argument('--folder_name_extension', nargs='?', default="sex_as_is_knn_lableflip_42", type=str, help='Set folder name extension')
parser.add_argument('--gendersetting', nargs='?', default=0, type=int, help='Specify the gender setting')
parser.add_argument('--traindatasetsizelimit', nargs='?', default=-1, type=int, help='Specify the train dataset size limit')
parser.add_argument('--differentialprivacy', nargs='?', default=0, type=int, help='Specify the settings for differential privacy')
parser.add_argument('--dscmode', nargs='?', default=42, type=int, help='Specify the dataset collection mode')
parser.add_argument('--use_specific_gpu', nargs='?', default=-1, type=int, help='Specify the dataset collection mode')

parser.add_argument('--non_weighted_aggregation', dest='weighted_aggregation', action='store_true')
parser.add_argument('--weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=False)
parser.set_defaults(gendersetting1=True)

parser.add_argument('--label_flip_experiment', dest='label_flip_experiment', action='store_true')
parser.set_defaults(label_flip_experiment=True)

parser.add_argument('--combine_weighted_and_unweighted_aggregation', dest='combine_weighted_and_unweighted_aggregation', action='store_true')
parser.set_defaults(combine_weighted_and_unweighted_aggregation=True)

args=parser.parse_args()

print("seed", args.seed, "start_environment", args.start_environment, "end_environment", args.end_environment, "gender_filter", args.gender_filter)

remove_files = True

if(remove_files):
    try:
        start = time.time()
        shutil.rmtree(os.path.join("..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data"))
        end = time.time()
        print("removed files in", end - start, "seconds")
    except Exception:
        pass

with open(os.path.join(os.getcwd(),"data", "ds_nih.pkl"), 'rb') as f:
    ds1_nih = pickle.load(f)
    
with open(os.path.join(os.getcwd(),"data", "ds_chexpert.pkl"), 'rb') as f:
    ds1_chexpert = pickle.load(f)

with open(os.path.join(os.getcwd(),"data", "ds_mimic.pkl"), 'rb') as f:
    ds1_mimic = pickle.load(f)

ds_list = [ds1_nih, ds1_chexpert, ds1_mimic]

with open('config.json') as config_file:
    config = json.load(config_file)

if(config["device_name"] == "bwforcluster"):
    for ds in ds_list:
        for scan in ds.scans:
            scan.learning_path = os.path.join(os.environ['TMPDIR'], scan.learning_path)

    print("finished adapting learning paths")

if(args.differentialprivacy > 0):
    ds_list = [ds_list[0]]


dsc = DataSetCollection(args.seed, ds_list, 0, 0.8, 0.8, 0.9, 0.9, 1, maximum_translation=args.maximum_translation, mode=args.dscmode, use_specific_gpu=args.use_specific_gpu)
if(args.label_flip_experiment):
    dsc.flip_labels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
if(config["device_name"] == "bwunicluster" or config["device_name"] == "bwforcluster"):
    print("bwcluster, so copying files to local")
    dsc.copy_files_to_local(["train", "val"])
    print("bwcluster, finished copying files to local")
sc = ShapleyCompute(name="default"+str(args.seed)+"_reduced_maxtranslation_"+str(args.maximum_translation)+args.folder_name_extension)

sc.set_clients(dsc.clients)
sc.compute_coalitions()
sc.create_federated_learning_environments(reverse_list=False, start_environment=args.start_environment, end_environment=args.end_environment, weighted_aggregation=args.weighted_aggregation, combine_weighted_and_unweighted_aggregation=args.combine_weighted_and_unweighted_aggregation, gender_filter=args.gender_filter, gendersetting=args.gendersetting, train_dataset_size_limit=args.traindatasetsizelimit, differentialprivacy=args.differentialprivacy, ending_condition_mode="global", use_specific_gpu=args.use_specific_gpu)
sc.run_federated_setting()