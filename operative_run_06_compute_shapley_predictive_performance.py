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
import globals

parser=argparse.ArgumentParser()
parser.add_argument('--seed', nargs='?', default=0, type=int, help='Specify the random seed')
parser.add_argument('--start_environment', nargs='?', default=0, type=int, help='Specify the start federated learning environment')
parser.add_argument('--end_environment', nargs='?', default=63, type=int, help='Specify the end federated learning environment')
parser.add_argument('--name_of_test_folder', nargs='?', default="default0_reduced_maxtranslation_0.1goodgendersetting_15k", type=str, help='Specify the name of the test folder')
parser.add_argument('--dscmode', nargs='?', default=3, type=int, help='Specify the dataset collection mode')

parser.add_argument('--non_weighted_aggregation', dest='weighted_aggregation', action='store_true')
parser.add_argument('--weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=False)

parser.add_argument('--combine_weighted_and_unweighted_aggregation', dest='combine_weighted_and_unweighted_aggregation', action='store_true')
parser.set_defaults(combine_weighted_and_unweighted_aggregation=True)

args=parser.parse_args()

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

dsc = DataSetCollection(args.seed, ds_list, 0, 0.8, 0.8, 0.9, 0.9, 1, mode=args.dscmode)
with open('config.json') as config_file:
    config = json.load(config_file)
sc = ShapleyCompute(name=args.name_of_test_folder)

sc.set_clients(dsc.clients)
sc.compute_coalitions()
sc.create_federated_learning_environments(reverse_list=False, start_environment=args.start_environment, end_environment=args.end_environment, weighted_aggregation=args.weighted_aggregation, combine_weighted_and_unweighted_aggregation=args.combine_weighted_and_unweighted_aggregation)


if __name__ == "__main__": 
    globals.initialize()

    based_on_age = False
    if(args.dscmode == 13):
        based_on_age = True

    sc.compute_shapley_values(based_on_predictive_performance=True, based_on_age=based_on_age)