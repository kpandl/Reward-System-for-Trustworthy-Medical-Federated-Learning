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
from os import listdir
from os.path import isfile, join
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv

parser=argparse.ArgumentParser()
parser.add_argument('--name_of_test_folder', nargs='?', default="default0_reduced_maxtranslation_0.1final_as_is", type=str, help='Specify the name of the test folder')

args=parser.parse_args()

print("name_of_test_folder", args.name_of_test_folder)

path = os.path.join(os.getcwd(), "results", args.name_of_test_folder, "constructed_federated_models_tests")
csv_base_list = [f for f in listdir(path) if isfile(join(path, f))]

def mean_confidence_stats(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def mean_and_half_interval_to_text(means, half_intervals):
    result = []
    for i in range(len(means)):
        result.append(f'{means[i]:.3f}'+' ± '+f'{half_intervals[i]:.3f}')
    return result

df_list = []

column_names = ["Coalition", "Aggregation type", "Client", "Model type", "Subgroup critereon", "Subgroup name", "Testing criteron", "Operating point", "Test scan count", "Average", "Average without no finding", "atelectasis", "cardiomegaly", "consolidation", "edema", "no_finding", "pleural_effusion", "pneumonia", "pneumothorax"]
column_names_for_numerical_results = ["Test scan count", "Average", "Average without no finding", "atelectasis", "cardiomegaly", "consolidation", "edema", "no_finding", "pleural_effusion", "pneumonia", "pneumothorax"]

path = os.path.join(os.getcwd(), "results", args.name_of_test_folder, "constructed_federated_models_tests_with_fairness_age")
Path(path).mkdir(parents=True, exist_ok=True)

for csv_file in csv_base_list:
    if(csv_file == "overall_testing.csv"):
        continue
    print("file:", csv_file)
    df = pd.read_csv(os.path.join(os.getcwd(), "results", args.name_of_test_folder, "constructed_federated_models_tests", csv_file))
    
    df["Coalition"] = df['Coalition'].str.replace('_weighted_aggregation','')
    
    df_with_fairness_metrics = df.copy()

    for coalition in df['Coalition'].unique():
        print("coalition:", coalition)
        df_coalition = df.loc[(df["Coalition"]==coalition)]

        for aggregation_type in df_coalition['Aggregation type'].unique():
            df_aggregation_type = df_coalition.loc[(df_coalition["Aggregation type"]==aggregation_type)]

            for client in df_aggregation_type['Client'].unique():
                df_client = df_aggregation_type.loc[(df_aggregation_type["Client"]==client)]
                
                for model_type in df_client['Model type'].unique():
                    df_model_type = df_client.loc[(df_client["Model type"]==model_type)]
                
                    for subgroup_critereon in df_client['Subgroup critereon'].unique()[1:]:
                        if(subgroup_critereon != "Age"):
                            continue
                        df_subgroup_critereon = df_model_type.loc[(df_model_type["Subgroup critereon"]==subgroup_critereon)]
                
                        for testing_criteron in df_subgroup_critereon['Testing criteron'].unique():
                            df_testing_criteron = df_subgroup_critereon.loc[(df_subgroup_critereon["Testing criteron"]==testing_criteron)]

                            for operating_point in df_testing_criteron['Operating point'].unique():
                                df_operating_point = df_testing_criteron.loc[(df_testing_criteron["Operating point"]==operating_point)]

                                for i in range(int(len(df_operating_point)/2)):
                                    if(client=="nih"):
                                        young_item = df_operating_point[(df_operating_point["Subgroup name"]=="0-38")].iloc[i]
                                        old_item = df_operating_point[(df_operating_point["Subgroup name"]=="57+")].iloc[i]
                                    if(client=="chexpert" or client=="mimic"):
                                        young_item = df_operating_point[(df_operating_point["Subgroup name"]=="0-52")].iloc[i]
                                        old_item = df_operating_point[(df_operating_point["Subgroup name"]=="71+")].iloc[i]


                                    fairness_item = young_item.copy()

                                    for column in column_names_for_numerical_results:
                                        fairness_item[column] = young_item[column] - old_item[column]
                                    
                                    fairness_item["Subgroup name"] ="Young - Old"
                                    df_with_fairness_metrics = df_with_fairness_metrics.append(fairness_item)

    df_with_fairness_metrics.to_csv(os.path.join(os.getcwd(), "results", args.name_of_test_folder, "constructed_federated_models_tests_with_fairness_age", csv_file))