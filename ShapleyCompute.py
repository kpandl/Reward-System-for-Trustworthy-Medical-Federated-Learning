import numpy as np
import os
from pathlib import Path
import random
import json
import pandas as pd
import math
from Patient import *
from Scan import *
from torch.utils.data import Dataset
import torch
import imageio
#from scipy.misc import imread
from matplotlib.pyplot import imread
from PIL import Image
import PIL
from shutil import copyfile
from itertools import chain, combinations
from FederatedLearningEnvironment import *
from scipy.special import comb
import pickle
import csv
import globals

class ShapleyCompute:
    
    def __init__(self, name=None):
        self.name = name
        self.clients = []
        self.FederatedLearningEnvironments = []

    def add_client(self, client):
        self.clients.append(client)

    def set_clients(self, clients):
        self.clients = clients
      

    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def compute_coalitions(self):
        self.coalitions = list(self.powerset(self.clients))

    def create_federated_learning_environments(self, reverse_list=False, start_environment=0, end_environment=8, weighted_aggregation=False, combine_weighted_and_unweighted_aggregation=False, gender_filter="none", gendersetting=0, train_dataset_size_limit=-1, differentialprivacy=0, function_to_run_directly=None, client_ids_from_last_fle=[], ending_condition_mode="local_clients", use_specific_gpu=-1):
        Path(os.path.join(os.getcwd(),'results',self.name)).mkdir(parents=True, exist_ok=True)
        self.FederatedLearningEnvironments = []

        fle_counter = 0

        for coalition in self.coalitions:
            coalition_list = list(coalition)
            if(len(coalition_list) > 0):
                if(fle_counter >= start_environment and fle_counter < end_environment):
                    if(function_to_run_directly==None):
                        self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode, use_specific_gpu=use_specific_gpu))
                    else:
                        last_fle=FederatedLearningEnvironment(list(self.coalitions[-1]), parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode)
                        specific_clients_from_last_fle = []
                        for client_id_from_last_fle in client_ids_from_last_fle:
                            specific_clients_from_last_fle.append(last_fle.client_list[client_id_from_last_fle])
                        function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode), specific_clients=specific_clients_from_last_fle)
                fle_counter += 1

        if(combine_weighted_and_unweighted_aggregation):
            for coalition in self.coalitions:
                coalition_list = list(coalition)
                if(len(coalition_list) >= 2):
                    if(fle_counter >= start_environment and fle_counter < end_environment):
                        if(function_to_run_directly==None):
                            self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=True, ending_condition_mode=ending_condition_mode))
                        else:
                            function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=True, ending_condition_mode=ending_condition_mode))
                    fle_counter += 1
        
        if(fle_counter >= start_environment and fle_counter < end_environment):
            if(function_to_run_directly==None):
                self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, merge_clients=True, ending_condition_mode=ending_condition_mode))
            else:
                function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, merge_clients=True, ending_condition_mode=ending_condition_mode))
        fle_counter += 1

        for fle in self.FederatedLearningEnvironments:
            for client in fle.client_list:
                client.ending_condition_for_local_adaptation_reached = True

        if(reverse_list):
            self.FederatedLearningEnvironments.reverse()

        print(len(self.FederatedLearningEnvironments), "FederatedLearningEnvironments existing")

    def compute_shapley_values(self, based_on_predictive_performance=False, based_on_age=False, originally_sex_based=False):
        N = len(self.clients)
        if(not originally_sex_based):
            path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley.csv")
        else:
            path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_age.csv")
        if(based_on_predictive_performance):
            if(not originally_sex_based):
                path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_performance.csv")
            else:
                path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_performance_age.csv")

        with open(path_Shapley_results_file, 'a') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(["dataset", "testing critereon", "operating point", "Shapley value"])

        list_of_shapley_values = []

        print("N", N)

        testing_critereons = ["AUC"]
        operating_points = ["-"]

        show_details = False

        for i in range(len(testing_critereons)):

            print("now considering", testing_critereons[i], operating_points[i])

            for client in self.clients:
                print("client:", client.get_name_of_dataset_train_and_addition())
                sum_for_shapley = 0
                S_without_client = []

                for fle in self.FederatedLearningEnvironments:
                    if(not client.get_name_of_dataset_train_and_addition() in [cl.get_name_of_dataset_train_and_addition() for cl in fle.client_list]):
                        S_without_client.append(fle)

                for fle in S_without_client:
                    if(not originally_sex_based):
                        path_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness", fle.name+"_testing.csv")
                    else:
                        path_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness_age", fle.name+"_testing.csv")
                    
                    if(path_fle_test in globals.model_dict.keys()):
                        df = globals.model_dict[path_fle_test]
                    else:
                        df = pd.read_csv(path_fle_test)
                        globals.model_dict[path_fle_test] = df

                    if(not based_on_predictive_performance and not based_on_age):
                        row = df.loc[(df["Coalition"]==fle.name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Female - Male") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                    if(based_on_predictive_performance):
                        row = df.loc[(df["Coalition"]==fle.name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="-") & (df["Model type"]=="global") & (df["Subgroup name"]=="All") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                    if(not based_on_predictive_performance and based_on_age):
                        row = df.loc[(df["Coalition"]==fle.name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & (df["Subgroup name"]=="Young - Old") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]

                    utility_coalition = row["Average"].mean()
                    len_coalition = len(fle.client_list)

                    extended_coalition_name = fle.get_name_of_extended_coalition(client)

                    if(not originally_sex_based):
                        path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness", extended_coalition_name+"_testing.csv")
                    else:
                        path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness_age", extended_coalition_name+"_testing.csv")
                    
                    if(path_extended_fle_test in globals.model_dict.keys()):
                        df = globals.model_dict[path_extended_fle_test]
                    else:
                        df = pd.read_csv(path_extended_fle_test)
                        globals.model_dict[path_extended_fle_test] = df

                    if(not based_on_predictive_performance and not based_on_age):
                        row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Female - Male") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                    if(based_on_predictive_performance):
                        row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="-") & (df["Model type"]=="global") & (df["Subgroup name"]=="All") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                    if(not based_on_predictive_performance and based_on_age):
                        row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & (df["Subgroup name"]=="Young - Old") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]

                    utility_extended_coalition = row["Average"].mean()

                    if(show_details):
                        print(extended_coalition_name, utility_extended_coalition)

                    sum_for_shapley += 1/(comb(N-1,len(fle.client_list)))*(utility_extended_coalition-utility_coalition)             

                # now empty set
                if(not based_on_predictive_performance):
                    utility_coalition = 0
                if(based_on_predictive_performance):
                    utility_coalition = 0.5
                len_coalition = 0

                extended_coalition_name = client.get_name_of_dataset_train_and_addition()

                if(not originally_sex_based):
                    path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness", extended_coalition_name+"_testing.csv")
                else:
                    path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness_age", extended_coalition_name+"_testing.csv")
                df = pd.read_csv(path_extended_fle_test)
                if(not based_on_predictive_performance and not based_on_age):
                    row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Female - Male") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                if(based_on_predictive_performance):
                    row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="-") & (df["Model type"]=="global") & (df["Subgroup name"]=="All") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
                if(not based_on_predictive_performance and based_on_age):
                    row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & (df["Subgroup name"]=="Young - Old") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]

                utility_extended_coalition = row["Average"].mean()

                sum_for_shapley += 1/(comb(N-1,0))*(utility_extended_coalition-utility_coalition)
            
                sum_for_shapley /= N
                print("finished one client:", client.get_name_of_dataset_train_and_addition(), sum_for_shapley)

                with open(path_Shapley_results_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([client.get_name_of_dataset_train_and_addition(), testing_critereons[i], operating_points[i], sum_for_shapley])

                list_of_shapley_values.append(sum_for_shapley.tolist())

                show_details=False
            
            extended_coalition_name = "chexpert_f_chexpert_m_mimic_f_mimic_m_nih_f_nih_m"
            if(based_on_age):
                extended_coalition_name = "chexpert_old_chexpert_young_mimic_old_mimic_young_nih_old_nih_young"
                
            if(not originally_sex_based):
                path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness", extended_coalition_name+"_testing.csv")
            else:
                path_extended_fle_test = os.path.join(os.getcwd(), "results", self.name, "constructed_federated_models_tests_with_fairness_age", extended_coalition_name+"_testing.csv")
            df = pd.read_csv(path_extended_fle_test)
            if(not based_on_predictive_performance and not based_on_age):
                row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Female - Male") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
            if(based_on_predictive_performance):
                row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="-") & (df["Model type"]=="global") & (df["Subgroup name"]=="All") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]
            if(not based_on_predictive_performance and based_on_age):
                row = df.loc[(df["Coalition"]==extended_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & (df["Subgroup name"]=="Young - Old") & (df["Testing criteron"]==testing_critereons[i]) & (df["Operating point"]==operating_points[i])]

            utility_extended_coalition = row["Average"].mean()

            print("Total utility", utility_extended_coalition)
                
    def run_federated_setting(self):

        for fle in self.FederatedLearningEnvironments:
            fle.run_learning_and_testing_process()

    def run_testing_setting(self):

        for fle in self.FederatedLearningEnvironments:
            fle.run_testing_process()

    def create_artificial_models(self, specific_fle=None, specific_clients=None):

        name_of_largest_coalition = "chexpert_f_chexpert_m_mimic_f_mimic_m_nih_f_nih_m"

        if(specific_fle==None):
            for fle in self.FederatedLearningEnvironments:
                fle.create_artificial_model(name_of_largest_coalition)
        else:
            if("young" in specific_fle.name or "medium" in specific_fle.name or "old" in specific_fle.name):
                name_of_largest_coalition = "chexpert_old_chexpert_young_mimic_old_mimic_young_nih_old_nih_young"
            specific_fle.create_artificial_model(name_of_largest_coalition)

    def test_artificial_models(self, specific_fle=None, specific_clients=None):

        if(specific_fle==None and specific_clients==None):
            test_clients = [self.FederatedLearningEnvironments[-1].client_list[0], self.FederatedLearningEnvironments[-1].client_list[2], self.FederatedLearningEnvironments[-1].client_list[4]]
            for fle in self.FederatedLearningEnvironments:
                fle.test_artificial_model(test_clients)
        else:
            specific_fle.test_artificial_model(specific_clients)