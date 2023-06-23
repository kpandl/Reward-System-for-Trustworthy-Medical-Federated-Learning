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

    def create_federated_learning_environments(self, reverse_list=False, start_environment=0, end_environment=8, weighted_aggregation=False, combine_weighted_and_unweighted_aggregation=False, gender_filter="none", gendersetting=0, train_dataset_size_limit=-1, differentialprivacy=0, function_to_run_directly=None, client_ids_from_last_fle=[], ending_condition_mode="local_clients", use_specific_gpu=-1, skip_client_deepcopy=False, track_runtime=False):
        Path(os.path.join(os.getcwd(),'results',self.name)).mkdir(parents=True, exist_ok=True)
        self.FederatedLearningEnvironments = []

        fle_counter = 0

        runtime_tracker = 0

        for coalition in self.coalitions:
            coalition_list = list(coalition)
            if(len(coalition_list) > 0):
                if(fle_counter >= start_environment and fle_counter < end_environment):
                    if(function_to_run_directly==None):
                        self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode, use_specific_gpu=use_specific_gpu, skip_client_deepcopy=skip_client_deepcopy))
                    else:
                        last_fle=FederatedLearningEnvironment(list(self.coalitions[-1]), parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode)
                        specific_clients_from_last_fle = []
                        for client_id_from_last_fle in client_ids_from_last_fle:
                            specific_clients_from_last_fle.append(last_fle.client_list[client_id_from_last_fle])
                        start = time.time()
                        function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=weighted_aggregation, gender_filter=gender_filter, gendersetting=gendersetting, train_dataset_size_limit=train_dataset_size_limit, differentialprivacy=differentialprivacy, ending_condition_mode=ending_condition_mode), specific_clients=specific_clients_from_last_fle)
                        end = time.time()
                        runtime_tracker += end-start
                fle_counter += 1

        if(combine_weighted_and_unweighted_aggregation):
            for coalition in self.coalitions:
                coalition_list = list(coalition)
                if(len(coalition_list) >= 2):
                    if(fle_counter >= start_environment and fle_counter < end_environment):
                        if(function_to_run_directly==None):
                            self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=True, ending_condition_mode=ending_condition_mode))
                        else:
                            start = time.time()
                            function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, weighted_aggregation=True, ending_condition_mode=ending_condition_mode))
                            end = time.time()
                            runtime_tracker += end-start

                    fle_counter += 1
        
        if(fle_counter >= start_environment and fle_counter < end_environment):
            if(function_to_run_directly==None):
                self.FederatedLearningEnvironments.append(FederatedLearningEnvironment(coalition_list, parent_dir=self.name, merge_clients=True, ending_condition_mode=ending_condition_mode))
            else:
                start = time.time()
                function_to_run_directly(specific_fle=FederatedLearningEnvironment(coalition_list, parent_dir=self.name, merge_clients=True, ending_condition_mode=ending_condition_mode))
                end = time.time()
                runtime_tracker += end-start
        fle_counter += 1

        if(runtime_tracker > 0 and track_runtime):
            print("Runtime for creating FLEs: ", runtime_tracker and track_runtime)
            path_runtime_file = os.path.join(os.getcwd(),'results',self.name,"runtime_tracker_OR.csv")
            with open(path_runtime_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["task","time_seconds"])
                writer.writerow(["creating_artificial_models",runtime_tracker])

        for fle in self.FederatedLearningEnvironments:
            for client in fle.client_list:
                client.ending_condition_for_local_adaptation_reached = True

        if(reverse_list):
            self.FederatedLearningEnvironments.reverse()

        print(len(self.FederatedLearningEnvironments), "FederatedLearningEnvironments existing")

    def compute_shapley_values(self, based_on_predictive_performance=False, based_on_age=False, originally_sex_based=False):
        fle_utilities = {}
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


                    fle_utilities[fle.name] = utility_coalition


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


                    fle_utilities[extended_coalition_name] = utility_extended_coalition


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
            
            path_to_name_of_largest_coalition = os.path.join(os.getcwd(), "results", self.name, "last_fle_name.txt")
            with open(path_to_name_of_largest_coalition, 'r') as f:
                extended_coalition_name = f.read()


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
        print("len(fle_utilities)", len(fle_utilities))
        if(based_on_predictive_performance):
            path_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_OR_method_predictive_performance_utilities.pkl")
        if(not based_on_predictive_performance and not based_on_age):
            path_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_OR_method_bias_gender_utilities.pkl")
        with open(path_file, 'wb') as handle:
            pickle.dump(fle_utilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_federated_setting(self):

        for fle in self.FederatedLearningEnvironments:
            fle.run_learning_and_testing_process()

    def run_testing_setting(self, test_with_largest_coalition=False):

        if(not test_with_largest_coalition):
            for fle in self.FederatedLearningEnvironments:
                fle.run_testing_process()
        else:
            
            for fle in self.FederatedLearningEnvironments:
                largest_coalition = self.FederatedLearningEnvironments[-1]
                a = 0
                largest_coalition_clients = largest_coalition.client_list

                #path_global_learning_rounds = os.path.join(os.getcwd(), "results", self.name, fle.client_list[0].dataset_train.datasetName+fle.client_list[0].name_addition, "federated_training.csv")
                path_global_learning_rounds = os.path.join(os.getcwd(), "results", self.name, fle.name, "federated_training.csv")
                df_global_learning_rounds = pd.read_csv(path_global_learning_rounds)
                global_learning_round = df_global_learning_rounds["communication round"].max()-10
                path_model = os.path.join(os.getcwd(), "results", self.name, fle.name, "global_" + str(global_learning_round).zfill(3) + ".pt")

                a = 0
                fle.client_list[0].load_local_model_path(path_model, fle.global_model)
                for client in largest_coalition_clients:
                    c = 0
                    
                    client.set_model(fle.client_list[0].model)
                    folderpath_testing_general = os.path.join(os.getcwd(), "results", self.name, fle.name)
                    client.run_testing(name_of_aggregation="federated averaging", load_model=False, ending_condition_mode=fle.ending_condition_mode, folderpath_testing_general=folderpath_testing_general, fle_name=fle.name)
                    b = 0
                a = 0

    def create_artificial_models(self, specific_fle=None, specific_clients=None):

        path_to_name_of_largest_coalition = os.path.join(os.getcwd(), "results", self.name, "last_fle_name.txt")
        with open(path_to_name_of_largest_coalition, 'r') as f:
            name_of_largest_coalition = f.read()

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

    def create_deep_features(self):
        self.FederatedLearningEnvironments[-1].create_deep_features()

    def compute_knn_shapley_values_federated(self, filter="none", specific_label_id=-1, noise=False, only_even_clients=False, limit_collected_dataset=-1):
        self.FederatedLearningEnvironments[-1].compute_knn_shapley_values_federated(filter, specific_label_id=specific_label_id, noise=noise, only_even_clients=only_even_clients, limit_collected_dataset=limit_collected_dataset)
        
    def compute_local_LR_models(self):
        self.FederatedLearningEnvironments[-1].compute_local_LR_models()
        
    def compute_LR_SV(self, measure="predictive_performance"):
        fle_utilities = {}
        last_fle = self.FederatedLearningEnvironments[-1]

        N = len(self.clients)
        path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_LR_method_" + measure + ".csv")

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

                    utility_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list, last_fle.name, measure)

                    fle_utilities[fle.name] = utility_coalition

                    extended_coalition_name = fle.get_name_of_extended_coalition(client)

                    utility_extended_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list + [client], last_fle.name, measure)
                    fle_utilities[extended_coalition_name] = utility_extended_coalition

                    if(show_details):
                        print(extended_coalition_name, utility_extended_coalition)

                    sum_for_shapley += 1/(comb(N-1,len(fle.client_list)))*(utility_extended_coalition-utility_coalition)             

                # now empty set
                if(measure == "predictive_performance"):
                    utility_coalition = 0.5
                else:
                    utility_coalition = 0
                
                utility_extended_coalition = self.get_FederatedlearningEnvironment_utility([client], last_fle.name, measure)

                sum_for_shapley += 1/(comb(N-1,0))*(utility_extended_coalition-utility_coalition)
            
                sum_for_shapley /= N
                print("finished one client:", client.get_name_of_dataset_train_and_addition(), sum_for_shapley)

                with open(path_Shapley_results_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([client.get_name_of_dataset_train_and_addition(), testing_critereons[i], operating_points[i], sum_for_shapley])

                list_of_shapley_values.append(sum_for_shapley.tolist())
        print("len(fle_utilities)", len(fle_utilities))
        with open(os.path.join(os.getcwd(), "results", self.name, "Shapley_LR_method_" + measure + "_utilities.pkl"), 'wb') as handle:
            pickle.dump(fle_utilities, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def compute_LR_SV_coalitions(self, measure="predictive_performance"):
        self.file_buffer = {}
        fle_utilities = {}
        last_fle = self.FederatedLearningEnvironments[-1]

        testing_critereons = ["AUC"]
        operating_points = ["-"]

        start_time = time.time()

        for i in range(len(testing_critereons)):
            print("now considering", testing_critereons[i], operating_points[i])

            for fle in self.FederatedLearningEnvironments:
                utility_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list, last_fle.name, measure)
                fle_utilities[fle.name] = utility_coalition

        end_time = time.time()
        print("time for computing utilities:", end_time-start_time)
        time_passed = end_time-start_time
        # store the time in a csv file
        path_runtime_file = os.path.join(os.getcwd(),'results',self.name,"runtime_tracker_LR.csv")
        with open(path_runtime_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["task","time_seconds"])
            writer.writerow(["aggregating_LR_coalitions",time_passed])

        print("len(fle_utilities)", len(fle_utilities))
        with open(os.path.join(os.getcwd(), "results", self.name, "Shapley_LR_method_" + measure + "_coalition_utilities.pkl"), 'wb') as handle:
            pickle.dump(fle_utilities, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def delete_deep_feature_arrays(self):
        for client in self.clients:
            client.delete_deep_feature_arrays()


    def compute_LR_SV_2(self, measure="predictive_performance"):
        last_fle = self.FederatedLearningEnvironments[-1]

        N = len(self.clients)
        path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_LR2_method_" + measure + ".csv")

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
                    utility_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list, last_fle.name, measure)
                    #utility_coalition = self.get_FederatedlearningEnvironment_utility_original(fle.name, measure)

                    extended_coalition_name = fle.get_name_of_extended_coalition(client)

                    utility_extended_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list+[client], last_fle.name, measure)

                    if(show_details):
                        print(extended_coalition_name, utility_extended_coalition)

                    sum_for_shapley += 1/(comb(N-1,len(fle.client_list)))*(utility_extended_coalition-utility_coalition)             

                # now empty set
                if(measure == "predictive_performance"):
                    utility_coalition = 0.5
                else:
                    utility_coalition = 0
                                
                utility_extended_coalition = self.get_FederatedlearningEnvironment_utility([client], last_fle.name, measure)

                sum_for_shapley += 1/(comb(N-1,0))*(utility_extended_coalition-utility_coalition)
            
                sum_for_shapley /= N
                print("finished one client:", client.get_name_of_dataset_train_and_addition(), sum_for_shapley)

                with open(path_Shapley_results_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([client.get_name_of_dataset_train_and_addition(), testing_critereons[i], operating_points[i], sum_for_shapley])

                list_of_shapley_values.append(sum_for_shapley.tolist())

                show_details=False
            
    def compute_original_SV(self, measure="predictive_performance"):
        fle_utilities = {}
        last_fle = self.FederatedLearningEnvironments[-1]

        N = len(self.clients)
        path_Shapley_results_file = os.path.join(os.getcwd(), "results", self.name, "Shapley_original_method_" + measure + ".csv")

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
                    #utility_coalition = self.get_FederatedlearningEnvironment_utility(fle.client_list, last_fle.name, measure)
                    utility_coalition = self.get_FederatedlearningEnvironment_utility_original(fle.name, measure)

                    fle_utilities[fle.name] = utility_coalition

                    extended_coalition_name = fle.get_name_of_extended_coalition(client)

                    utility_extended_coalition = self.get_FederatedlearningEnvironment_utility_original(extended_coalition_name, measure)
                    fle_utilities[extended_coalition_name] = utility_extended_coalition

                    if(show_details):
                        print(extended_coalition_name, utility_extended_coalition)

                    sum_for_shapley += 1/(comb(N-1,len(fle.client_list)))*(utility_extended_coalition-utility_coalition)             

                # now empty set
                if(measure == "predictive_performance"):
                    utility_coalition = 0.5
                else:
                    utility_coalition = 0
                
                utility_extended_coalition = self.get_FederatedlearningEnvironment_utility_original(client.dataset_train.datasetName+client.name_addition, measure)

                sum_for_shapley += 1/(comb(N-1,0))*(utility_extended_coalition-utility_coalition)
            
                sum_for_shapley /= N
                print("finished one client:", client.get_name_of_dataset_train_and_addition(), sum_for_shapley)

                with open(path_Shapley_results_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerow([client.get_name_of_dataset_train_and_addition(), testing_critereons[i], operating_points[i], sum_for_shapley])

                list_of_shapley_values.append(sum_for_shapley.tolist())

                show_details=False
        print("len(fle_utilities)", len(fle_utilities))
        with open(os.path.join(os.getcwd(), "results", self.name, "Shapley_original_method_" + measure + "_utilities.pkl"), 'wb') as handle:
            pickle.dump(fle_utilities, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_FederatedlearningEnvironment_utility_generalized(self, method, measure, fle, largest_fle):
        pass

    def get_FederatedlearningEnvironment_utility(self, client_list, largest_coalition_name, measure):
        accumulated_test_results = None
        first_client = True

        client_names_total = ""
        for client in client_list:
            client_names_total += client.get_name_of_dataset_train_and_addition() + " "

        print("client_names_total", client_names_total)

        path_true_labels = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, client_list[0].get_name_of_dataset_train_and_addition(), "LR_model_test_results_corresponding_labels.pkl")

        if(not path_true_labels in self.file_buffer.keys()):
            with open(path_true_labels, 'rb') as handle:
                self.file_buffer[path_true_labels] = pickle.load(handle)
        dict_true_labels = self.file_buffer[path_true_labels]
        
        st = SubgroupTest(None, None, None)
        
        aucs_test_prediction = []
        for key in dict_true_labels:
            first_client=True
            
            if(measure == "bias_gender"):
                path_genders = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, key, "genders_dataset_test.npz")
                
                if(not path_genders in self.file_buffer.keys()):
                    self.file_buffer[path_genders] = np.load(path_genders, allow_pickle=True)["arr_0"]
                genders_test_dataset = self.file_buffer[path_genders]

                index_male = []
                index_female = []

                for index, gender in enumerate(genders_test_dataset):
                    if(gender == "F"):
                        index_female.append(index)
                    if(gender == "M"):
                        index_male.append(index)

                male_gt=np.take(dict_true_labels[key],index_male,0)
                female_gt=np.take(dict_true_labels[key],index_female,0)

            if(measure == "bias_age"):
                path_ages = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, key, "ages_dataset_test.npz")
                
                if(not path_ages in self.file_buffer.keys()):
                    self.file_buffer[path_ages] = np.load(path_ages, allow_pickle=True)["arr_0"]
                ages_test_dataset = self.file_buffer[path_ages]

                index_young = []
                index_old = []

                for index, age in enumerate(ages_test_dataset):
                    
                    if("nih" in key):
                        threshold_young = 38
                        threshold_old = 57

                    if("chexpert" in key or "mimic" in key):
                        threshold_young = 52
                        threshold_old = 71

                    if(age != None and age <= threshold_young):
                        index_young.append(index)
                    if(age != None and age >= threshold_old):
                        index_old.append(index)

                young_gt=np.take(dict_true_labels[key],index_young,0)
                old_gt=np.take(dict_true_labels[key],index_old,0)

            for client in client_list:
                path_client_results = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, client_list[0].get_name_of_dataset_train_and_addition(), "LR_model_test_results.pkl")
                
                if(not path_client_results in self.file_buffer.keys()):
                    with open(path_client_results, 'rb') as handle:
                        self.file_buffer[path_client_results] = pickle.load(handle)
                dict_client_test_results = self.file_buffer[path_client_results]

                if(first_client):
                    accumulated_client_test_results = dict_client_test_results[key]
                else:
                    accumulated_client_test_results = np.add(accumulated_client_test_results, dict_client_test_results[key])

                first_client = False

            if(measure == "predictive_performance"):
                aucs_for_test_dataset = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(dict_true_labels[key]))), torch.from_numpy(dict_true_labels[key]), torch.from_numpy(accumulated_client_test_results))
                print("aucs_for_test_dataset", aucs_for_test_dataset)
                aucs_test_prediction.append(aucs_for_test_dataset)

            if(measure == "bias_gender"):
                #gt_male = np.take(dict_true_labels[key],index_male,0)
                #gt_female = np.take(dict_true_labels[key],index_female,0)
                predictions_male = np.take(accumulated_client_test_results, index_male, 0)
                predictions_female = np.take(accumulated_client_test_results, index_female, 0)

                aucs_male = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(male_gt))), torch.from_numpy(male_gt), torch.from_numpy(predictions_male))
                aucs_female = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(female_gt))), torch.from_numpy(female_gt), torch.from_numpy(predictions_female))

                bias = np.mean(aucs_female) - np.mean(aucs_male)
                aucs_test_prediction.append(bias)

            if(measure == "bias_age"):
                predictions_young = np.take(accumulated_client_test_results, index_young, 0)
                predictions_old = np.take(accumulated_client_test_results, index_old, 0)

                aucs_young = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(young_gt))), torch.from_numpy(young_gt), torch.from_numpy(predictions_young))
                aucs_old = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(old_gt))), torch.from_numpy(old_gt), torch.from_numpy(predictions_old))

                bias = np.mean(aucs_young) - np.mean(aucs_old)
                aucs_test_prediction.append(bias)
        
        mean_result = np.mean(np.mean(aucs_test_prediction))

       

        # largest_coalition_name
        if(len(client_list) == 6):
            print("get_FederatedlearningEnvironment_utility, requesting results for coalition of size 6")
            
            path_df = os.path.join(os.getcwd(),'results',self.name, "testing.csv")
            # check if path exists
            if(os.path.exists(path_df)):
                df = pd.read_csv(path_df)
            else:
                path_df = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, "overall_testing.csv")
                if(os.path.exists(path_df)):
                    df = pd.read_csv(path_df)
                else:
                    path_df = os.path.join(os.getcwd(),'results',self.name, largest_coalition_name, "testing.csv")
                    df = pd.read_csv(path_df)


            if(measure == "predictive_performance"):
                row = df.loc[(df["Coalition"]==largest_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="-") & (df["Model type"]=="global") & (df["Subgroup name"]=="All") & (df["Testing criteron"]=="AUC") & (df["Operating point"]=="-")]
                mean_result = row["Average"].mean()

            elif(measure =="bias_gender"):
                row_female = df.loc[(df["Coalition"]==largest_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Female") & (df["Testing criteron"]=="AUC") & (df["Operating point"]=="-")]
                row_male = df.loc[(df["Coalition"]==largest_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Gender") & (df["Model type"]=="global") & (df["Subgroup name"]=="Male") & (df["Testing criteron"]=="AUC") & (df["Operating point"]=="-")]
                mean_result = row_female["Average"].mean() - row_male["Average"].mean()

            elif(measure =="bias_age"):
                row_young = df.loc[(df["Coalition"]==largest_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & ((df["Subgroup name"]=="0-38") | (df["Subgroup name"]=="0-52")) & (df["Testing criteron"]=="AUC") & (df["Operating point"]=="-")]
                row_old = df.loc[(df["Coalition"]==largest_coalition_name) & ((df["Aggregation type"]=="none") | (df["Aggregation type"]=="federated averaging")) & (df["Subgroup critereon"]=="Age") & (df["Model type"]=="global") & ((df["Subgroup name"]=="57+") | (df["Subgroup name"]=="71+")) & (df["Testing criteron"]=="AUC") & (df["Operating point"]=="-")]
                mean_result = row_young["Average"].mean() - row_old["Average"].mean()


            #print("row[Average]", row["Average"])
            print("mean_result", mean_result)

        print("mean_result", mean_result)

        return mean_result
   
    def get_FederatedlearningEnvironment_utility_original(self, coalition_name, measure):
        path_test_results = os.path.join(os.getcwd(),'results',self.name, coalition_name, "testing.csv")
        path_test_results = os.path.join(os.getcwd(),'results',self.name, coalition_name, "overall_testing.csv")
        df_test_results = pd.read_csv(path_test_results)


        if(measure == "predictive_performance"):
            df_filtered = df_test_results[(df_test_results["Coalition"]==coalition_name) & (df_test_results["Subgroup critereon"]=="-") & (df_test_results["Subgroup name"]=="All")]
            mean_result = df_filtered["Average"].mean()

        if(measure == "bias_gender"):
            df_filtered_female = df_test_results[(df_test_results["Coalition"]==coalition_name) & (df_test_results["Subgroup critereon"]=="Gender") & (df_test_results["Subgroup name"]=="Female")]
            df_filtered_male = df_test_results[(df_test_results["Coalition"]==coalition_name) & (df_test_results["Subgroup critereon"]=="Gender") & (df_test_results["Subgroup name"]=="Male")]
            mean_result_female = df_filtered_female["Average"].mean()
            mean_result_male = df_filtered_male["Average"].mean()
            mean_result = mean_result_female - mean_result_male

        print("get_FederatedlearningEnvironment_utility_original", mean_result)

        return mean_result