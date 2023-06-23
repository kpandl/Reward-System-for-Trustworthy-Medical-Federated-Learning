import numpy as np
import os
from pathlib import Path
from torchvision import  models
from torch import nn
import torchvision.transforms as transforms
from batchiterator import *
import time
import copy
import json
from shutil import copyfile
import time
import multiprocessing
import os.path
import csv
import math
from ThreadedCopy import *
import time
import pickle
from Client import *
from datetime import date
import datetime
from os import listdir
from os.path import isfile, join
import shutil
from densenet import DenseNet121

class FederatedLearningEnvironment:
    
    def __init__(self, client_list, parent_dir=None, merge_clients=False, weighted_aggregation=False, gender_filter="none", gendersetting=0, train_dataset_size_limit=-1, differentialprivacy=0, ending_condition_mode="local_clients", use_specific_gpu=-1, skip_client_deepcopy=False):

        if(not skip_client_deepcopy):
            self.client_list = copy.deepcopy(client_list)
        else:
            self.client_list = client_list
        if(merge_clients):
            for client in self.client_list[1:]:
                self.client_list[0].merge_client(client)

            self.client_list = [self.client_list[0]]

        self.parent_dir = parent_dir

        if(differentialprivacy > 0):
            for client in self.client_list:
                client.set_differentialprivacy(differentialprivacy)

        if(gender_filter != "none" or gendersetting > 0):
            for i, client in enumerate(self.client_list):
                if(gendersetting==1 and client.get_name_of_dataset_train() == "nih"):
                    gender_filter = "F"
                if(gendersetting==2 and client.get_name_of_dataset_train() == "nih"):
                    gender_filter = "M"
                if(gendersetting==1 and client.get_name_of_dataset_train() == "chexpert"):
                    gender_filter = "F"
                if(gendersetting==2 and client.get_name_of_dataset_train() == "chexpert"):
                    gender_filter = "M"
                if(gendersetting==1 and client.get_name_of_dataset_train() == "mimic"):
                    gender_filter = "M"
                if(gendersetting==2 and client.get_name_of_dataset_train() == "mimic"):
                    gender_filter = "F"
                print("gender filter train, before", len(client.dataset_train))
                client.dataset_train.scans = [scan for scan in client.dataset_train.scans if scan.corresponding_patient.gender==gender_filter]
                print("gender filter train, after", len(client.dataset_train), gender_filter)
                print("gender filter val, before", len(client.dataset_val))
                client.dataset_val.scans = [scan for scan in client.dataset_val.scans if scan.corresponding_patient.gender==gender_filter]
                print("gender filter val, after", len(client.dataset_val), gender_filter)
        
        if(train_dataset_size_limit != -1):
            client.dataset_train.scans = client.dataset_train.scans[:train_dataset_size_limit]
            client.dataset_val.scans = client.dataset_val.scans[:int(train_dataset_size_limit/5)]

        name_list = [client.get_name_of_dataset_train()+client.name_addition for client in client_list]
        name_list.sort()

        name = ""
        for item in name_list:
            name += item + "_"

        self.name = name[:-1]

        if(weighted_aggregation):
            self.name += "_weighted_aggregation"
        
        if(merge_clients):
            self.name += "_centralized"

        for client in self.client_list:
            client.set_parent_dir(os.path.join(os.getcwd(), "results", self.parent_dir, self.name))
            client.set_parent_coalition_name(self.name)

        self.weighted_aggregation = weighted_aggregation

        self.global_model = models.densenet121(pretrained=True)
        num_ftrs = self.global_model.classifier.in_features
        self.global_model.classifier = nn.Sequential(nn.Linear(num_ftrs, 8), nn.Sigmoid())

        self.average_epoch_loss_val = []

        if(len(self.client_list) == 1):
            print("with LR decay")
            self.client_list[0].local_learning_rate_decay = True

        self.ending_condition_mode = ending_condition_mode
        self.use_specific_gpu = use_specific_gpu

        print("created FederatedLearningEnvironment", self.name)

    def run_learning_and_testing_process(self):
        self.communication_round_number = 0
        Path(os.path.join(os.getcwd(),'results',self.parent_dir, self.name)).mkdir(parents=True, exist_ok=True)

        if(not os.path.isfile(os.path.join(os.getcwd(),"results", self.parent_dir, self.name, "finished.txt"))):

            state_dir = os.path.join(os.getcwd(),"results", self.parent_dir, self.name)

            filenames = next(walk(state_dir), (None, None, []))[2]
            filenames.sort()

            if(len(filenames) > 0):
                state = torch.load(os.path.join(state_dir, filenames[-1:][0]))
                self.global_model.load_state_dict(state["model_state_dict"])
                self.communication_round_number = state["communication_round_number"]

            for client in self.client_list:
                client.set_model(self.global_model)
                client.load_state()
                

            maximum_state = max([client.communication_round_number for client in self.client_list])
            self.communication_round_number = maximum_state

            for client in self.client_list:
                if(client.communication_round_number < self.communication_round_number):
                    client.prepare_for_next_round(self.communication_round_number)
                if(client.communication_round_number > 0 and not client.local_training_completed):
                    client.federated_round_local_train(model=self.global_model)
                    
            if(self.communication_round_number > 0):
                self.aggregate()
                    
            for client in self.client_list:
                if(client.communication_round_number > 0 and not client.local_validation_completed):
                    client.federated_round_global_model_validation(self.global_model)
                    
            for client in self.client_list:
                if(client.ending_condition_for_federated_learning_reached and not client.ending_condition_for_local_adaptation_reached):
                    client.run_local_adaptation(self.global_model)

            maximum_state = max([client.communication_round_number for client in self.client_list])
            minimum_state = min([client.communication_round_number for client in self.client_list])

            assert (maximum_state == minimum_state),"Client's don't reach the same state"

            self.communication_round_number = maximum_state

            ending_condition_list = [client.ending_condition_for_federated_learning_reached for client in self.client_list]
            
            if(self.ending_condition_mode=="local_clients"):
                ending_condition_for_loop_not_reached = sum(ending_condition_list) < len(self.client_list)
            if(self.ending_condition_mode=="global"):
                ending_condition_for_loop_not_reached = len(self.average_epoch_loss_val) < 10 or min(self.average_epoch_loss_val) in self.average_epoch_loss_val[-10:]

            while(ending_condition_for_loop_not_reached):
                for client in self.client_list:
                    if(not client.local_training_completed):
                        client.federated_round_local_train(model=self.global_model)
                self.aggregate()
                self.checkpoint(model=self.global_model)
                for client in self.client_list:
                    if(not client.local_validation_completed):
                        client.federated_round_global_model_validation(self.global_model)

                average_val_loss = sum([client.epoch_losses_local_adaptation_validation[-1] for client in self.client_list])/len(self.client_list)
                self.average_epoch_loss_val.append(average_val_loss)

                path = os.path.join(os.getcwd(),"results", self.parent_dir, self.name, "federated_training.csv")
                with open(path, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if (self.communication_round_number == 0):
                        logwriter.writerow(["communication round", "average val loss"])
                    logwriter.writerow([self.communication_round_number, average_val_loss])

                self.communication_round_number += 1
                for client in self.client_list:
                    client.prepare_for_next_round(self.communication_round_number)
                ending_condition_list = [client.ending_condition_for_federated_learning_reached for client in self.client_list]

                if(self.ending_condition_mode=="local_clients"):
                    ending_condition_for_loop_not_reached = sum(ending_condition_list) < len(self.client_list)
                if(self.ending_condition_mode=="global"):
                    ending_condition_for_loop_not_reached = len(self.average_epoch_loss_val) < 10 or min(self.average_epoch_loss_val) in self.average_epoch_loss_val[-10:]

            ending_condition_list = [client.ending_condition_for_local_adaptation_reached for client in self.client_list]
            while(sum(ending_condition_list) < len(self.client_list)):
                for client in self.client_list:
                    client.run_local_adaptation(self.global_model)
                    ending_condition_list = [client.ending_condition_for_local_adaptation_reached for client in self.client_list]

            with open(os.path.join(os.getcwd(),"results", self.parent_dir, self.name, "finished.txt"), 'w') as f:
                f.write(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))

        print("entirely finished with federated learning for", self.parent_dir, self.name)

    def run_testing_process(self, load_model=True, skip_nonfemale_clients=False):
        name_of_aggregation = ""
        if(self.weighted_aggregation):
            name_of_aggregation = "federated weighted averaging"
        else:
            name_of_aggregation = "federated averaging"

        if(len(self.client_list) == 1):
            name_of_aggregation = "none"

        for client in self.client_list:
            if(skip_nonfemale_clients and client.name_addition != "_f"):
                continue
            client.set_model(self.global_model)
            print("now testing a client", client.get_name_of_dataset_train())
            client.run_testing(name_of_aggregation=name_of_aggregation, load_model=load_model, ending_condition_mode=self.ending_condition_mode)

    def checkpoint(self, model):

        print('saving checkpoint of global model for round', self.communication_round_number)
        state = {
            'model': model,
            'model_state_dict': model.state_dict(),
            'communication_round_number': self.communication_round_number
        }
        path = os.path.join(os.getcwd(),"results", self.parent_dir, self.name, "global_"+str(self.communication_round_number).zfill(3)+".pt")
        torch.save(state, path)
        print("finished saving")

    def aggregate(self):
        local_models = [client.get_local_model_and_dataset_train_size()[0] for client in self.client_list]
        train_dataset_sizes = [client.get_local_model_and_dataset_train_size()[1] for client in self.client_list]
        
        global_model_copy = copy.deepcopy(self.global_model)
        global_dict = global_model_copy.state_dict()

        i = 0

        for k in global_dict.keys():
            if(not self.weighted_aggregation):
                global_dict[k] = torch.stack([local_models[i].state_dict()[k].float() for i in range(len(local_models))], 0).mean(0)
            else:
                global_dict[k] = torch.stack([len(train_dataset_sizes) * train_dataset_sizes[i] * local_models[i].state_dict()[k].float() / sum(train_dataset_sizes) for i in range(len(local_models))], 0).mean(0)
            i += 1
        self.global_model.load_state_dict(global_dict)

    def create_artificial_model(self, name_of_largest_coalition):
        Path(os.path.join(os.getcwd(),"results", self.parent_dir, "constructed_federated_models")).mkdir(parents=True, exist_ok=True)
        path_artificial_model_output = os.path.join(os.getcwd(),"results", self.parent_dir, "constructed_federated_models", self.name + ".pt")
        path_largest_coalition = os.path.join(os.getcwd(),"results", self.parent_dir, name_of_largest_coalition)

        print("path_largest_coalition", path_largest_coalition)

        model_synthetic = models.densenet121(pretrained=True)
        num_ftrs_model_synthetic = model_synthetic.classifier.in_features
        model_synthetic.classifier = nn.Sequential(nn.Linear(num_ftrs_model_synthetic, 8), nn.Sigmoid())
        model_synthetic_state_dict = model_synthetic.state_dict()

        model_synthetic_original = copy.deepcopy(model_synthetic)

        model_previous_round = models.densenet121(pretrained=True)
        num_ftrs_model_previous_round = model_previous_round.classifier.in_features
        model_previous_round.classifier = nn.Sequential(nn.Linear(num_ftrs_model_previous_round, 8), nn.Sigmoid())

        df_global_largest_coalition = pd.read_csv(os.path.join(path_largest_coalition, "federated_training.csv"))
        best_epoch_global = df_global_largest_coalition["communication round"][df_global_largest_coalition["average val loss"].idxmin()]

        files_global_models = ["global_" + str(i).zfill(3) + ".pt" for i in range(best_epoch_global+1)]

        if(self.name == name_of_largest_coalition):
            shutil.copyfile(join(path_largest_coalition, files_global_models[-1]), path_artificial_model_output)
        else:
            for i in range(len(files_global_models)):
                round_number_this_round = i
                round_number_last_round = round_number_this_round - 1

                file_name_global = "global_" + str(round_number_last_round).zfill(3) + ".pt"

                if(i > 0):
                    global_model_previous_round_state_dict = torch.load(join(path_largest_coalition, file_name_global), map_location=torch.device('cpu'))["model_state_dict"]
                else:
                    global_model_previous_round_state_dict = model_synthetic_state_dict

                model_previous_round.load_state_dict(global_model_previous_round_state_dict)

                gradient_list = []

                start = time.time()

                for client in self.client_list:
                    client.load_local_model_from_round_number_or_dict(model_synthetic_original, path_largest_coalition, i, 0)
                    start2 = time.time()
                    gradient = client.compute_gradients_from_previous_global_model_and_local_model(model_previous_round)
                    end2 = time.time()
                    print("gradient computation", end2 - start2)
                    gradient_list.append(gradient)
                
                end = time.time()
                print("in loop over clients", end - start)

                start = time.time()
                for key in model_synthetic.state_dict():
                    computation = sum([gradient[key] for gradient in gradient_list])
                    computation = computation / len(gradient_list)
                    divisor = math.exp(i/10)
                    if(model_synthetic_state_dict[key].dtype==torch.int64):
                        computation = computation.int()
                    model_synthetic_state_dict[key] += computation
                
                end = time.time()
                print("in loop over keys", end - start)

                print("finished for round", i)
            model_synthetic.load_state_dict(model_synthetic_state_dict)
            state = {
                    'model_state_dict': model_synthetic.state_dict()
                }
            torch.save(state, path_artificial_model_output)

    def test_artificial_model(self, list_of_test_clients):
        path_artificial_model_output = os.path.join(os.getcwd(),"results", self.parent_dir, "constructed_federated_models", self.name + ".pt")
        
        if(self.use_specific_gpu==-1):
            loaded_state = torch.load(path_artificial_model_output, map_location='cuda:0')
        else:
            loaded_state = torch.load(path_artificial_model_output, map_location='cuda:'+str(self.use_specific_gpu))

        print("loading model for test:",path_artificial_model_output)
        self.global_model.load_state_dict(loaded_state["model_state_dict"])

        self.client_list = list_of_test_clients

        for client in self.client_list:
            client.set_parent_dir(os.path.join(os.getcwd(), "results", self.parent_dir, self.name))
            client.set_parent_coalition_name(self.name)

        self.run_testing_process(load_model=False)

    def get_name_of_extended_coalition(self, additional_client):

        client_list = copy.deepcopy(self.client_list)
        client_list.append(additional_client)
        
        name_list = [client.get_name_of_dataset_train()+client.name_addition for client in client_list]
        name_list.sort()

        name = ""
        for item in name_list:
            name += item + "_"

        name = name[:-1]

        return name

    def create_deep_features(self):
        for client in self.client_list:
            client.create_deep_features()
        a = 0

    def compute_knn_shapley_values(self):
        for client in self.client_list:
            for i in range(8):
                client.compute_knn_shapley_values(i)
        a = 0

    def compute_knn_shapley_values_federated(self, filter="none", specific_label_id=-1, noise=False, noise_seed=0, only_even_clients=False, limit_collected_dataset=-1):

        if(noise):
            np.random.seed(noise_seed)

        if(specific_label_id == -1):
            list_of_labels = list(range(8))
        else:
            list_of_labels = [specific_label_id]

        compute_time = 0
        total_time = 0

        start_total = time.time()
        
        if(limit_collected_dataset != -1 and specific_label_id == -1):
            list_of_labels = [0]

        for i in list_of_labels:
            var = 0
            j = 0
            
            for client in self.client_list:
                X_train_deep, X_test_deep_balanced, y_train_deep, y_test_deep_balanced = client.get_loaded_deep_features(i, filter)
                a = 0

                if(noise==True):
                    standard_deviations_X = np.std(X_train_deep, axis=0)

                    #first_values = []
                    #second_values = []
                    #for k in range(len(X_train_deep)):
                    #    first_values.append(X_train_deep[k][0][0][0])
                    #    second_values.append(X_train_deep[k][0][0][1])
                
                    for k in range(len(X_train_deep)):
                        X_train_deep[k] = X_train_deep[k] + np.random.normal(0, 100*standard_deviations_X)

                if(j==0):
                    X_train_deep_federated = X_train_deep
                    y_train_deep_federated = y_train_deep
                    X_test_deep_federated = X_test_deep_balanced
                    y_test_deep_federated = y_test_deep_balanced
                else:
                    X_train_deep_federated = np.concatenate((X_train_deep_federated, X_train_deep), axis=0)
                    y_train_deep_federated = np.concatenate((y_train_deep_federated, y_train_deep), axis=0)
                    if(not only_even_clients or (only_even_clients and j%2==0)):
                        X_test_deep_federated = np.concatenate((X_test_deep_federated, X_test_deep_balanced), axis=0)
                        y_test_deep_federated = np.concatenate((y_test_deep_federated, y_test_deep_balanced), axis=0)
                j += 1

            if(limit_collected_dataset != -1):
                print("limiting collected dataset to", limit_collected_dataset, "from original", len(X_train_deep_federated))
                X_train_deep_federated = X_train_deep_federated[:limit_collected_dataset]
                y_train_deep_federated = y_train_deep_federated[:limit_collected_dataset]

            start_compute = time.time()
            self.client_list[0].compute_knn_shapley_values_based_on_loaded_deep_features(X_train_deep_federated, X_test_deep_federated, y_train_deep_federated, y_test_deep_federated, i, filter, limit_collected_dataset=limit_collected_dataset)
            end_compute = time.time()
            compute_time += end_compute - start_compute

        end_total = time.time()
        total_time = end_total - start_total

        path_store_time = os.path.join(os.getcwd(), "results", self.parent_dir, self.name, "time_knn_shapley_values_federated.csv")
        if(not os.path.exists(path_store_time)):
            with open(path_store_time, 'w') as f:
                f.write("limit,compute_time,total_time\n")
        with open(path_store_time, 'a') as f:
            f.write(str(limit_collected_dataset)+","+str(compute_time)+","+str(total_time)+"\n")
        

    def compute_local_LR_models(self):
        #j = 0

        test_dataset_sizes = []

        dict_X_test_deep_balanced = {}
        dict_y_test_deep_balanced = {}
        dict_test_dataset_sizes = {}
            
        for client in self.client_list:
            print("loading deep features for client", client.name_addition)
            _, X_test_deep_balanced, _, y_test_deep_balanced = client.get_loaded_deep_features(0, "all", False)
            dict_X_test_deep_balanced[client.get_name_of_dataset_train_and_addition()] = X_test_deep_balanced
            dict_y_test_deep_balanced[client.get_name_of_dataset_train_and_addition()] = y_test_deep_balanced
            dict_test_dataset_sizes[client.get_name_of_dataset_train_and_addition()] = len(y_test_deep_balanced)
            #if(j==0):
            #    X_test_deep_federated = X_test_deep_balanced
            #    y_test_deep_federated = y_test_deep_balanced
            #else:
            #    X_test_deep_federated = np.concatenate((X_test_deep_federated, X_test_deep_balanced), axis=0)
            #    y_test_deep_federated = np.concatenate((y_test_deep_federated, y_test_deep_balanced), axis=0)
            #j += 1

        #for i in range(8):            
        for client in self.client_list:
            #X_train_deep, _, y_train_deep, _ = client.get_loaded_deep_features(i, filter="none")
            client.compute_LR_model(dict_X_test_deep_balanced, dict_y_test_deep_balanced, dict_test_dataset_sizes)

            #self.client_list[0].compute_knn_shapley_values_based_on_loaded_deep_features(X_train_deep_federated, X_test_deep_federated, y_train_deep_federated, y_test_deep_federated, i, filter)