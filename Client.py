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
from matplotlib.pyplot import imread
from PIL import Image
import PIL
from shutil import copyfile
from os import walk
from torch import nn
import time
from batchiterator import *
import csv
import copy
from ThreadedCopy import *
from sklearn.metrics import roc_auc_score
import sklearn.metrics
import sklearn.metrics as sklm
from SubgroupTest import SubgroupTest
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from os.path import isfile, join
import globals

class Client:
    
  def __init__(self, dataset_train, dataset_val, dataset_test, name_addition="", use_specific_gpu=-1):
    self.dataset_train = dataset_train
    self.dataset_val = dataset_val
    self.dataset_test = dataset_test
    self.name_addition = name_addition
    self.use_specific_gpu=use_specific_gpu

    
    self.local_training_completed = False
    self.local_validation_completed = False
    self.local_adaptation_completed = False

    self.local_epoch_counter = 0

    self.ending_condition_for_federated_learning_reached = False
    self.ending_condition_for_local_adaptation_reached = False

    self.communication_round_number = 0

    self.optimizer = None
    self.epoch_losses_train = []
    self.epoch_losses_local_adaptation_training = []
    self.epoch_losses_local_adaptation_validation = []

    self.batch_size = 32

    self.optimizer_name = "SGD"
    self.LR = 0.01
    self.LR_local_adaptation = 0.01
    
    self.keep_local_optimizer = False
    self.local_epochs = 1

    self.epoch_loss_val_list = []

    self.local_learning_rate_decay = False

    self.differentialprivacy = 0
    
    
    print(torch.cuda.device_count(), 'GPUs available')
    if(use_specific_gpu==-1):
      if torch.cuda.is_available():
        self.device = torch.device("cuda")
      else:
        self.device = torch.device("cpu")
    else:
      self.device = torch.device("cuda:"+str(use_specific_gpu))

    self.subgroup_test_list = []

    self.subgroup_test_list.append(SubgroupTest("-", "All", lambda x: True))

    self.subgroup_test_list.append(SubgroupTest("Gender", "Female", lambda x: x.corresponding_patient.gender == "F"))
    self.subgroup_test_list.append(SubgroupTest("Gender", "Male", lambda x: x.corresponding_patient.gender == "M"))
    
    print("Test dataset name", self.dataset_test.datasetName)

    if(self.dataset_test.datasetName == "nih"):
      self.subgroup_test_list.append(SubgroupTest("Age", "0-38", lambda x: x.patient_age != None and x.patient_age >= 0 and x.patient_age <= 38))
      self.subgroup_test_list.append(SubgroupTest("Age", "57+", lambda x: x.patient_age != None and x.patient_age >= 57))

    if(self.dataset_test.datasetName == "chexpert" or self.dataset_test.datasetName == "mimic"):
      self.subgroup_test_list.append(SubgroupTest("Age", "0-52", lambda x: x.patient_age != None and x.patient_age >= 0 and x.patient_age <= 52))
      self.subgroup_test_list.append(SubgroupTest("Age", "71+", lambda x: x.patient_age != None and x.patient_age >= 71))

    if(self.get_name_of_dataset_train() == "mimic"):
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "White", lambda x: x.corresponding_patient.ethnicity == "WHITE"))
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "Asian", lambda x: x.corresponding_patient.ethnicity == "ASIAN"))
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "Black", lambda x: x.corresponding_patient.ethnicity == "BLACK/AFRICAN AMERICAN"))
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "Hispanic", lambda x: x.corresponding_patient.ethnicity == "HISPANIC/LATINO"))
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "Native", lambda x: x.corresponding_patient.ethnicity == "AMERICAN INDIAN/ALASKA NATIVE"))
      self.subgroup_test_list.append(SubgroupTest("Ethnicity", "Other", lambda x: x.corresponding_patient.ethnicity == "OTHER"))
      
      self.subgroup_test_list.append(SubgroupTest("Insurance", "Other", lambda x: x.corresponding_patient.insurance == "Other"))
      self.subgroup_test_list.append(SubgroupTest("Insurance", "Medicare", lambda x: x.corresponding_patient.insurance == "Medicare"))
      self.subgroup_test_list.append(SubgroupTest("Insurance", "Medicaid", lambda x: x.corresponding_patient.insurance == "Medicaid"))

  def get_name_of_dataset_train(self):
    return self.dataset_train.datasetName
    
  def get_name_of_dataset_train_and_addition(self):
    return self.dataset_train.datasetName + self.name_addition

  def set_differentialprivacy(self, differentialprivacy):
    self.differentialprivacy = differentialprivacy
    self.privacy_engine = PrivacyEngine()

  
  def checkpoint_train(self, folder_name, file_name, model):
    print('saving checkpoint of local model for dataset', self.get_name_of_dataset_train(), 'round', self.communication_round_number, 'and local epoch', self.local_epoch_counter)
    state = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'communication_round_number': self.communication_round_number,
        'local_epoch_counter': self.local_epoch_counter,
        'local_training_completed': self.local_training_completed,
        'epoch_losses_train': self.epoch_losses_train,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'LR_local_adaptation': self.LR_local_adaptation,
        'epoch_losses_local_adaptation_validation': self.epoch_losses_local_adaptation_validation
    }

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, folder_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    
    torch.save(state, os.path.join(path, file_name))
    print("finished saving")
  
  def checkpoint_validate(self, folder_name, file_name):
    print('saving checkpoint of validation of dataset', self.get_name_of_dataset_train(), 'round', self.communication_round_number)
    state = {
        'local_validation_completed': self.local_validation_completed,
        'epoch_loss_val_list': self.epoch_loss_val_list,
        'LR': self.LR
    }

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, folder_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    
    torch.save(state, os.path.join(path, file_name))
    print("finished saving")

  def local_train(self, model=None):
    start = time.time()
    criterion = nn.BCELoss().to(self.device)
    best_loss = 999999
    workers = 20
    train_loader = torch.utils.data.DataLoader(self.dataset_train,batch_size=self.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    optimizer = self.optimizer
    model = model.to(self.device)

    if(self.differentialprivacy > 0):
      model = ModuleValidator.fix(model)
      model, optimizer, train_loader = self.privacy_engine.make_private(
      module=model,
      optimizer=optimizer,
      data_loader=train_loader,
      noise_multiplier=1.1,
      max_grad_norm=1.0)
      model = model.to(self.device)

    train_df_size = len(self.dataset_train)
    print("train scans", train_df_size)
    phase = 'train'
    running_loss = BatchIterator(model=model, phase=phase, Data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=self.device, differentialprivacy=self.differentialprivacy)
    epoch_loss_train = running_loss / train_df_size
    self.epoch_losses_train.append(epoch_loss_train.item())
    end = time.time()
    print("one local trainig completed in", end - start, "seconds")

    if(self.differentialprivacy > 0):
      epsilon, best_alpha = self.privacy_engine.accountant.get_privacy_spent(
            delta=1/(10*train_df_size)
        )
      print(
          f"Train Epoch: ka \t"
          f"Loss: {np.mean(epoch_loss_train):.6f} "
          f"(ε = {epsilon:.2f}, δ = {1/(10*train_df_size)}) for α = {best_alpha}"
      )

    self.model = model
    return epoch_loss_train

  def federated_round_local_train(self, model=None):
    LR = self.LR

    self.model = copy.deepcopy(model)

    if(not self.keep_local_optimizer):
      if(self.optimizer_name == "Adam"):
        self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR)
      if(self.optimizer_name == "SGD"):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LR, momentum=0)

    while(self.local_epoch_counter < self.local_epochs):
      epoch_loss_train = self.local_train(model=self.model)

      path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "training_of_local_models.csv")
      with open(path, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        if (self.communication_round_number == 0 and self.local_epoch_counter == 0):
          logwriter.writerow(["communication round", "local epoch", "LR", "training loss"])
        logwriter.writerow([self.communication_round_number, self.local_epoch_counter, self.LR, epoch_loss_train.item()])

      if(self.local_epoch_counter == self.local_epochs - 1):
        self.local_training_completed = True
      self.checkpoint_train("federated_training", "_"+str(self.communication_round_number).zfill(3)+"_"+str(self.local_epoch_counter).zfill(3)+".pt", self.model)
      
      self.local_epoch_counter += 1


  def local_validation(self, model=None):
    model = model.to(self.device)

    start = time.time()

    criterion = nn.BCELoss().to(self.device)

    batch_size = 16
    workers = 16

    val_loader = torch.utils.data.DataLoader(self.dataset_val,batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    val_df_size = len(self.dataset_val)

    print("val scans", val_df_size)

    phase = 'val'
    optimizer = None
    running_loss = BatchIterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=self.device)
    epoch_loss_val = running_loss / val_df_size
    print("Validation_loss:", epoch_loss_val)

    end = time.time()
    print("one local validation completed in", end - start, "seconds")

    self.epoch_loss_val_list.append(epoch_loss_val.item())

    return epoch_loss_val.item()

  def federated_round_global_model_validation(self, model):
    validation_loss = self.local_validation(model)
    self.epoch_losses_local_adaptation_validation.append(validation_loss)

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "validation_of_global_model.csv")
    with open(path, 'a') as logfile:
      logwriter = csv.writer(logfile, delimiter=',')
      if (self.communication_round_number == 0):
        logwriter.writerow(["communication round", "validation loss"])
      logwriter.writerow([self.communication_round_number, validation_loss])
      
    self.local_validation_completed = True

    if(self.local_learning_rate_decay and min(self.epoch_loss_val_list) not in self.epoch_loss_val_list[-3:]):
      self.LR = self.LR / 2

    self.checkpoint_validate("federated_model_validation", str(self.communication_round_number).zfill(3)+".pt")
      

  def run_testing(self, name_of_aggregation="", load_model=True, ending_condition_mode="local_clients"):
    a = 0
    val_loss_adaptation = 99999
    testing_model_type = "global"

    if(load_model):
      if(ending_condition_mode=="local_clients"):
        df_global = pd.read_csv(os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "validation_of_global_model.csv"))
        my_file = Path(os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "adaptation_of_best_global_model.csv"))
        if my_file.is_file():
          df_adaptation = pd.read_csv(os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "adaptation_of_best_global_model.csv"))
          best_epoch_adaptation = df_adaptation["local_epoch"][df_adaptation["validation_loss"].idxmin()]
          val_loss_adaptation = df_adaptation["validation_loss"][best_epoch_adaptation]

        print("---")

        best_epoch_global = df_global["communication round"][df_global["validation loss"].idxmin()]
        val_loss_global = df_global["validation loss"][best_epoch_global]

        path = os.path.join(self.parent_dir, "global_"+str(best_epoch_global).zfill(3)+".pt")
      
      if(ending_condition_mode=="global"):
        a = 0
        df_global = pd.read_csv(os.path.join(self.parent_dir, "federated_training.csv"))
        best_epoch_global = df_global["communication round"][df_global["average val loss"].idxmin()]

        path = os.path.join(self.parent_dir, "global_"+str(best_epoch_global).zfill(3)+".pt")
        print("ending_condition_mode:", ending_condition_mode, "loading", path)
      
      loaded_state = torch.load(path)
      print("loading model for test:",path)
      self.model.load_state_dict(loaded_state["model_state_dict"])

    self.local_test(self.model, testing_model_type=testing_model_type, name_of_aggregation=name_of_aggregation, load_model=load_model)

  def local_test(self, model, testing_model_type="", name_of_aggregation="", load_model=True):
    print("testing", testing_model_type)
    test_loader = torch.utils.data.DataLoader(self.dataset_test,batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = model.to(self.device)

    model.eval()
    out_pred = torch.FloatTensor().to(self.device)  # tensor stores prediction values
    out_gt = torch.FloatTensor().to(self.device)  # tensor stores groundtruth values
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            out_gt = torch.cat((out_gt, target), 0)
            out_pred = torch.cat((out_pred, output), 0)

    val_loader = torch.utils.data.DataLoader(self.dataset_val,batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model.eval()
    out_pred_val = torch.FloatTensor().to(self.device)  # tensor stores prediction values
    out_gt_val = torch.FloatTensor().to(self.device)  # tensor stores groundtruth values
    i = 0
    with torch.no_grad():
      for data, target in val_loader:
        data, target = data.to(self.device), target.to(self.device)
        output = model(data)
        out_gt_val = torch.cat((out_gt_val, target), 0)
        out_pred_val = torch.cat((out_pred_val, output), 0)
        if(i == 0):
          print("out_gt_val", out_gt_val, "out_pred_val", out_pred_val)
        i += 1

    for i in range(8):
      gt = out_gt_val.to("cpu")[:,i].numpy().astype(int)
      pred = out_pred_val.to("cpu")[:,i].numpy()
      print("gt", gt)
      print("pred", pred)

    for subgroup_test in self.subgroup_test_list:
      subgroup_test.analyze_test(self.dataset_test.scans, out_gt.to("cpu"), out_pred.to("cpu"))

    a = 0

    if(load_model):
      path_client = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "testing.csv")
      path_overall = os.path.join(self.parent_dir, "..", "testing.csv")
    else:
      print("self.parent_dir", self.parent_dir)
      print("self.get_name_of_dataset_train()", self.get_name_of_dataset_train())

      print("coalition_name", self.parent_coalition_name)

      Path(os.path.join(self.parent_dir, "..", "constructed_federated_models_tests")).mkdir(parents=True, exist_ok=True)
      path_client = os.path.join(self.parent_dir, "..", "constructed_federated_models_tests", self.parent_coalition_name + "_testing.csv")
      path_overall = os.path.join(self.parent_dir, "..", "constructed_federated_models_tests", "overall_testing.csv")

    for path in [path_overall, path_client]:

      file_existed_before = Path(path).is_file()

      with open(path, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')

        if(not file_existed_before):
          logwriter.writerow(["Coalition", "Aggregation type", "Client", "Model type", "Subgroup critereon", "Subgroup name", "Testing criteron", "Operating point", "Test scan count", "Average", "Average without no finding", "atelectasis", "cardiomegaly", "consolidation", "edema", "no_finding", "pleural_effusion", "pneumonia", "pneumothorax"])

        for subgroup_test in self.subgroup_test_list:
          subgroup_test.write_test_analysis(logwriter, self.parent_coalition_name, name_of_aggregation, self.get_name_of_dataset_train(), testing_model_type)

        print("Finished writing to", path)
    
  def set_model(self, model):
    self.model = copy.deepcopy(model)

  def load_local_model_from_round_number(self, model_architecture, path_largest_coalition, round_number, local_epoch_number):
    path_of_state_dict = os.path.join(path_largest_coalition, self.get_name_of_dataset_train() + self.name_addition, "federated_training","_"+str(round_number).zfill(3)+"_"+str(local_epoch_number).zfill(3)+".pt")
    print("path_of_state_dict", path_of_state_dict)
    print("self.parent_coalition_name", self.parent_coalition_name)
    print("self.parent_dir", self.parent_dir)
    state_dict = torch.load(path_of_state_dict, map_location=torch.device('cpu'))["model_state_dict"]
    self.model = copy.deepcopy(model_architecture)
    self.model.load_state_dict(state_dict)

  def load_local_model_from_round_number_or_dict(self, model_architecture, path_largest_coalition, round_number, local_epoch_number):
    path_of_state_dict = os.path.join(path_largest_coalition, self.get_name_of_dataset_train() + self.name_addition, "federated_training","_"+str(round_number).zfill(3)+"_"+str(local_epoch_number).zfill(3)+".pt")

    if(path_of_state_dict in globals.model_dict.keys()):
      self.model = globals.model_dict[path_of_state_dict]
    else:
      print("path_of_state_dict", path_of_state_dict)
      print("self.parent_coalition_name", self.parent_coalition_name)
      print("self.parent_dir", self.parent_dir)
      state_dict = torch.load(path_of_state_dict, map_location=torch.device('cpu'))["model_state_dict"]
      self.model = copy.deepcopy(model_architecture)
      self.model.load_state_dict(state_dict)
      globals.model_dict[path_of_state_dict] = self.model

  def compute_gradients_from_previous_global_model_and_local_model(self, global_model):

    gradients = copy.deepcopy(global_model.state_dict())

    for key in global_model.state_dict():
      if(not ".norm" in key):
        gradients[key] = self.model.state_dict()[key] - global_model.state_dict()[key]
      else:
        gradients[key] -= gradients[key]

    return gradients

  def run_local_adaptation(self, model_architecture):
    idx_optimal_global_model = min(range(len(self.epoch_loss_val_list)), key=self.epoch_loss_val_list.__getitem__)
    print("best global model", idx_optimal_global_model)
    path = os.path.join(self.parent_dir, "global_"+str(idx_optimal_global_model).zfill(3)+".pt")

    self.model = copy.deepcopy(model_architecture)
    self.model.load_state_dict(torch.load(path)["model_state_dict"])

    if(not self.keep_local_optimizer):
      if(self.optimizer_name == "Adam"):
        self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.LR_local_adaptation)
      if(self.optimizer_name == "SGD"):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR_local_adaptation, momentum=0)

    self.local_epoch_counter = 0

    path_csv = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "adaptation_of_best_global_model.csv")
    my_file = Path(path_csv)
    if my_file.is_file():
      print("continuing previous local adaptation")
      self.load_state()
      self.prepare_for_next_local_adaptation_round()
      self.local_epoch_counter += 1

    while(not self.ending_condition_for_local_adaptation_reached):
      epoch_loss_train = self.local_train(model=self.model)
      self.epoch_losses_local_adaptation_training.append(epoch_loss_train)

      epoch_loss_val = self.local_validation(model=self.model)

      path_csv = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "adaptation_of_best_global_model.csv")
      with open(path_csv, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        if (self.local_epoch_counter == 0):
          logwriter.writerow(["local_epoch", "LR", "training_loss", "validation_loss"])
        logwriter.writerow([self.local_epoch_counter, self.LR_local_adaptation, epoch_loss_train.item(), epoch_loss_val])

      self.checkpoint_train("local_adaptation", "global_"+str(idx_optimal_global_model).zfill(3)+"_"+str(self.local_epoch_counter).zfill(3)+".pt", self.model)
      self.prepare_for_next_local_adaptation_round()
      self.local_epoch_counter += 1
    

  def prepare_for_next_local_adaptation_round(self):
    if(min(self.epoch_losses_local_adaptation_validation) not in self.epoch_losses_local_adaptation_validation[-3:]):
      if(not self.keep_local_optimizer):
        self.LR_local_adaptation = self.LR_local_adaptation / 2
        if(self.optimizer_name == "Adam"):
          self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.LR_local_adaptation)
        if(self.optimizer_name == "SGD"):
          self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR_local_adaptation, momentum=0)

    if(min(self.epoch_losses_local_adaptation_validation) not in self.epoch_losses_local_adaptation_validation[-6:]):
      self.ending_condition_for_local_adaptation_reached = True
      

  def prepare_for_next_round(self, communication_round_number):
    self.communication_round_number = communication_round_number

    self.local_training_completed = False
    self.local_validation_completed = False

    self.local_epoch_counter = 0

    self.ending_condition_for_federated_learning_reached = (not (min(self.epoch_loss_val_list) in self.epoch_loss_val_list[-6:]))

  def get_local_model_and_dataset_train_size(self):
    return self.model, len(self.dataset_train)

  def set_parent_dir(self, parent_dir):
    self.parent_dir = parent_dir

  def set_parent_coalition_name(self, parent_coalition_name):
    self.parent_coalition_name = parent_coalition_name

  def merge_client(self, client):
    self.dataset_train.merge_with_another_dataset(client.dataset_train)
    self.dataset_val.merge_with_another_dataset(client.dataset_val)
    self.dataset_test.merge_with_another_dataset(client.dataset_test)

  def load_state(self):

    if(self.optimizer_name == "Adam"):
      self.optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.LR)
    if(self.optimizer_name == "SGD"):
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR, momentum=0)
    
    state_dir_federated_training = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "federated_training")
    Path(state_dir_federated_training).mkdir(parents=True, exist_ok=True)

    filenames = next(walk(state_dir_federated_training), (None, None, []))[2]
    filenames.sort()
    if(len(filenames) > 0):
      path_of_latest_state = os.path.join(state_dir_federated_training, filenames[-1])
      loaded_state = torch.load(path_of_latest_state)
      
      self.model.load_state_dict(loaded_state["model_state_dict"])
      self.communication_round_number = loaded_state["communication_round_number"]
      self.local_epoch_counter = loaded_state["local_epoch_counter"]
      self.local_training_completed = loaded_state["local_training_completed"]
      self.epoch_losses_train = loaded_state["epoch_losses_train"]
      self.optimizer.load_state_dict(loaded_state["optimizer_state_dict"])

    
    state_dir_federated_model_validation = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "federated_model_validation")
    Path(state_dir_federated_model_validation).mkdir(parents=True, exist_ok=True)

    filenames = next(walk(state_dir_federated_model_validation), (None, None, []))[2]
    filenames.sort()
    if(len(filenames) > 0):
      path_of_latest_state = os.path.join(state_dir_federated_model_validation, filenames[-1])
      loaded_state = torch.load(path_of_latest_state)

      self.local_validation_completed = loaded_state["local_validation_completed"]
      self.epoch_loss_val_list = loaded_state["epoch_loss_val_list"]
      self.LR = loaded_state["LR"]

      if(filenames[-1] != str(self.communication_round_number).zfill(3)+".pt"):
        self.local_validation_completed = False
    
    state_dir_local_adaptation = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "local_adaptation")
    Path(state_dir_local_adaptation).mkdir(parents=True, exist_ok=True)

    filenames = next(walk(state_dir_local_adaptation), (None, None, []))[2]
    filenames.sort()
    if(len(filenames) > 0):
      path_of_latest_state = os.path.join(state_dir_local_adaptation, filenames[-1])
      loaded_state = torch.load(path_of_latest_state)
      
      self.model.load_state_dict(loaded_state["model_state_dict"])
      self.communication_round_number = loaded_state["communication_round_number"]
      self.local_epoch_counter = loaded_state["local_epoch_counter"]
      self.local_training_completed = loaded_state["local_training_completed"]
      self.epoch_losses_train = loaded_state["epoch_losses_train"]
      self.optimizer.load_state_dict(loaded_state["optimizer_state_dict"])
      self.LR_local_adaptation = loaded_state["LR_local_adaptation"]
      self.epoch_losses_local_adaptation_validation = loaded_state["epoch_losses_local_adaptation_validation"]

    self.model = self.model.to(self.device)

  def copy_files_to_local(self, dataset_name_list):
    dataset_list = []
    with open('config.json') as config_file:
      config = json.load(config_file)

    if("train" in dataset_name_list):
      dataset_list.append(self.dataset_train)
    if("val" in dataset_name_list):
      dataset_list.append(self.dataset_val)
    if("test" in dataset_name_list):
      dataset_list.append(self.dataset_test)
      
    for dataset in dataset_list:
      print("now copying files for", dataset.datasetName)
      i = 0
      file_list = []

      if(config["device_name"] == "bwunicluster"):
        Path(os.path.join("..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data2", dataset.datasetName)).mkdir(parents=True, exist_ok=True)
        dest_path = os.path.join("..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data2", dataset.datasetName)
      if(config["device_name"] == "bwforcluster"):
        Path(os.path.join(os.environ['TMPDIR'], "ml_data2", dataset.datasetName)).mkdir(parents=True, exist_ok=True)
        dest_path = os.path.join(os.environ['TMPDIR'], "ml_data2", dataset.datasetName)
      
      Path(dest_path).mkdir(parents=True, exist_ok=True)

      if(dataset.datasetName == "mimic" or dataset.datasetName=="nih"):
        start = time.time()
        for i, scan in enumerate(dataset.scans):
          if(not os.path.isfile(scan.learning_path)):
            file_list.append(scan.original_scan_path)
          if(i == 0):
            print("original", scan.original_scan_path)
            print("learning", scan.learning_path)
            print("dest path", dest_path)
        if(len(file_list) > 0):
          a = ThreadedCopy(file_list, dest_path)
        end = time.time()
        print("Copied", i, "files in", end - start, "seconds")
        my_file = Path(dataset.scans[0].learning_path)
        print("test, exists?", dataset.scans[0].learning_path, ":", my_file.is_file())
        
      if(dataset.datasetName == "chexpert"):
        start = time.time()
        for scan in dataset.scans:
          i += 1
          if(not os.path.isfile(scan.learning_path)):
            tmp_a = scan.learning_path.rfind("/")
            dest_path = scan.learning_path[0:tmp_a]
            Path(dest_path).mkdir(parents=True, exist_ok=True)
            copyfile(scan.original_scan_path, scan.learning_path)
          if(i%math.floor(len(dataset.scans)/20) == 0):
            print(i, "out of", len(dataset.scans), "chexpert files copied")