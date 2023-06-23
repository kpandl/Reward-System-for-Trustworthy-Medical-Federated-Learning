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
import torchvision.transforms as transforms
from densenet import DenseNet121
from torchvision import  models
from sklearn import datasets, linear_model
from sklearn import metrics
import pickle

class Client:

  def get_features(self, name):
    def hook(model, input, output):
      self.features[name] = output.detach()
    return hook
    
  def __init__(self, dataset_train, dataset_val, dataset_test, name_addition="", use_specific_gpu=-1):
    self.dataset_train = dataset_train
    self.dataset_val = dataset_val
    self.dataset_test = dataset_test
    self.name_addition = name_addition
    self.use_specific_gpu=use_specific_gpu
    self.features = {}

    
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
      

  def run_testing(self, name_of_aggregation="", load_model=True, ending_condition_mode="local_clients", folderpath_testing_general=None, fle_name=None):
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
      
      if(self.use_specific_gpu==-1):
        loaded_state = torch.load(path, map_location='cuda:0')
      else:
        loaded_state = torch.load(path, map_location='cuda:'+str(self.use_specific_gpu))
      print("loading model for test:",path)

      self.model.load_state_dict(loaded_state["model_state_dict"])

    self.local_test(self.model, testing_model_type=testing_model_type, name_of_aggregation=name_of_aggregation, load_model=load_model, folderpath_testing_general=folderpath_testing_general, fle_name=fle_name)

  def local_test(self, model, testing_model_type="", name_of_aggregation="", load_model=True, folderpath_testing_general=None, fle_name=None):
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

      if(folderpath_testing_general==None):
        Path(os.path.join(self.parent_dir, "..", "constructed_federated_models_tests")).mkdir(parents=True, exist_ok=True)
        path_client = os.path.join(self.parent_dir, "..", "constructed_federated_models_tests", self.parent_coalition_name + "_testing.csv")
        path_overall = os.path.join(self.parent_dir, "..", "constructed_federated_models_tests", "overall_testing.csv")
      else:
        path_client = None
        path_overall = os.path.join(folderpath_testing_general, "overall_testing.csv")

    for path in [path_overall, path_client]:
      if(path==None):
        continue
      file_existed_before = Path(path).is_file()

      with open(path, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')

        if(not file_existed_before):
          logwriter.writerow(["Coalition", "Aggregation type", "Client", "Model type", "Subgroup critereon", "Subgroup name", "Testing criteron", "Operating point", "Test scan count", "Average", "Average without no finding", "atelectasis", "cardiomegaly", "consolidation", "edema", "no_finding", "pleural_effusion", "pneumonia", "pneumothorax"])

        for subgroup_test in self.subgroup_test_list:
          if(fle_name==None):
            subgroup_test.write_test_analysis(logwriter, self.parent_coalition_name, name_of_aggregation, self.get_name_of_dataset_train(), testing_model_type)
          else:
             subgroup_test.write_test_analysis(logwriter, fle_name, name_of_aggregation, self.get_name_of_dataset_train(), testing_model_type)


        print("Finished writing to", path)


  def extract_deep_features(self, model, dataset, dataset_name):
    print("extracting deep features", dataset_name)
    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = model.to(self.device)

    PREDS = []
    #FEATS = []
    #features = {}
    

    model.eval()

    #model.denseblock4.register_forward_hook(self.get_features('denseblock4'))
    model.features.denseblock4.register_forward_hook(self.get_features('denseblock4'))


    #out_pred = torch.FloatTensor().to("cpu")  # tensor stores prediction values
    out_gt = torch.FloatTensor().to("cpu")  # tensor stores groundtruth values
    out_features = torch.FloatTensor().to("cpu")  # tensor stores prediction values

    #batch_counter = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(self.device), target.to(self.device)
            #output = model(data, "deepfeatures")
            preds = model(data)

            #PREDS.append(preds.detach().cpu().numpy())
            #FEATS.append(self.features['denseblock4'].cpu().numpy())

            out_gt = torch.cat((out_gt, target.to("cpu")), 0)
            #out_pred = torch.cat((out_pred, output.to("cpu")), 0)
            out_features = torch.cat((out_features, self.features['denseblock4'].cpu()), 0)

            #batch_counter += 1

    genders = []
    ages = []
    for i in range(len(dataset)):
      genders.append(dataset.scans[i].corresponding_patient.gender)
      ages.append(dataset.scans[i].patient_age)

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_features_dataset_"+dataset_name+".npz")
    np.savez_compressed(path, out_features)
    print("saved at", path)

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "labels_dataset_"+dataset_name+".npz")
    np.savez_compressed(path, out_gt)
    print("saved at", path)
    
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "genders_dataset_"+dataset_name+".npz")
    np.savez_compressed(path, np.array(genders))
    print("saved at", path)
    
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "ages_dataset_"+dataset_name+".npz")
    np.savez_compressed(path, np.array(ages))
    print("saved at", path)






    
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
    
  def load_local_model_path(self, path_of_state_dict, model_architecture):
    print("path_of_state_dict", path_of_state_dict)
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

  def create_deep_features(self):

    df_global = pd.read_csv(os.path.join(self.parent_dir, "federated_training.csv"))
    best_epoch_global = df_global["communication round"][df_global["average val loss"].idxmin()]

    path = os.path.join(self.parent_dir, "global_"+str(best_epoch_global).zfill(3)+".pt")


    #path = os.path.join(self.parent_dir, "global_" + str(round_number).zfill(3) + ".pt")
    #model = DenseNet121()
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(num_ftrs, 8), nn.Sigmoid())

    loaded_model_state_dict = torch.load(path, map_location=torch.device('cpu'))["model_state_dict"]
    model.load_state_dict(loaded_model_state_dict)

    #list_trained = list(loaded_model_state_dict.keys())
    #list_new = list(model.state_dict().keys())

    self.dataset_train.set_transform(transform=transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    self.extract_deep_features(model, self.dataset_train, "train")
    
    self.dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    self.extract_deep_features(model, self.dataset_test, "test")

    a = 0

  def compute_knn_shapley_values(self, index_condition=0, filter="none"):
    
    X_train_deep, X_test_deep_balanced, y_train_deep, y_test_deep_balanced = self.get_loaded_deep_features(index_condition, filter)

    k=3

    deep_knn_values, scores_mean, *_ = self.original_knn_shapley(k, X_train_deep, X_test_deep_balanced[0:600], y_train_deep, y_test_deep_balanced[0:600], index_condition)
    
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_knn_values_balanced_label_"+str(index_condition)+".npz")
    np.savez_compressed(path, knn=deep_knn_values)
    print("saved at", path)

    print("sum", sum(deep_knn_values))
    print("scores_mean", scores_mean)
    print("finished obtaining balanced knn shapley values")
    
    compute_unbalanced = False

    if(compute_unbalanced):
      deep_knn_values, scores_mean, *_ = self.original_knn_shapley(k, X_train_deep, X_test_deep[0:600], y_train_deep, y_test_deep[0:600], index_condition)
      
      path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_knn_values_unbalanced"+".npz")
      np.savez_compressed(path, knn=deep_knn_values)

      print("sum", sum(deep_knn_values))
      print("scores_mean", scores_mean)
      print("finished obtaining unbalanced knn shapley values")

    a = 0

  def delete_deep_feature_arrays(self):
    path_train_deep = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_features_dataset_train.npz")
    print("deleting", path_train_deep)
    os.remove(path_train_deep)

    path_test_deep = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_features_dataset_test.npz")
    print("deleting", path_test_deep)
    os.remove(path_test_deep)

  def compute_knn_shapley_values_based_on_loaded_deep_features(self, X_train_deep, X_test_deep_balanced, y_train_deep, y_test_deep_balanced, index_condition=0, filter="none", limit_collected_dataset=-1):
    
    k=3

    deep_knn_values, scores_mean, *_ = self.original_knn_shapley(k, X_train_deep, X_test_deep_balanced, y_train_deep, y_test_deep_balanced, index_condition)
    
    filter_addition = ""
    if(filter != "none"):
      filter_addition = filter + "_"

    name_addition = ""
    if(limit_collected_dataset != -1):
      name_addition = "_limit_" + str(limit_collected_dataset)

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_knn_values_balanced_federated_" + filter_addition + "label_"+str(index_condition)+name_addition+".npz")
    np.savez_compressed(path, knn=deep_knn_values)
    print("saved at", path)

    print("sum", sum(deep_knn_values))
    print("scores_mean", scores_mean)
    print("finished obtaining balanced knn shapley values")

    a = 0

  def get_loaded_deep_features(self, index_condition=0, filter="none", load_train=True):
    
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_features_dataset_"+"train"+".npz")
    if(load_train):
      with open(path, "rb") as f:
        X_train_deep = np.load(f)["arr_0"]#[:40000]
    else:
      X_train_deep = np.zeros((1,1))


    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "labels_dataset_"+"train"+".npz")
    if(load_train):
      with open(path, "rb") as f:
        y_train_deep = np.load(f)["arr_0"]#[:40000]
    else:
      y_train_deep = np.zeros((1,1))
      
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "deep_features_dataset_"+"test"+".npz")
    with open(path, "rb") as f:
      X_test_deep = np.load(f)["arr_0"]#[:40000]

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "labels_dataset_"+"test"+".npz")
    with open(path, "rb") as f:
      y_test_deep = np.load(f)["arr_0"]#[:40000]

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "genders_dataset_"+"test"+".npz")
    with open(path, "rb") as f:
      genders_test = np.load(f, allow_pickle=True)["arr_0"]

    if(filter=="none"):
      
      X_test_deep_0=[]
      X_test_deep_1=[]
      y_test_deep_0=[]
      y_test_deep_1=[]

      for i in range(len(y_test_deep)):
        label=y_test_deep[i][index_condition]
        if(label==0.0):
          X_test_deep_0.append(X_test_deep[i])
          y_test_deep_0.append(y_test_deep[i])
        if(label==1.0):
          X_test_deep_1.append(X_test_deep[i])
          y_test_deep_1.append(y_test_deep[i])
      
      X_test_deep_balanced = np.array(X_test_deep_0[0:300] + X_test_deep_1[0:300])
      y_test_deep_balanced = np.array(y_test_deep_0[0:300] + y_test_deep_1[0:300])
    
    if(filter=="male" or filter =="female"):
      
      X_test_deep_0_male=[]
      X_test_deep_0_female=[]
      X_test_deep_1_male=[]
      X_test_deep_1_female=[]

      y_test_deep_0_male=[]
      y_test_deep_0_female=[]
      y_test_deep_1_male=[]
      y_test_deep_1_female=[]

      for i in range(len(y_test_deep)):
        label=y_test_deep[i][index_condition]
        if(label==0.0 and genders_test[i]=="M"):
          X_test_deep_0_male.append(X_test_deep[i])
          y_test_deep_0_male.append(y_test_deep[i])
        if(label==0.0 and genders_test[i]=="F"):
          X_test_deep_0_female.append(X_test_deep[i])
          y_test_deep_0_female.append(y_test_deep[i])
        if(label==1.0 and genders_test[i]=="M"):
          X_test_deep_1_male.append(X_test_deep[i])
          y_test_deep_1_male.append(y_test_deep[i])
        if(label==1.0 and genders_test[i]=="F"):
          X_test_deep_1_female.append(X_test_deep[i])
          y_test_deep_1_female.append(y_test_deep[i])
      
        if(filter == "male"):
          X_test_deep_balanced = np.array(X_test_deep_0_male[0:150] + X_test_deep_1_male[0:150])
          y_test_deep_balanced = np.array(y_test_deep_0_male[0:150] + y_test_deep_1_male[0:150])
        if(filter == "female"):
          X_test_deep_balanced = np.array(X_test_deep_0_female[0:150] + X_test_deep_1_female[0:150])
          y_test_deep_balanced = np.array(y_test_deep_0_female[0:150] + y_test_deep_1_female[0:150])
    
    if(filter=="all"):
      X_test_deep_balanced = X_test_deep
      y_test_deep_balanced = y_test_deep

    return X_train_deep, X_test_deep_balanced, y_train_deep, y_test_deep_balanced

  def original_knn_shapley(self, K, trainX, valX, trainy, valy, index_condition=0):
      #valy_of_position_is_1 = []
      score_original = []
      """calculates KNN-Shapley values

                  # Arguments
                          K: number of nearest neighbors
                          trainX: deep features of train images
                          valX: deep features of test images
                          trainy: deep features of train labels
                          valy: deep features of test labels

                  # Returns
                          KNN values"""

      N = trainX.shape[0]
      M = valX.shape[0]
      c = 1
      value = np.zeros(N)
      #     value = [[] for i in range(N) ]
      scores = []
      false_result_idxs = []
      for i in range(M):
          #if(valy[i][index_condition] == 1):
          #    valy_of_position_is_1.append(i)
          print("Step", i, "from", M)
          X = valX[i]
          y = valy[i]

          s = np.zeros(N)
          diff = (trainX - X).reshape(N, -1)  # calculate the distances between valX and every trainX data point

          dist = np.einsum('ij, ij->i', diff, diff)  # output the sum distance
          idx = np.argsort(dist)  # ascend the distance
          ans = trainy[idx]

          # calculate test performance
          score = 0.0

          for j in range(min(K, N)):
              score += float(ans[j][index_condition] == y[index_condition])
          score_original.append(score/K)
          if (score > min(K, N) / 2):
              scores.append(1)
          else:
              scores.append(0)
              false_result_idxs.append(i)

          s[idx[N - 1]] = float(ans[N - 1][index_condition] == y[index_condition]) * c / N
          cur = N - 2
          for j in range(N - 1):
              s[idx[cur]] = s[idx[cur + 1]] + float(int(ans[cur][index_condition] == y[index_condition]) - int(ans[cur + 1][index_condition] == y[index_condition])) * c / K * (
                          min(cur + 1, K) / (cur + 1))
              cur -= 1

          for j in range(N):
              value[j] += s[j]

      for i in range(N):
          value[i] /= M

      return value, np.mean(score_original), false_result_idxs

      

  def compute_LR_model(self, dict_X_test_deep_balanced, dict_y_test_deep_balanced, dict_test_dataset_sizes):
    
    #i = 0
    #X_test_deep_balanced = None
    #y_test_deep_balanced = None

    #for key in dict_X_test_deep_balanced.keys():
    #  if(i==0):
    #    X_test_deep_balanced = dict_X_test_deep_balanced[key]
    #    y_test_deep_balanced = dict_y_test_deep_balanced[key]
    #  else:
    #    X_test_deep_balanced = np.concatenate((X_test_deep_balanced, dict_X_test_deep_balanced[key]), axis=0)
    #    y_test_deep_balanced = np.concatenate((y_test_deep_balanced, dict_y_test_deep_balanced[key]), axis=0)
    #  i+=1
    print("loading data")

    X_train_deep, _, y_train_deep, _ = self.get_loaded_deep_features(0, "all")
    regr = linear_model.LinearRegression()

    print("reshaping data 1")
    res = X_train_deep
    res2 = np.reshape(res, (len(X_train_deep), 65536))
    
    print("fitting model")
    regr.fit(res2, y_train_deep)
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "LR_model"+".pkl")
    with open(path, 'wb') as file:
      pickle.dump(regr, file)

    print("predicting")
    train_predictions = regr.predict(res2)
    print("calculating train metrics")
    st = SubgroupTest(None, None, None)
    aucs_train_prediction = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(train_predictions))), torch.from_numpy(y_train_deep), torch.from_numpy(train_predictions))
    print("AUCs train prediction", aucs_train_prediction)
    print("Average train", np.mean(aucs_train_prediction))
    print("Num of train predictions", len(train_predictions))

    dict_test_predictions = {}
    
    for key in dict_X_test_deep_balanced.keys():
      print("reshaping data 2, key", key)
      res = dict_X_test_deep_balanced[key]
      res2 = np.reshape(res, (len(res), 65536))
      print("predicting")
      test_predictions = regr.predict(res2)
      aucs_test_prediction = st.compute_roc_auc_for_subgroup_and_conditions(list(range(len(test_predictions))), torch.from_numpy(dict_y_test_deep_balanced[key]), torch.from_numpy(test_predictions))
      dict_test_predictions[key] = test_predictions
      print("AUCs test prediction", aucs_test_prediction)
      print("Average test", np.mean(aucs_test_prediction))
      print("Num of test points", len(test_predictions))

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "LR_model_test_results"+".pkl")
    #np.savez_compressed(path, test_predictions)
    with open(path, 'wb') as file:
      pickle.dump(dict_test_predictions, file)
    
    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "LR_model_test_results_corresponding_labels"+".pkl")
    #np.savez_compressed(path, y_test_deep_balanced)
    with open(path, 'wb') as file:
      pickle.dump(dict_y_test_deep_balanced, file)

    path = os.path.join(self.parent_dir, self.get_name_of_dataset_train() + self.name_addition, "client_test_dataset_sizes"+".pkl")
    with open(path, 'wb') as file:
      pickle.dump(dict_test_dataset_sizes, file)

  def flip_labels(self, probability):
    return self.dataset_train.flip_labels(probability)