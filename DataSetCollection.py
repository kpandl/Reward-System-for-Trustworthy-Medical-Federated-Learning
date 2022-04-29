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

class DataSetCollection:
    
  def __init__(self, seed, list_of_datasets, percentage_train_start, percentage_train_end, percentage_val_start, percentage_val_end, percentage_test_start, percentage_test_end, maximum_translation = 0, mode=0, use_specific_gpu=-1):

      for dataset in list_of_datasets:
          dataset.random_shuffle_patients(seed)

      self.clients = []

      if(mode >= 4 and mode <= 5):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 5):
              dataset_test.scans = dataset_test.scans[0:20]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp.generate_subset_absolute(0, 21000)
            last_patient_added_train_male_25, dataset_train_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_75+1, 7000)
            last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_25+1, 5250)
            _, dataset_val_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_val_male_75+1, 1750)

            last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp.generate_subset_absolute(0, 21000)
            last_patient_added_train_female_25, dataset_train_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_75+1, 7000)
            last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_25+1, 5250)
            _, dataset_val_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_val_female_75+1, 1750)

            dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_25, merge_names=False)
            dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_25, merge_names=False)
            
            dataset_train_female = dataset_train_female_75.merge_with_another_dataset(dataset_train_male_25, merge_names=False)
            dataset_val_female = dataset_val_female_75.merge_with_another_dataset(dataset_val_male_25, merge_names=False)

            if(mode == 5):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_train_female.scans = dataset_train_female.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]
              dataset_val_female.scans = dataset_val_female.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_train_female.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_val_female.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_female.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_female, dataset_val_female, dataset_test, name_addition="_f", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_m", use_specific_gpu=use_specific_gpu))

      if(mode >= 6 and mode <= 7):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 7):
              dataset_test.scans = dataset_test.scans[0:20]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_100, dataset_train_male_100 = dataset_train_male_tmp.generate_subset_absolute(0, 28000)
            _, dataset_val_male_100 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_100+1, 7000)

            last_patient_added_train_female_100, dataset_train_female_100 = dataset_train_female_tmp.generate_subset_absolute(0, 28000)
            _, dataset_val_female_100 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_100+1, 7000)

            dataset_train_male = dataset_train_male_100
            dataset_train_female = dataset_train_female_100
            dataset_val_male = dataset_val_male_100
            dataset_val_female = dataset_val_female_100

            if(mode == 7):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_train_female.scans = dataset_train_female.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]
              dataset_val_female.scans = dataset_val_female.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_train_female.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_val_female.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_female.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_female, dataset_val_female, dataset_test, name_addition="_f", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_m", use_specific_gpu=use_specific_gpu))

      if(mode >= 14 and mode <= 15):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 15):
              dataset_test.scans = dataset_test.scans[0:20]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp.generate_subset_absolute(0, 14000)
            last_patient_added_train_male_25, dataset_train_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_75+1, 14000)
            last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_25+1, 3500)
            _, dataset_val_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_val_male_75+1, 3500)

            last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp.generate_subset_absolute(0, 14000)
            last_patient_added_train_female_25, dataset_train_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_75+1, 14000)
            last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_25+1, 3500)
            _, dataset_val_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_val_female_75+1, 3500)

            dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_25, merge_names=False)
            dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_25, merge_names=False)
            
            dataset_train_female = dataset_train_female_75.merge_with_another_dataset(dataset_train_male_25, merge_names=False)
            dataset_val_female = dataset_val_female_75.merge_with_another_dataset(dataset_val_male_25, merge_names=False)

            if(mode == 15):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_train_female.scans = dataset_train_female.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]
              dataset_val_female.scans = dataset_val_female.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_train_female.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_val_female.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_female.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_female, dataset_val_female, dataset_test, name_addition="_f", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_m", use_specific_gpu=use_specific_gpu))

      if(mode >= 18 and mode <= 19):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 19):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_old_tmp, dataset_train_young_tmp = dataset_train_tmp.split_into_old_patient_and_young_patient_dataset_based_on_quartiles()

            last_patient_added_train_young_larger, dataset_train_young_larger = dataset_train_young_tmp.generate_subset_absolute(0, 20800)
            last_patient_added_val_young_larger, dataset_val_young_larger = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_train_young_larger+1, 5200)

            last_patient_added_train_old_larger, dataset_train_old_larger = dataset_train_old_tmp.generate_subset_absolute(0, 20800)
            last_patient_added_val_old_larger, dataset_val_old_larger = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_train_old_larger+1, 5200)

            dataset_train_young = dataset_train_young_larger
            dataset_val_young = dataset_val_young_larger
            
            dataset_train_old = dataset_train_old_larger
            dataset_val_old = dataset_val_old_larger

            print("length of train datasets", len(dataset_train_young), len(dataset_train_old))
            print("length of val datasets", len(dataset_val_young), len(dataset_val_old))

            if(mode == 19):
              dataset_train_young.scans = dataset_train_young.scans[0:20]
              dataset_train_old.scans = dataset_train_old.scans[0:20]
              dataset_val_young.scans = dataset_val_young.scans[0:20]
              dataset_val_old.scans = dataset_val_old.scans[0:20]

            dataset_train_young.set_label_mode("intersection")
            dataset_train_old.set_label_mode("intersection")
            dataset_val_young.set_label_mode("intersection")
            dataset_val_old.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_young.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_old.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_young, dataset_val_young, dataset_test, name_addition="_young", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_old, dataset_val_old, dataset_test, name_addition="_old", use_specific_gpu=use_specific_gpu))

      if(mode >= 20 and mode <= 21):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              num_scans_female_train = 12182
              num_scans_female_val = 3045
              num_scans_male_train = 15818
              num_scans_male_val = 3955
            if(dataset.datasetName == "chexpert"):
              num_scans_female_train = 11377
              num_scans_female_val = 2844
              num_scans_male_train = 16623
              num_scans_male_val = 4156
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 13271
              num_scans_female_val = 3318
              num_scans_male_train = 14729
              num_scans_male_val = 3682
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 21):
              dataset_test.scans = dataset_test.scans[0:20]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp.generate_subset_absolute(0, num_scans_male_train)
            last_patient_added_train_male_25, dataset_train_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_train)
            last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_25+1, num_scans_male_val)
            _, dataset_val_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_val)

            last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp.generate_subset_absolute(0, num_scans_female_train)
            last_patient_added_train_female_25, dataset_train_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_train)
            last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_25+1, num_scans_female_val)
            _, dataset_val_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_val_female_75+1, num_scans_female_val)

            dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_25, merge_names=False)
            dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_25, merge_names=False)
            
            dataset_train_female = dataset_train_female_75.merge_with_another_dataset(dataset_train_male_25, merge_names=False)
            dataset_val_female = dataset_val_female_75.merge_with_another_dataset(dataset_val_male_25, merge_names=False)

            if(mode == 21):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_train_female.scans = dataset_train_female.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]
              dataset_val_female.scans = dataset_val_female.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_train_female.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_val_female.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_female.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_female, dataset_val_female, dataset_test, name_addition="_f", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_m", use_specific_gpu=use_specific_gpu))

      if(mode >= 22 and mode <= 23):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 23):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_old_tmp, dataset_train_young_tmp = dataset_train_tmp.split_into_old_patient_and_young_patient_dataset_based_on_quartiles()

            last_patient_added_train_young_larger, dataset_train_young_larger = dataset_train_young_tmp.generate_subset_absolute(0, 15600)
            last_patient_added_val_young_larger, dataset_val_young_larger = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_train_young_larger+1, 3900)
            last_patient_added_train_young_smaller, dataset_train_young_smaller = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_val_young_larger+1, 5200)
            _, dataset_val_young_smaller = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_train_young_smaller+1, 1300)

            last_patient_added_train_old_larger, dataset_train_old_larger = dataset_train_old_tmp.generate_subset_absolute(0, 15600)
            last_patient_added_val_old_larger, dataset_val_old_larger = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_train_old_larger+1, 3900)
            last_patient_added_train_old_smaller, dataset_train_old_smaller = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_val_old_larger+1, 5200)
            _, dataset_val_old_smaller = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_train_old_smaller+1, 1300)


            dataset_train_young = dataset_train_young_larger.merge_with_another_dataset(dataset_train_old_smaller, merge_names=False)
            dataset_val_young = dataset_val_young_larger.merge_with_another_dataset(dataset_val_old_smaller, merge_names=False)
            
            dataset_train_old = dataset_train_old_larger.merge_with_another_dataset(dataset_train_young_smaller, merge_names=False)
            dataset_val_old = dataset_val_old_larger.merge_with_another_dataset(dataset_val_young_smaller, merge_names=False)

            print("length of train datasets", len(dataset_train_young), len(dataset_train_old))
            print("length of val datasets", len(dataset_val_young), len(dataset_val_old))

            if(mode == 23):
              dataset_train_young.scans = dataset_train_young.scans[0:20]
              dataset_train_old.scans = dataset_train_old.scans[0:20]
              dataset_val_young.scans = dataset_val_young.scans[0:20]
              dataset_val_old.scans = dataset_val_old.scans[0:20]

            dataset_train_young.set_label_mode("intersection")
            dataset_train_old.set_label_mode("intersection")
            dataset_val_young.set_label_mode("intersection")
            dataset_val_old.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_young.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_old.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_young, dataset_val_young, dataset_test, name_addition="_young", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_old, dataset_val_old, dataset_test, name_addition="_old", use_specific_gpu=use_specific_gpu))
            
      if(mode >= 24 and mode <= 25):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 25):
              dataset_test.scans = dataset_test.scans[0:20]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp.generate_subset_absolute(0, 7000)
            last_patient_added_train_male_25, dataset_train_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_75+1, 7000)
            last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_25+1, 1600)
            _, dataset_val_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_val_male_75+1, 1600)

            last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp.generate_subset_absolute(0, 7000)
            last_patient_added_train_female_25, dataset_train_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_75+1, 7000)
            last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_25+1, 1600)
            _, dataset_val_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_val_female_75+1, 1600)

            dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_25, merge_names=False)
            dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_25, merge_names=False)
            
            dataset_train_female = dataset_train_female_75.merge_with_another_dataset(dataset_train_male_25, merge_names=False)
            dataset_val_female = dataset_val_female_75.merge_with_another_dataset(dataset_val_male_25, merge_names=False)

            if(mode == 25):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_train_female.scans = dataset_train_female.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]
              dataset_val_female.scans = dataset_val_female.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_train_female.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_val_female.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_female.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_female.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_female, dataset_val_female, dataset_test, name_addition="_f", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_m", use_specific_gpu=use_specific_gpu))

      if(mode >= 26 and mode <= 27):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 27):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_old_tmp, dataset_train_young_tmp = dataset_train_tmp.split_into_old_patient_and_young_patient_dataset_based_on_quartiles()

            last_patient_added_train_young_larger, dataset_train_young_larger = dataset_train_young_tmp.generate_subset_absolute(0, 10400)
            last_patient_added_val_young_larger, dataset_val_young_larger = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_train_young_larger+1, 2600)
            last_patient_added_train_young_smaller, dataset_train_young_smaller = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_val_young_larger+1, 10400)
            _, dataset_val_young_smaller = dataset_train_young_tmp.generate_subset_absolute(last_patient_added_train_young_smaller+1, 2600)

            last_patient_added_train_old_larger, dataset_train_old_larger = dataset_train_old_tmp.generate_subset_absolute(0, 10400)
            last_patient_added_val_old_larger, dataset_val_old_larger = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_train_old_larger+1, 5200)
            last_patient_added_train_old_smaller, dataset_train_old_smaller = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_val_old_larger+1, 10400)
            _, dataset_val_old_smaller = dataset_train_old_tmp.generate_subset_absolute(last_patient_added_train_old_smaller+1, 5200)


            dataset_train_young = dataset_train_young_larger.merge_with_another_dataset(dataset_train_old_smaller, merge_names=False)
            dataset_val_young = dataset_val_young_larger.merge_with_another_dataset(dataset_val_old_smaller, merge_names=False)
            
            dataset_train_old = dataset_train_old_larger.merge_with_another_dataset(dataset_train_young_smaller, merge_names=False)
            dataset_val_old = dataset_val_old_larger.merge_with_another_dataset(dataset_val_young_smaller, merge_names=False)

            print("length of train datasets", len(dataset_train_young), len(dataset_train_old))
            print("length of val datasets", len(dataset_val_young), len(dataset_val_old))

            if(mode == 27):
              dataset_train_young.scans = dataset_train_young.scans[0:20]
              dataset_train_old.scans = dataset_train_old.scans[0:20]
              dataset_val_young.scans = dataset_val_young.scans[0:20]
              dataset_val_old.scans = dataset_val_old.scans[0:20]

            dataset_train_young.set_label_mode("intersection")
            dataset_train_old.set_label_mode("intersection")
            dataset_val_young.set_label_mode("intersection")
            dataset_val_old.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_young.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_old.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_young, dataset_val_young, dataset_test, name_addition="_young", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_old, dataset_val_old, dataset_test, name_addition="_old", use_specific_gpu=use_specific_gpu))

      if(mode >= 28 and mode <= 29):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 29):
              dataset_test.scans = dataset_test.scans[0:20]

            last_patient_added_train_young_larger, dataset_train_young_larger = dataset_train_tmp.generate_subset_absolute(0, 20800)
            last_patient_added_val_young_larger, dataset_val_young_larger = dataset_train_tmp.generate_subset_absolute(last_patient_added_train_young_larger+1, 5200)

            last_patient_added_train_old_larger, dataset_train_old_larger = dataset_train_tmp.generate_subset_absolute(last_patient_added_val_young_larger+1, 20800)
            last_patient_added_val_old_larger, dataset_val_old_larger = dataset_train_tmp.generate_subset_absolute(last_patient_added_train_old_larger+1, 5200)

            dataset_train_young = dataset_train_young_larger
            dataset_val_young = dataset_val_young_larger
            
            dataset_train_old = dataset_train_old_larger
            dataset_val_old = dataset_val_old_larger

            print("length of train datasets", len(dataset_train_young), len(dataset_train_old))
            print("length of val datasets", len(dataset_val_young), len(dataset_val_old))

            if(mode == 29):
              dataset_train_young.scans = dataset_train_young.scans[0:20]
              dataset_train_old.scans = dataset_train_old.scans[0:20]
              dataset_val_young.scans = dataset_val_young.scans[0:20]
              dataset_val_old.scans = dataset_val_old.scans[0:20]

            dataset_train_young.set_label_mode("intersection")
            dataset_train_old.set_label_mode("intersection")
            dataset_val_young.set_label_mode("intersection")
            dataset_val_old.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
            else:
              dataset_train_young.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_train_old.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            dataset_val_young.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_val_old.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            self.clients.append(Client(dataset_train_young, dataset_val_young, dataset_test, name_addition="_young", use_specific_gpu=use_specific_gpu))
            self.clients.append(Client(dataset_train_old, dataset_val_old, dataset_test, name_addition="_old", use_specific_gpu=use_specific_gpu))

      self.random_seed = seed
      np.random.seed(self.random_seed)

  def copy_files_to_local(self, dataset_name_list):
    for client in self.clients:
        client.copy_files_to_local(dataset_name_list)

