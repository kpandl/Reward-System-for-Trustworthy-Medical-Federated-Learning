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


      if(mode >= 8 and mode <= 9):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 9):
              dataset_test.scans = dataset_test.scans[0:1000]
            
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
            
            if(mode == 9):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))


            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

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


      if(mode >= 16 and mode <= 17):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 17):
              dataset_test.scans = dataset_test.scans[0:1000]
            
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
            
            if(mode == 17):
              dataset_train_male.scans = dataset_train_male.scans[0:20]
              dataset_val_male.scans = dataset_val_male.scans[0:20]

            dataset_train_male.set_label_mode("intersection")
            dataset_val_male.set_label_mode("intersection")
            dataset_test.set_label_mode("intersection")

            if(maximum_translation==0):
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

            else:
              dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))


            dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

            dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

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



      if(mode >= 30 and mode <= 31):
        for dataset in list_of_datasets:
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 31):
              dataset_test.scans = dataset_test.scans[0:50]
            
            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()

            last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp.generate_subset_absolute(0, 100)
            last_patient_added_train_male_25, dataset_train_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_75+1, 100)
            last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_train_male_25+1, 30)
            _, dataset_val_male_25 = dataset_train_male_tmp.generate_subset_absolute(last_patient_added_val_male_75+1, 30)

            last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp.generate_subset_absolute(0, 100)
            last_patient_added_train_female_25, dataset_train_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_75+1, 100)
            last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_train_female_25+1, 30)
            _, dataset_val_female_25 = dataset_train_female_tmp.generate_subset_absolute(last_patient_added_val_female_75+1, 30)

            dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_25, merge_names=False)
            dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_25, merge_names=False)
            
            dataset_train_female = dataset_train_female_75.merge_with_another_dataset(dataset_train_male_25, merge_names=False)
            dataset_val_female = dataset_val_female_75.merge_with_another_dataset(dataset_val_male_25, merge_names=False)

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



      if(mode >= 40 and mode <= 41):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 13271
              num_scans_female_val = 3318
              num_scans_male_train = 14729
              num_scans_male_val = 3682
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 41):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(6):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 41):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))



      if(mode >= 42 and mode <= 43):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 5406
              num_scans_female_val = 1352
              num_scans_male_train = 6000#14729
              num_scans_male_val = 1500#3682
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 43):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(18):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 43):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))





      if(mode >= 44 and mode <= 45):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 13271
              num_scans_female_val = 3318
              num_scans_male_train = 14729
              num_scans_male_val = 3682
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            if(mode == 45):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(3):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 45):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))






      if(mode >= 46 and mode <= 47):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              dataset_train_tmp_nih = dataset.generate_subset(0, 0.8)
              dataset_test_nih = dataset.generate_subset(0.8, 1)
            if(dataset.datasetName == "chexpert"):
              dataset_train_tmp_chexpert = dataset.generate_subset(0, 0.8)
              dataset_test_chexpert = dataset.generate_subset(0.8, 1)
            if(dataset.datasetName == "mimic"):
              dataset_train_tmp_mimic = dataset.generate_subset(0, 0.8)
              dataset_test_mimic = dataset.generate_subset(0.8, 1)
              


        for i in range(3):
          dataset_train_tmp_nih_client = dataset_train_tmp_nih.generate_subset(i/6, (i+1)/6)
          dataset_train_tmp_chexpert_client = dataset_train_tmp_chexpert.generate_subset(i/6, (i+1)/6)
          dataset_train_tmp_mimic_client = dataset_train_tmp_mimic.generate_subset(i/6, (i+1)/6)

          dataset_test_nih_client = dataset_test_nih.generate_subset(i/6, (i+1)/6)
          dataset_test_chexpert_client = dataset_test_chexpert.generate_subset(i/6, (i+1)/6)
          dataset_test_mimic_client = dataset_test_mimic.generate_subset(i/6, (i+1)/6)

          dataset_val_tmp_nih_client = copy.deepcopy(dataset_train_tmp_nih_client)
          dataset_val_tmp_chexpert_client = copy.deepcopy(dataset_train_tmp_chexpert_client)
          dataset_val_tmp_mimic_client = copy.deepcopy(dataset_train_tmp_mimic_client)

          dataset_val_tmp_nih_client.scans = dataset_val_tmp_nih_client.scans[9333:11666]
          dataset_val_tmp_chexpert_client.scans = dataset_val_tmp_chexpert_client.scans[9333:11666]
          dataset_val_tmp_mimic_client.scans = dataset_val_tmp_mimic_client.scans[9333:11666]

          dataset_val_tmp_nih_client.update_patient_list_based_on_scans()
          dataset_val_tmp_chexpert_client.update_patient_list_based_on_scans()
          dataset_val_tmp_mimic_client.update_patient_list_based_on_scans()

          dataset_train_tmp_nih_client.scans = dataset_train_tmp_nih_client.scans[0:9333]
          dataset_train_tmp_chexpert_client.scans = dataset_train_tmp_chexpert_client.scans[0:9333]
          dataset_train_tmp_mimic_client.scans = dataset_train_tmp_mimic_client.scans[0:9333]

          dataset_train_tmp_nih_client.update_patient_list_based_on_scans()
          dataset_train_tmp_chexpert_client.update_patient_list_based_on_scans()
          dataset_train_tmp_mimic_client.update_patient_list_based_on_scans()

          dataset_test_nih_client.scans = dataset_test_nih_client.scans[0:2333]
          dataset_test_chexpert_client.scans = dataset_test_chexpert_client.scans[0:2333]
          dataset_test_mimic_client.scans = dataset_test_mimic_client.scans[0:2333]

          dataset_test_nih_client.update_patient_list_based_on_scans()
          dataset_test_chexpert_client.update_patient_list_based_on_scans()
          dataset_test_mimic_client.update_patient_list_based_on_scans()

          dataset_val_tmp_nih_client.merge_with_another_dataset(dataset_val_tmp_chexpert_client)
          dataset_val_tmp_nih_client.merge_with_another_dataset(dataset_val_tmp_mimic_client)

          dataset_train_tmp_nih_client.merge_with_another_dataset(dataset_train_tmp_chexpert_client)
          dataset_train_tmp_nih_client.merge_with_another_dataset(dataset_train_tmp_mimic_client)

          dataset_test_nih_client.merge_with_another_dataset(dataset_test_chexpert_client)
          dataset_test_nih_client.merge_with_another_dataset(dataset_test_mimic_client)

          dataset_val_tmp_nih_client.set_label_mode("intersection")
          dataset_train_tmp_nih_client.set_label_mode("intersection")
          dataset_test_nih_client.set_label_mode("intersection")

          if(maximum_translation==0):
            dataset_train_tmp_nih_client.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
          else:
            dataset_train_tmp_nih_client.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
            ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

          dataset_val_tmp_nih_client.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
          ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ]))
          dataset_test_nih_client.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
          ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ]))          
        
          if(mode == 47):
            dataset_train_tmp_nih_client.scans = dataset_train_tmp_nih_client.scans[0:20]
            dataset_val_tmp_nih_client.scans = dataset_val_tmp_nih_client.scans[0:20]
            dataset_test_nih_client.scans = dataset_test_nih_client.scans[0:20]

          self.clients.append(Client(dataset_train_tmp_nih_client, dataset_val_tmp_nih_client, dataset_test_nih_client, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))




      if(mode >= 48 and mode <= 49):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 49):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(2):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 49):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))





      if(mode >= 50 and mode <= 51):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 51):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(4):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 51):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))






      if(mode >= 52 and mode <= 53):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 53):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(6):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 53):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))





      if(mode >= 54 and mode <= 55):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 55):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(8):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 55):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))





      if(mode >= 56 and mode <= 57):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 57):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(10):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 57):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))






      if(mode >= 58 and mode <= 59):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 59):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(12):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 59):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))







      if(mode >= 60 and mode <= 61):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 61):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(14):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 61):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))






      if(mode >= 62 and mode <= 63):
        for dataset in list_of_datasets:
            if(dataset.datasetName == "nih"):
              continue
            if(dataset.datasetName == "chexpert"):
              continue
            if(dataset.datasetName == "mimic"):
              num_scans_female_train = 4740
              num_scans_female_val = 1185
              num_scans_male_train = 5260
              num_scans_male_val = 1315
            dataset_train_tmp = dataset.generate_subset(0, 0.8)
            dataset_test = dataset.generate_subset(0.8, 1)

            dataset_test.scans = dataset_test.scans[0:3600]

            if(mode == 63):
              dataset_test.scans = dataset_test.scans[0:20]

            dataset_train_male_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_male_tmp.scans = [scan for scan in dataset_train_male_tmp.scans if scan.corresponding_patient.gender=="M"]
            dataset_train_male_tmp.update_patient_list_based_on_scans()
            # 146750

            dataset_train_female_tmp = copy.deepcopy(dataset_train_tmp)
            dataset_train_female_tmp.scans = [scan for scan in dataset_train_female_tmp.scans if scan.corresponding_patient.gender=="F"]
            dataset_train_female_tmp.update_patient_list_based_on_scans()
            # 131559

            last_patient_added_val_male_75 = 0
            last_patient_added_val_female_75 = 0
            for i in range(16):
              dataset_train_male_tmp2 = copy.deepcopy(dataset_train_male_tmp)
              dataset_train_female_tmp2 = copy.deepcopy(dataset_train_female_tmp)
              
              last_patient_added_train_male_75, dataset_train_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_val_male_75+1, num_scans_male_train)
              last_patient_added_val_male_75, dataset_val_male_75 = dataset_train_male_tmp2.generate_subset_absolute(last_patient_added_train_male_75+1, num_scans_male_val)

              last_patient_added_train_female_75, dataset_train_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_val_female_75 + 1, num_scans_female_train)
              last_patient_added_val_female_75, dataset_val_female_75 = dataset_train_female_tmp2.generate_subset_absolute(last_patient_added_train_female_75+1, num_scans_female_val)

              dataset_train_male = dataset_train_male_75.merge_with_another_dataset(dataset_train_female_75, merge_names=False)
              dataset_val_male = dataset_val_male_75.merge_with_another_dataset(dataset_val_female_75, merge_names=False)
              
              if(mode == 63):
                dataset_train_male.scans = dataset_train_male.scans[0:20]
                dataset_val_male.scans = dataset_val_male.scans[0:20]

              dataset_train_male.set_label_mode("intersection")
              dataset_val_male.set_label_mode("intersection")
              dataset_test.set_label_mode("intersection")

              if(maximum_translation==0):
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
              else:
                dataset_train_male.set_transform(transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(maximum_translation, maximum_translation)), transforms.RandomRotation(15), transforms.Resize((256,256)), transforms.CenterCrop(256), transforms.ToTensor()
                ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

              dataset_val_male.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))
              dataset_test.set_transform(transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()
              ,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ]))

              self.clients.append(Client(dataset_train_male, dataset_val_male, dataset_test, name_addition="_"+str(i), use_specific_gpu=use_specific_gpu))






      self.random_seed = seed
      np.random.seed(self.random_seed)

  def flip_labels(self, client_indices, probability, store_flipped_label_indices=False, result_folder_name=None):
    old_state = random.getstate()
    random.seed(self.random_seed)
    flipped_label_indices = {}
    i = 0
    accumulated_length_client = 0
    if(type(probability)!=list):
      probability = [probability]*len(client_indices)
    
    print("len(self.clients)", len(self.clients))

    for count, client_index in enumerate(client_indices):
      accumulated_length_client = len(self.clients[client_index].dataset_train) * client_index
      label_indices = self.clients[client_index].flip_labels(probability[count])
      a = 0
      #if(i == 0):
      #  flipped_label_indices = label_indices
      #else:
      for key in label_indices:
        flipped_label_indices[key+accumulated_length_client] = label_indices[key]

      i += 1
    random.setstate(old_state)
    if(store_flipped_label_indices):
      path = os.path.join(os.getcwd(), "results", result_folder_name, "flipped_label_indices"+".pkl")
      with open(path, 'wb') as file:
        pickle.dump(flipped_label_indices, file)
    a = 0

  def copy_files_to_local(self, dataset_name_list):
    for client in self.clients:
        client.copy_files_to_local(dataset_name_list)

