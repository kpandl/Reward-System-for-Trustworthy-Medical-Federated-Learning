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
from math import comb
from DataSetCollection import *
from shutil import copyfile
from PIL import Image

def get_rescaled_size(width, height, min_target_value):
    min_actual = min(width, height)
    scaling_factor = min_target_value / min_actual

    return int(width * scaling_factor), int(height * scaling_factor)

target_pixel = 400

with open('config.json') as config_file:
    config_original = json.load(config_file)

with open('config_resized.json') as config_file:
    config_resized = json.load(config_file)

with open(os.path.join(os.getcwd(),"data", "ds_nih.pkl"), 'rb') as f:
    ds1_nih = pickle.load(f)
    
with open(os.path.join(os.getcwd(),"data", "ds_chexpert.pkl"), 'rb') as f:
    ds1_chexpert = pickle.load(f)

with open(os.path.join(os.getcwd(),"data", "ds_mimic.pkl"), 'rb') as f:
    ds1_mimic = pickle.load(f)

ds_list = [ds1_nih, ds1_chexpert, ds1_mimic]

print("starting")

for ds in ds_list:

    if(ds.datasetName == "nih"):
        i = 0
        Path(config_resized["path_to_data_entry_nih"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_data_entry_nih"], config_resized["path_to_data_entry_nih"])

        for scan in ds.scans:
            dest_path = os.path.join(config_resized["path_to_data_images_nih_absolute"], scan.relative_scan_path)
            dest_file = Path(dest_path)
            dest_file_parent = Path(dest_path).parent

            if(i == 0):
                print("hi")
                print("cwd", os.getcwd())
                print(dest_path)
                print(dest_file)
                print(dest_file_parent)

            if not dest_file.is_file():
                Path(dest_file).parent.mkdir(parents=True, exist_ok=True)

                image = Image.open(scan.original_scan_path)
                w, h = get_rescaled_size(image.width, image.height, target_pixel)
                image = image.resize((w,h))
                image.save(fp=dest_file)

            i += 1
            if(i % 1000 == 0):
                print(str(i), "finished")

        print("NIH finished")

    if(ds.datasetName == "chexpert"):

        i = 0

        Path(config_resized["path_to_train_csv_chexpert"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_train_csv_chexpert"], config_resized["path_to_train_csv_chexpert"])

        for scan in ds.scans:
            dest_path = os.path.join(config_resized["path_to_data_images_chexpert_absolute"], scan.relative_scan_path)
            dest_file = Path(dest_path)

            if not dest_file.is_file():
                Path(dest_file).parent.mkdir(parents=True, exist_ok=True)
                image = Image.open(scan.original_scan_path)
                w, h = get_rescaled_size(image.width, image.height, target_pixel)
                image = image.resize((w,h))
                image.save(fp=dest_file)

            i += 1
            if(i % 1000 == 0):
                print(str(i), "finished")

        print("Chexpert finished")

    if(ds.datasetName == "mimic"):

        i = 0

        Path(config_resized["path_to_metadata_csv_mimic"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_metadata_csv_mimic"], config_resized["path_to_metadata_csv_mimic"])

        Path(config_resized["path_to_chexpertlabels_csv_mimic"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_chexpertlabels_csv_mimic"], config_resized["path_to_chexpertlabels_csv_mimic"])

        Path(config_resized["path_to_patientdetails_csv_mimic"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_patientdetails_csv_mimic"], config_resized["path_to_patientdetails_csv_mimic"])

        Path(config_resized["path_to_patientages_csv_mimic"]).parent.mkdir(parents=True, exist_ok=True)
        copyfile(config_original["path_to_patientages_csv_mimic"], config_resized["path_to_patientages_csv_mimic"])

        for scan in ds.scans:
            dest_path = os.path.join(config_resized["path_to_data_images_mimic_absolute"], scan.relative_scan_path)
            dest_file = Path(dest_path)

            if not dest_file.is_file():
                Path(dest_file).parent.mkdir(parents=True, exist_ok=True)
                image = Image.open(scan.original_scan_path)
                w, h = get_rescaled_size(image.width, image.height, target_pixel)
                image = image.resize((w,h))
                image.save(fp=dest_file)

            i += 1
            if(i % 1000 == 0):
                print(str(i), "finished")

        print("MIMIC finished")

print("Completely finished")