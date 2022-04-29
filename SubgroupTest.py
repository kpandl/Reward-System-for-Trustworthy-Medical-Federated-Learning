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

class SubgroupTest:
    
  def __init__(self, critereon, subgroup_name, filter_function):
    self.critereon = critereon
    self.subgroup_name = subgroup_name
    self.filter_function = filter_function

  def analyze_test(self, scan_list, gt, pred):
    scan_selection = []
    for i in range(len(scan_list)):
      if(self.filter_function(scan_list[i])):
        scan_selection.append(i)

    self.results_sample_size = len(scan_selection)
    self.results_auc = self.compute_roc_auc_for_subgroup_and_conditions(scan_selection, gt, pred)

  def write_test_analysis(self, logwriter, coalition_name, name_of_aggregation, client_name, model_type):
    logwriter.writerow([coalition_name, name_of_aggregation, client_name, model_type, self.critereon, self.subgroup_name, "AUC", "-", self.results_sample_size, sum(self.results_auc)/len(self.results_auc), sum(self.results_auc[:4] + self.results_auc[5:])/len(self.results_auc[:4] + self.results_auc[5:]), *self.results_auc])

  def compute_roc_auc_for_subgroup_and_conditions(self, subgroup_indices, ground_truth, predictions):
    AUCs = []

    ground_truth_for_indices = np.take(ground_truth.numpy(), subgroup_indices, 0)
    predictions_for_indices = np.take(predictions.numpy(), subgroup_indices, 0)

    for condition_id in range(len(predictions_for_indices[0])):
        gt_array = ground_truth_for_indices[:,condition_id]
        pred_array = predictions_for_indices[:,condition_id]
        if(len(set(gt_array)) > 1):
            auc_forCondition = roc_auc_score(gt_array, pred_array)
        else:
            auc_forCondition = np.NaN
        #print("AUC for condition", condition_id, ":", auc_forCondition)
        AUCs.append(auc_forCondition)

    return AUCs