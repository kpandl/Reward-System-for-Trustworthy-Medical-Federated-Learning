import numpy as np
import os
from pathlib import Path

class Scan:
    
  def __init__(self, dataset_name, corresponding_patient, patient_age, view_position, original_scan_path, learning_path, pre_store_path, relative_scan_path, **kwargs):
      self.dataset_name = dataset_name
      self.corresponding_patient = corresponding_patient
      self.patient_age = patient_age
      self.view_position = view_position
      self.original_scan_path = original_scan_path
      self.learning_path = learning_path
      self.pre_store_path = pre_store_path
      self.relative_scan_path = relative_scan_path
      if("atelectasis" in kwargs):
        self.atelectasis = kwargs['atelectasis']
      if("cardiomegaly" in kwargs):
        self.cardiomegaly = kwargs['cardiomegaly']
      if("consolidation" in kwargs):
        self.consolidation = kwargs['consolidation']
      if("edema" in kwargs):
        self.edema = kwargs['edema']
      if("emphysema" in kwargs):
        self.emphysema = kwargs['emphysema']
      if("no_finding" in kwargs):
        self.no_finding = kwargs['no_finding']
      if("nodule" in kwargs):
        self.nodule = kwargs['nodule']
      if("pleural_effusion" in kwargs):
        self.pleural_effusion = kwargs['pleural_effusion']
      if("pneumonia" in kwargs):
        self.pneumonia = kwargs['pneumonia']
      if("pleural_thickening" in kwargs):
        self.pleural_thickening = kwargs['pleural_thickening']
      if("pneumothorax" in kwargs):
        self.pneumothorax = kwargs['pneumothorax']
      if("enlarged_cardiomediastinum" in kwargs):
        self.enlarged_cardiomediastinum = kwargs['enlarged_cardiomediastinum']
      if("fracture" in kwargs):
        self.fracture = kwargs['fracture']
      if("hernia" in kwargs):
        self.hernia = kwargs['hernia']
      if("infiltration" in kwargs):
        self.infiltration = kwargs['infiltration']
      if("lung_lesion" in kwargs):
        self.lung_lesion = kwargs['lung_lesion']
      if("lung_opacity" in kwargs):
        self.lung_opacity = kwargs['lung_opacity']
      if("mass" in kwargs):
        self.mass = kwargs['mass']
      if("pleural_other" in kwargs):
        self.pleural_other = kwargs['pleural_other']
      if("support_devices" in kwargs):
        self.support_devices = kwargs['support_devices']
      if("study_id" in kwargs):
        self.study_id = kwargs['study_id']
      
  def contains_scan_with_positive_condition(self, condition_name):
    for index, row in self.dfScans.iterrows():
      if(row[condition_name] == 1):
        return True
    return False