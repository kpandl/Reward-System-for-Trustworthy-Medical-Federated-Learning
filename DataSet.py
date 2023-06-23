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
import torchvision.transforms as transforms
import pandas as pd

class DataSet(Dataset):
    
  def __init__(self, datasetName, transform=None, **kwargs):
    self.datasetName = datasetName
    self.transform = transform

    self.patients = []
    self.scans = []

    self.demo_mode = False

    self.use_pre_stored_files = False

    # open config
    with open('config.json') as config_file:
        config = json.load(config_file)  
    self.config = config

    if(not "patients" in kwargs or not "scans" in kwargs):
      if(datasetName=="nih"):
        df = pd.read_csv(config["path_to_data_entry_nih"])
        conditions = {}

        for patient_id in df['Patient ID'].unique():
          scans_for_patient = df[df['Patient ID']==patient_id]

          ethnicity = None
          insurance = None
          language = None
          gender = scans_for_patient['Patient Gender'].mode()[0] # "M" or "F"

          patient = Patient(patient_id, ethnicity, insurance, language, gender)

          scan_list = []

          for index_j, row in scans_for_patient.iterrows():
            age = row['Patient Age']

            row_conditions = row["Finding Labels"].split("|")
            atelectasis = 1 if "Atelectasis" in row_conditions else None
            cardiomegaly = 1 if "Cardiomegaly" in row_conditions else None
            consolidation = 1 if "Consolidation" in row_conditions else None
            edema = 1 if "Edema" in row_conditions else None
            emphysema = 1 if "Emphysema" in row_conditions else None
            nodule = 1 if "Nodule" in row_conditions else None
            no_finding = 1 if "No Finding" in row_conditions else None
            pleural_effusion = 1 if "Effusion" in row_conditions else None
            pleural_thickening = 1 if "Pleural_Thickening" in row_conditions else None
            pneumonia = 1 if "Pneumonia" in row_conditions else None
            pneumothorax = 1 if "Pneumothorax" in row_conditions else None
            fibrosis = 1 if "Fibrosis" in row_conditions else None
                        
            if(config["device_name"] == "notebook"):
              row['Image Index'] = "00030799_000.png"

            original_scan_path = os.path.join(os.getcwd(), *config["path_to_data_images_nih"].split("/"), row['Image Index'])
            learning_path = original_scan_path
            relative_scan_path = os.path.join(row['Image Index'])

            if(config["device_name"] == "bwunicluster"):
              learning_path = os.path.join(os.getcwd(), "..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data2", "nih", row['Image Index'])

            if(config["device_name"] == "bwforcluster"):
              learning_path = os.path.join("ml_data2", "nih", row['Image Index'])

            pre_store_path = learning_path+".npz"

            scan_list.append(Scan("nih", patient, age, row['View Position'], original_scan_path, learning_path, pre_store_path, relative_scan_path, atelectasis=atelectasis, cardiomegaly=cardiomegaly, consolidation=consolidation, edema=edema, emphysema=emphysema, nodule=nodule, no_finding=no_finding, pleural_effusion=pleural_effusion, pleural_thickening=pleural_thickening, pneumonia=pneumonia, pneumothorax=pneumothorax, fibrosis=fibrosis))
          
          patient.add_scan_list(scan_list)

          self.patients += [patient]
          self.scans += scan_list

      if(datasetName=="chexpert"):
        df = pd.read_csv(config["path_to_train_csv_chexpert"])
        df['Patient'] = pd.Series([int(i.split("patient")[1].split("/")[0]) for i in df['Path']], index=df.index)
        df['Study'] = pd.Series([int(i.split("study")[1].split("/")[0]) for i in df['Path']], index=df.index)
        df['Corresponding Datasplit'] = pd.Series([[] for i in df['Path']], index=df.index)
        df['globalScanID']=list(df.index)

        for patient_id in df['Patient'].unique():
          b = 0

          scans = df[df['Patient']==patient_id]

          ethnicity = None
          insurance = None
          language = None
          gender_tmp = scans['Sex']
          if(gender_tmp.iloc[0] == "Male"):
            gender = "M"
          if(gender_tmp.iloc[0] == "Female"):
            gender = "F"

          patient = Patient(patient_id, ethnicity, insurance, language, gender)

          scan_list = []

          for index_j, row in scans.iterrows():
            age = row['Age']
            
            if(config["device_name"] == "notebook"):
              row['Path'] = "CheXpert-v1.0-small/train/patient64537/study1/view1_frontal.jpg"
            
            original_scan_path = os.path.join(os.getcwd(), *config["path_to_data_images_chexpert"].split("/"), *row['Path'].split("/"))
            learning_path = original_scan_path
            relative_scan_path = os.path.join(*row['Path'].split("/"))

            if(config["device_name"] == "bwunicluster"):
              learning_path = os.path.join(os.getcwd(), "..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data2", "chexpert", *row['Path'].split("/"))

            if(config["device_name"] == "bwforcluster"):
              learning_path = os.path.join("ml_data2", "chexpert", *row['Path'].split("/"))

            pre_store_path = learning_path+".npz"

            scan_list.append(Scan("chexpert", patient, age, row['AP/PA'], original_scan_path, learning_path, pre_store_path, relative_scan_path, atelectasis=row['Atelectasis'], cardiomegaly=row['Cardiomegaly'], consolidation=row['Consolidation'], edema=row['Edema'], enlarged_cardiomediastinum=row['Enlarged Cardiomediastinum'], fracture=row['Fracture'], lung_lesion=row['Lung Lesion'], lung_opacity=row['Lung Opacity'], no_finding=row['No Finding'], pleural_effusion=row['Pleural Effusion'], pleural_other=row['Pleural Other'], pneumonia=row['Pneumonia'], pneumothorax=row['Pneumothorax'], support_devices=row['Support Devices']))

          patient.add_scan_list(scan_list)

          self.patients += [patient]
          self.scans += scan_list

      if(datasetName=="mimic"):
        df = pd.read_csv(config["path_to_metadata_csv_mimic"])
        df_labels = pd.read_csv(config["path_to_chexpertlabels_csv_mimic"])
        df_patientInfos = pd.read_csv(config["path_to_patientdetails_csv_mimic"])
        df_patientInfos_ages_genders = pd.read_csv(config["path_to_patientages_csv_mimic"])

        df['globalScanID']=list(df.index)

        largest_unused_patient_id = 0
        assigned_subject_id_dict = {}

        for index, row in df.iterrows():
          subject_id = row['subject_id']
          if(not subject_id in assigned_subject_id_dict):
            assigned_subject_id_dict[subject_id] = largest_unused_patient_id
            largest_unused_patient_id += 1

        df['globalPatientID'] = df['subject_id'].map(assigned_subject_id_dict)

        mergedDf = pd.merge(df, df_labels, on=['study_id', 'subject_id'])

        patientCounter_withInfos = 0
        patientCounter_withoutInfos = 0

        for index, subject_id in enumerate(mergedDf['subject_id'].unique()):
          subDf = mergedDf[mergedDf['subject_id']==subject_id]

          patientInfos = df_patientInfos[df_patientInfos['subject_id']==subject_id]

          df_ages_genders_infos = df_patientInfos_ages_genders[df_patientInfos_ages_genders['subject_id']==subject_id]

          ethnicity = None
          insurance = None
          language = None
          gender = None

          if(len(patientInfos) > 0):
            ethnicity = patientInfos.ethnicity.mode()[0]
            insurance = patientInfos.insurance.mode()[0]
            language = patientInfos.language.mode()[0]
            gender = df_ages_genders_infos.iloc[0]['gender']
            patientCounter_withInfos += 1
          else:
            patientCounter_withoutInfos += 1

          patient = Patient(subject_id, ethnicity, insurance, language, gender)

          scan_list = []

          for index_j, row in subDf.iterrows():
            age = None
            if(len(df_ages_genders_infos) > 0):
              age = df_ages_genders_infos.iloc[0]['anchor_age']
                        
            if(config["device_name"] == "notebook"):
              row['dicom_id'] = "5aea5877-40b40fee-5bccd163-ca1bf0ce-a95c213d"
              row['study_id']=52341872
              patient.global_id=10999737
            original_scan_path = os.path.join(os.getcwd(), *config["path_to_data_images_mimic"].split("/"), "p"+str(patient.global_id)[:2],"p"+str(patient.global_id),"s"+str(row['study_id']),row['dicom_id']+".jpg")
            learning_path = original_scan_path
            relative_scan_path = os.path.join("p"+str(patient.global_id)[:2],"p"+str(patient.global_id),"s"+str(row['study_id']),row['dicom_id']+".jpg")

            if(config["device_name"] == "bwunicluster"):
              learning_path = os.path.join(os.getcwd(), "..", "..", "..", "..", "..", "..", "..", "tmp", "ml_data2", "mimic", row['dicom_id']+".jpg")

            if(config["device_name"] == "bwforcluster"):
              learning_path = os.path.join("ml_data2", "mimic", row['dicom_id']+".jpg")

            pre_store_path = learning_path+".npz"

            scan_list.append(Scan("mimic", patient, age, row['ViewPosition'], original_scan_path, learning_path, pre_store_path, relative_scan_path, atelectasis=row['Atelectasis'], cardiomegaly=row['Cardiomegaly'], consolidation=row['Consolidation'], edema=row['Edema'], enlarged_cardiomediastinum=row['Enlarged Cardiomediastinum'], fracture=row['Fracture'], lung_lesion=row['Lung Lesion'], lung_opacity=row['Lung Opacity'], no_finding=row['No Finding'], pleural_effusion=row['Pleural Effusion'], pleural_other=row['Pleural Other'], pneumonia=row['Pneumonia'], pneumothorax=row['Pneumothorax'], support_devices=row['Support Devices'], study_id=row['study_id']))
            
          patient.add_scan_list(scan_list)

          self.patients += [patient]
          self.scans += scan_list

    else:
      self.patients=kwargs["patients"]
      self.scans=kwargs["scans"]

    if("label_mode" in kwargs):
      self.label_mode = kwargs["label_mode"]

  def update_config(self):
    with open('config.json') as config_file:
        config = json.load(config_file)  
    self.config = config

  def pre_store(self):
      self.set_transform(transform=transforms.Compose([transforms.Resize((320,320)),transforms.ToTensor()]))
      print("starting for loop")
      for i in range(len(self)):
        my_file = Path(self.scans[i].pre_store_path+"abc")
        if not my_file.is_file():
          img = self[i][0][0]
          np.savez_compressed(self.scans[i].pre_store_path, img=img.numpy())
          if(i%1000==0):
            print(str(i), "finished")

  def __len__(self):
      return len(self.scans)

  def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      exception_occured = False
      my_file_original = Path(self.scans[idx].original_scan_path)
      my_file = Path(self.scans[idx].learning_path)
        
      if(self.scans[idx].dataset_name=="nih"):
        if my_file.is_file():
          try:
            img = imageio.imread(self.scans[idx].learning_path)
          except:
            print("Exception occured in imageio.imread(self.scans[idx].learning_path) for path, idx", self.scans[idx].learning_path, ",", idx)
            return self[0]
        else:
          if my_file_original.is_file():
            tmp_a = self.scans[idx].learning_path.rfind("/")
            dest_path = self.scans[idx].learning_path[0:tmp_a]
            Path(dest_path).mkdir(parents=True, exist_ok=True)
            copyfile(self.scans[idx].original_scan_path, self.scans[idx].learning_path)
            print("reading image after copying", self.scans[idx].original_scan_path, "to", self.scans[idx].learning_path)
            img = imageio.imread(self.scans[idx].learning_path)
          else:
            print("my_file_original not existing for path, idx", self.scans[idx].learning_path, ",", idx)
            return self[0]

      if(self.scans[idx].dataset_name=="mimic" or self.scans[idx].dataset_name=="chexpert"):
        if(not self.use_pre_stored_files):
          if my_file.is_file():
            try:
              img = imread(self.scans[idx].learning_path)
            except:
              exception_occured = True
              print("Exception occured in imread(self.scans[idx].learning_path) for idx", idx)
              return self[0]
          else:
            tmp_a = self.scans[idx].learning_path.rfind("/")
            dest_path = self.scans[idx].learning_path[0:tmp_a]
            Path(dest_path).mkdir(parents=True, exist_ok=True)
            copyfile(self.scans[idx].original_scan_path, self.scans[idx].learning_path)
            print("reading image after copying", self.scans[idx].original_scan_path, "to", self.scans[idx].learning_path)
            img = imread(self.scans[idx].learning_path)
        else:
          img = np.load(self.scans[idx].pre_store_path, allow_pickle=True)["img"]

      if(not self.use_pre_stored_files):
        try:
          if(self.scans[idx].dataset_name=="mimic" or self.scans[idx].dataset_name=="chexpert"):
            if len(img.shape) == 2:
              img = img[:, :, np.newaxis]
              img = np.concatenate([img, img, img], axis=2)

          if(self.scans[idx].dataset_name=="nih"):
            if len(img.shape) == 2:
              img = img[:, :, np.newaxis]
              img = np.concatenate([img, img, img], axis=2)
            if len(img.shape)>2:
              img = img[:,:,0]
              img = img[:, :, np.newaxis]
              img = np.concatenate([img, img, img], axis=2)
        except:
          exception_occured = True
          print("Exception occured in array reconfiguration block for idx", idx)
          return self[0]
        
      if(self.use_pre_stored_files):
        img = img * 255
        img = img.astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

      try:
        img = Image.fromarray(img)
      except:
        exception_occured = True
        print("Exception occured in Image.fromarray(img) for idx", idx)
        return self[0]

      landmarks = np.array([int(self.scans[idx].atelectasis==1)])
      sample = {'image': img, 'landmarks': landmarks}

      if(self.label_mode == "intersection"):
        label = torch.FloatTensor(np.array([float(self.scans[idx].atelectasis==1), float(self.scans[idx].cardiomegaly==1), float(self.scans[idx].consolidation==1), float(self.scans[idx].edema==1), float(self.scans[idx].no_finding==1), float(self.scans[idx].pleural_effusion==1), float(self.scans[idx].pneumonia==1), float(self.scans[idx].pneumothorax==1)], dtype=float))

      label_a = torch.FloatTensor(np.zeros(5, dtype=float))

      if self.transform:
          sample["image"] = self.transform(sample["image"])

      return sample["image"], label

  def set_demo_mode(self, demo_mode):
    self.demo_mode = demo_mode

  def merge_with_another_dataset(self, other_dataset, merge_names=True):
    self.patients += other_dataset.patients
    self.scans += other_dataset.scans
    if(merge_names):
      self.datasetName += "_" + other_dataset.datasetName
    return self

  def set_label_mode(self, label_mode):
    self.label_mode = label_mode
    if(self.label_mode == "intersection"):
      self.label_names = ["atelectasis", "cardiomegaly", "consolidation", "edema", "no_finding", "pleural_effusion", "pneumonia", "pneumothorax"]

  def get_label_count(self):
    if(self.label_mode == "intersection"):
      return len(self.label_names)

  def set_transform(self, transform):
    self.transform = transform

  def generate_subset(self, start_share, end_share):
    num_scans_total = len(self.scans)

    start_index = math.floor(start_share * num_scans_total)
    end_index = math.floor(end_share * num_scans_total)

    scan_counter = 0
    patient_counder = 0

    for index, patient in enumerate(self.patients):
      if(scan_counter >= start_index):
        break
      patient_counder += 1
      scan_counter += len(patient.scans)
    
    patients_new = []
    scans_new = []

    while(start_share*num_scans_total+len(scans_new) < end_share*num_scans_total):
      if(patient_counder > len(self.patients) - 1):
        break
      patients_new += [self.patients[patient_counder]]
      scans_new += self.patients[patient_counder].scans
      patient_counder += 1

    print("Subset of", self.datasetName, "with", len(patients_new), "patients, ", len(scans_new), "scans")
    return DataSet(self.datasetName, patients=patients_new, scans=scans_new)

  def generate_subset_absolute(self, start_patient, num_scans):

    scan_counter = 0
    patients_new = []
    scans_new = []

    last_patient_added = start_patient

    for index, patient in enumerate(self.patients):
      if(index >= start_patient and scan_counter < num_scans):
        
        patients_new += [patient]
        scans_new += patient.scans
        scan_counter += len(patient.scans)

        last_patient_added = index
    
    print("Subset of", self.datasetName, "with", len(patients_new), "patients, ", len(scans_new), "scans")
    return last_patient_added, DataSet(self.datasetName, patients=patients_new, scans=scans_new[0:num_scans])

  def update_patient_list_based_on_scans(self):

    self.patients = []
    for scan in self.scans:
      if(scan.corresponding_patient not in self.patients):
        self.patients.append(scan.corresponding_patient)

  def random_shuffle_patients(self, seed):
    random.seed(seed)
    random.shuffle(self.patients)

  def random_shuffle_scans(self, seed):
    random.seed(seed)
    random.shuffle(self.scans)

  def split_into_old_patient_and_young_patient_dataset_based_on_quartiles(self):

    corresponding_ages_of_scans = [scan.patient_age for scan in self.scans]
    df_corresponding_ages_of_scans = pd.DataFrame(corresponding_ages_of_scans)
    df_quartiles = df_corresponding_ages_of_scans.quantile([0.3, 0.7])

    young_boundry = int(df_quartiles.values[0][0])
    old_boundry = int(df_quartiles.values[1][0])

    ds_old, ds_young = self.split_into_old_patient_and_young_patient_dataset(age_boundry_young=young_boundry, age_boundry_old=old_boundry)

    return ds_old, ds_young

  def split_into_old_patient_and_young_patient_dataset(self, age_boundry_young=29, age_boundry_old=65):

    patients_young = []
    scans_young = []

    patients_old = []
    scans_old = []

    for patient in self.patients:
      ages_of_patient = [scan.patient_age for scan in patient.scans]

      young_scan_counter = 0
      old_scan_counter = 0

      for age in ages_of_patient:
        if(age != None and age <= age_boundry_young):
          young_scan_counter += 1
        if(age != None and age >= age_boundry_old):
          old_scan_counter += 1

      if(old_scan_counter > young_scan_counter and old_scan_counter > 0):
        patients_old.append(patient)
        for scan in patient.scans:
          if(scan.patient_age != None and scan.patient_age >= age_boundry_old):
            scans_old.append(scan)

      if(young_scan_counter >= old_scan_counter and young_scan_counter > 0):
        patients_young.append(patient)
        for scan in patient.scans:
          if(scan.patient_age != None and scan.patient_age <= age_boundry_young):
            scans_young.append(scan)

    return DataSet(self.datasetName, patients=patients_old, scans=scans_old), DataSet(self.datasetName, patients=patients_young, scans=scans_young)
    
  def flip_labels(self, probability):
    flipped_label_indices = {}
    if(self.label_mode == "intersection"):
      for idx in range(len(self.scans)):
        for label_index in range(len(self.label_names)):
          if(random.random() < probability):
            if(getattr(self.scans[idx],self.label_names[label_index]) == 1):
              setattr(self.scans[idx],self.label_names[label_index], 0)
            else:
              setattr(self.scans[idx],self.label_names[label_index], 1)
            if(idx in flipped_label_indices.keys()):
              flipped_label_indices[idx] += 1
            else:
              flipped_label_indices[idx] = 1
    return flipped_label_indices