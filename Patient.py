import numpy as np
import os
from pathlib import Path

class Patient:
    
  def __init__(self, global_id, ethnicity, insurance, language, gender):
      self.global_id = global_id
      self.ethnicity = ethnicity
      self.insurance = insurance
      self.language = language
      self.gender = gender

      self.scans = []

  def add_scan_list(self, scan_list):
    self.scans += scan_list

  def contains_scan_with_positive_condition(self, condition_name):
    for index, row in self.dfScans.iterrows():
      if(row[condition_name] == 1):
        return True
    return False