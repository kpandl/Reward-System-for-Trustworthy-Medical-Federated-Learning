import numpy as np
import os
from pathlib import Path
import random
from DataSet import *
import pickle

print("running")

ds1_nih = DataSet("nih")
with open(os.path.join(os.getcwd(),"data", "ds_nih.pkl"), 'wb') as output:
    pickle.dump(ds1_nih, output, pickle.HIGHEST_PROTOCOL)

ds1_mimic = DataSet("mimic")
with open(os.path.join(os.getcwd(),"data", "ds_mimic.pkl"), 'wb') as output:
    pickle.dump(ds1_mimic, output, pickle.HIGHEST_PROTOCOL)

ds1_chexpert = DataSet("chexpert")
with open(os.path.join(os.getcwd(),"data", "ds_chexpert.pkl"), 'wb') as output:
    pickle.dump(ds1_chexpert, output, pickle.HIGHEST_PROTOCOL)






print("finished")