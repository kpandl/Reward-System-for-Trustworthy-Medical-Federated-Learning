
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path
from scipy import stats
import copy

# tests significance of sex-based rewards for the NIH institutions

other_path_list = ["_reduced_maxtranslation_0.1final_100_0"]
amount_of_experiments = 40

SVs_path1 = []
SVs_other = []

for path_1 in other_path_list:

    print("path", path_1)

    other_path_list_smaller = copy.deepcopy(other_path_list)
    #other_path_list_smaller.remove(path_1)

    for other_path in other_path_list_smaller:

        print("other_path", other_path)

        SVs_path1 = []
        SVs_other = []
        for i in range(amount_of_experiments):
            df_1 = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+path_1, "Reward_with_coalition.csv"))
            df_other = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+other_path, "Reward_with_coalition.csv"))

            SVs_path1.append(df_1.iloc[0]["reward"])
            SVs_other.append(df_other.iloc[1]["reward"])

        a = stats.ttest_rel(SVs_path1, SVs_other)
        print(a)