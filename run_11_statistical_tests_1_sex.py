
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path
from scipy import stats
import copy

# tests overall AUC across different splits

other_path_list = ["_reduced_maxtranslation_0.1final_100_0", "_reduced_maxtranslation_0.1final_50_50", "_reduced_maxtranslation_0.1final_as_is", "_reduced_maxtranslation_0.1final_75_25"]
amount_of_experiments = 40

SVs_path1 = []
SVs_other = []

for path_1 in other_path_list:

    other_path_list_smaller = copy.deepcopy(other_path_list)
    other_path_list_smaller.remove(path_1)

    print(path_1)

    for other_path in other_path_list_smaller:
        SVs_path1 = []
        SVs_other = []
        for i in range(amount_of_experiments):
            df_1 = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+path_1, "Shapley_LR_method_predictive_performance_with_coalition.csv"))
            df_other = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+other_path, "Shapley_LR_method_predictive_performance_with_coalition.csv"))

            SVs_path1.append(df_1.iloc[-1]["Shapley value"])
            SVs_other.append(df_other.iloc[-1]["Shapley value"])

        a = stats.ttest_rel(SVs_path1, SVs_other)
        print(a)