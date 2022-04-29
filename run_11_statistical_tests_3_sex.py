
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path
from scipy import stats
import copy

# A general finding is that institutions with predominantly female patient data contribute more toward the overall AUROC than the corresponding institutions with predominantly male patient data for 100%/0% and 75%/25% sex-based splits. In the 75%/25% split, this finding is statistically significant for the CXR institutions and in the 100%/0% split, it is significant for all institutions. Similarly for the age 100%/0% and 75%/25% age-based splits,

other_path_list = ["_reduced_maxtranslation_0.1final_75_25", "_reduced_maxtranslation_0.1final_100_0"]
amount_of_experiments = 40

for j in range(3):
    print("inst", j, "and", j+1)
    for path_1 in other_path_list:

        print(path_1)

        SVs_inst1 = []
        SVs_inst2 = []

        for i in range(amount_of_experiments):
            df_1 = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+path_1, "Shapley_performance_with_coalition.csv"))

            SVs_inst1.append(df_1.iloc[j*2]["Shapley value"])
            SVs_inst2.append(df_1.iloc[j*2+1]["Shapley value"])

        a = stats.ttest_rel(SVs_inst1, SVs_inst2)
        print(a)



