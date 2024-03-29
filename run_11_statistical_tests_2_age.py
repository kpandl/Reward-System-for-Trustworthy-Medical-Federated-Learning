
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path
from scipy import stats
import copy

# test SVs for performance for 50/50 and as_is

other_path_list = ["_reduced_maxtranslation_0.1final_age_quantile_50_50", "_reduced_maxtranslation_0.1final_age_as_is"]
other_path_list = ["_reduced_maxtranslation_0.1final_age_quantile_50_50", "_reduced_maxtranslation_0.1final_age_quantile_75_25", "_reduced_maxtranslation_0.1final_age_quantile_100_0", "_reduced_maxtranslation_0.1final_age_as_is"]
amount_of_experiments = 40

for j in range(3):
    print("inst", j, "and", j+1)
    for path_1 in other_path_list:

        #print(path_1)

        SVs_inst1 = []
        SVs_inst2 = []

        for i in range(amount_of_experiments):
            df_1 = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(i)+path_1, "Shapley_LR_method_predictive_performance_age_with_coalition.csv"))

            if(i == 0):
                print(path_1, df_1.iloc[j*2]["dataset"], "and", df_1.iloc[j*2+1]["dataset"])


            SVs_inst1.append(df_1.iloc[j*2]["Shapley value"])
            SVs_inst2.append(df_1.iloc[j*2+1]["Shapley value"])

        a = stats.ttest_rel(SVs_inst1, SVs_inst2)
        print(a)



