
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

final_path_list = ["_reduced_maxtranslation_0.1final_100_0", "_reduced_maxtranslation_0.1final_50_50", "_reduced_maxtranslation_0.1final_as_is", "_reduced_maxtranslation_0.1final_75_25", "_reduced_maxtranslation_0.1final_age_quantile_100_0", "_reduced_maxtranslation_0.1final_age_quantile_75_25", "_reduced_maxtranslation_0.1final_age_quantile_50_50", "_reduced_maxtranslation_0.1final_age_as_is"]
amounts_of_experiments = [40, 40, 40, 40, 40, 40, 40, 40]

def mean_confidence_stats(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def mean_and_half_interval_to_text(means, half_intervals):
    result = []
    for i in range(len(means)):
        result.append(f'{means[i]:.6f}'+' ± '+f'{half_intervals[i]:.6f}')
    return result

for mode in ["Performance", "Bias", "Bias_Age"]:
    for i, _ in enumerate(final_path_list):

        seed = 0

        while True:
            
            if(mode == "Performance"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_performance.csv")
                path_with_coalition = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_performance_with_coalition.csv")

            if(mode == "Bias"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley.csv")
                path_with_coalition = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_with_coalition.csv")

            if(mode == "Bias_Age"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_age.csv")
                path_with_coalition = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_age_with_coalition.csv")

            if os.path.isfile(path):
                df = pd.read_csv(os.path.join(path))
                auc_rows = df.loc[df['testing critereon'] == "AUC"]

                coalition_auc_gain = sum(auc_rows["Shapley value"])

                df_new = df.append({'dataset': "coalition", "testing critereon": "AUC", "operating point": "-", "Shapley value": coalition_auc_gain}, ignore_index = True)

                df_new.to_csv(path_with_coalition)
            else:
                break

            seed += 1