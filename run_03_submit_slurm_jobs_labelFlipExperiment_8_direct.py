import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

final_path_list_gender = ["_reduced_maxtranslation_0.1_labelflip_40"]
final_path_list_age = []

age = False
age_addition = ""

if(age):
    final_path_list = final_path_list_age
    age_addition_pp = "_age"
    age_addition = "_age"
else:
    final_path_list = final_path_list_gender
    age_addition_pp = ""
    age_addition = "_gender"

amounts_of_experiments = [12] * len(final_path_list)

def mean_confidence_stats(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #return m, m-h, m+h
    return m, h

def mean_and_half_interval_to_text(means, half_intervals):
    result = []
    for i in range(len(means)):
        result.append(f'{means[i]:.6f}'+' ± '+f'{half_intervals[i]:.6f}')
    return result

for i, _ in enumerate(final_path_list):

    seed_list = list(range(amounts_of_experiments[i]))

    df_list = []

    column_names = ["Changed labels", "mean SV"]

    for seed in seed_list:
        df = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_federated_performance_per_datapoint_flipped_consideration.csv"))
        df_list.append(df)

    Path(os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i])).mkdir(parents=True, exist_ok=True)
    
    path_means_and_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Shapley_federated_performance_per_datapoint_flipped_consideration_means_and_hs.csv")
    path_means = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Shapley_federated_performance_per_datapoint_flipped_consideration_means.csv")
    path_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Shapley_federated_performance_per_datapoint_flipped_consideration_hs.csv")
    
    with open(path_means_and_hs, 'a', newline="") as logfile, open(path_means, 'a', newline="") as logfile_means, open(path_hs, 'a', newline="") as logfile_half_spans:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter_means = csv.writer(logfile_means, delimiter=',')
        logwriter_half_spans = csv.writer(logfile_half_spans, delimiter=',')

        logwriter.writerow(column_names)
        logwriter.writerow([final_path_list[i]])
        logwriter_means.writerow(column_names)
        logwriter_half_spans.writerow(column_names)

        for row in df_list[0].iterrows():
            this_row = row[1]
            other_rows = []
            a = 0
            for df in df_list[len(column_names)-3:]:
                a += 1
                other_rows.append(df.iloc[row[0]])

            this_row_means = []
            this_row_half_spans = []

            for column_name in column_names[len(column_names)-1:]:
                values = []
                values.append(this_row[column_name])
                for other_row in other_rows:
                    values.append(other_row[column_name])
                mean, half_span = mean_confidence_stats(values)
                this_row_means.append(100*mean)
                this_row_half_spans.append(100*half_span)
            
            this_row_combined_values = mean_and_half_interval_to_text(this_row_means, this_row_half_spans)

            logwriter.writerow([this_row[i] for i in column_names[:len(column_names)-2]]+this_row_combined_values)
            logwriter_means.writerow([this_row[i] for i in column_names[:len(column_names)-2]]+this_row_means)
            logwriter_half_spans.writerow([this_row[i] for i in column_names[:len(column_names)-2]]+this_row_half_spans)