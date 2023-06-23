
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

for mode in ["Performance", "Bias", "Bias_Age", "Performance_Age"]:
    for i, final_path in enumerate(final_path_list):

        if(mode == "Performance" and "age" in final_path):
            continue
        if(mode == "Bias" and "age" in final_path):
            continue
        if(mode == "Bias_Age" and (not "age" in final_path and final_path != "_reduced_maxtranslation_0.1final_as_is")):
            continue
        if(mode == "Performance_Age" and not "age" in final_path):
            continue

        #if(mode == "Bias_Age" and final_path != "_reduced_maxtranslation_0.1final_as_is"):
        #    continue

        seed_list = list(range(amounts_of_experiments[i]))

        df_list = []

        column_names = ["institution","reward","profit"]

        for seed in seed_list:
            if(mode == "Performance"):
                df = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_performance_with_coalition.csv"))
            if(mode == "Bias"):
                df = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_with_coalition.csv"))
            if(mode == "Bias_Age"):
                df = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_age_with_coalition.csv"))
            if(mode == "Performance_Age"):
                df = pd.read_csv(os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_performance_age_with_coalition.csv"))

            df_list.append(df)

        Path(os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i])).mkdir(parents=True, exist_ok=True)
        
        if(mode == "Performance"):
            path_means_and_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_means_and_hs.csv")
            path_means = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_means.csv")
            path_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_hs.csv")
        if(mode == "Bias"):
            path_means_and_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_bias_means_and_hs.csv")
            path_means = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_bias_means.csv")
            path_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_bias_hs.csv")
        if(mode == "Bias_Age"):
            path_means_and_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_age_bias_means_and_hs.csv")
            path_means = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_age_bias_means.csv")
            path_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_age_bias_hs.csv")
        if(mode == "Performance_Age"):
            path_means_and_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_means_and_hs.csv")
            path_means = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_means.csv")
            path_hs = os.path.join(os.getcwd(), "results", "plot_documents"+final_path_list[i], "Profits_performance_hs.csv")
        
        if os.path.exists(path_means_and_hs):
            os.remove(path_means_and_hs)
        if os.path.exists(path_means):
            os.remove(path_means)
        if os.path.exists(path_hs):
            os.remove(path_hs)

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
                
                for df in df_list[1:]:
                    other_rows.append(df.iloc[row[0]])

                this_row_means = []
                this_row_half_spans = []

                for column_name in column_names[1:]:
                    values = []
                    values.append(this_row[column_name])
                    for other_row in other_rows:
                        values.append(other_row[column_name])
                    mean, half_span = mean_confidence_stats(values)
                    this_row_means.append(mean)
                    this_row_half_spans.append(half_span)
                
                this_row_combined_values = mean_and_half_interval_to_text(this_row_means, this_row_half_spans)

                logwriter.writerow([this_row[i] for i in column_names[:1]]+this_row_combined_values)
                logwriter_means.writerow([this_row[i] for i in column_names[:1]]+this_row_means)
                logwriter_half_spans.writerow([this_row[i] for i in column_names[:1]]+this_row_half_spans)