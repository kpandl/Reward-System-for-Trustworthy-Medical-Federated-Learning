
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

final_path_list = ["_reduced_maxtranslation_0.1final_100_0", "_reduced_maxtranslation_0.1final_50_50", "_reduced_maxtranslation_0.1final_as_is", "_reduced_maxtranslation_0.1final_75_25", "_reduced_maxtranslation_0.1final_age_quantile_100_0", "_reduced_maxtranslation_0.1final_age_quantile_75_25", "_reduced_maxtranslation_0.1final_age_quantile_50_50", "_reduced_maxtranslation_0.1final_age_as_is"]
amounts_of_experiments = [40, 40, 40, 40, 40, 40, 40, 40]

for mode in ["Performance", "Bias", "Bias_Age", "Performance_Age"]:
    for i, _ in enumerate(final_path_list):

        seed = 0

        while True:
            
            if(mode == "Performance"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_LR_method_predictive_performance_with_coalition.csv")
                path_reward = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_performance_with_coalition.csv")
            if(mode == "Bias"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_LR_method_bias_gender_with_coalition.csv")
                path_reward = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_with_coalition.csv")
            if(mode == "Bias_Age"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_LR_method_bias_age_with_coalition.csv")
                path_reward = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_age_with_coalition.csv")
            if(mode == "Performance_Age"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_LR_method_predictive_performance_age_with_coalition.csv")
                path_reward = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_performance_age_with_coalition.csv")
                if(final_path_list[i] == "_reduced_maxtranslation_0.1final_age_quantile_100_0"):
                    a = 0


            if os.path.isfile(path):
                df = pd.read_csv(os.path.join(path))

                df_profits = pd.DataFrame(columns=['institution', 'reward', 'profit'])

                if(mode == 'Performance' or mode == 'Performance_Age'):
                    coalition_utility = df.iloc[-1]["Shapley value"]
                    reward_pot_distributed = 60 * coalition_utility / 0.5
                    for j in range(6):
                        reward = df.iloc[j]["Shapley value"] * 60 / 0.5
                        profit = reward - reward_pot_distributed / 6
                        df_profits.loc[j] = [df.iloc[j].dataset, reward, profit]
                        
                if(mode == 'Bias' or mode == 'Bias_Age'):
                    coalition_utility = df.iloc[-1]["Shapley value"]

                    shapley_values = df.iloc[0:6]["Shapley value"]

                    if(coalition_utility >= 0):
                        sign = 1
                    else:
                        sign = -1

                    sign_times_shapley = []

                    for j in range(6):
                        sign_times_shapley.append(shapley_values[j]*sign)
                    
                    w = np.argmax(sign_times_shapley)

                    delta=[]

                    for j in range(6):
                        delta.append(shapley_values[w]-shapley_values[j])

                    sum_delta = sum(delta)

                    reward_pot_distributed = 60 * (1 - abs(coalition_utility))/1 # unit of coalition_utility is %

                    for j in range(6):
                        #reward = (delta[j] / sum_delta) * reward_pot_distributed
                        reward = (delta[j] / (6*shapley_values[w]-coalition_utility)) * reward_pot_distributed
                        profit = reward - reward_pot_distributed / 6
                        df_profits.loc[j] = [df.iloc[j].dataset, reward, profit]

                df_profits.to_csv(path_reward)
            else:
                break

            seed += 1
            print(seed)