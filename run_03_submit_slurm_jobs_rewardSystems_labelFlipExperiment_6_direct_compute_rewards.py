
import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

final_path_list = ["_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.05", "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.075", "_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.025"]
amounts_of_experiments = [12, 12, 12]

for mode in ["Performance"]:
    for i, _ in enumerate(final_path_list):

        seed = 0

        while True:
            
            if(mode == "Performance"):
                path = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Shapley_LR_method_predictive_performance.csv")
                path_reward = os.path.join(os.getcwd(), "results", "default"+str(seed)+final_path_list[i], "Reward_performance.csv")

            if os.path.isfile(path):
                print("File exists")
                df = pd.read_csv(os.path.join(path))

                df_profits = pd.DataFrame(columns=['institution', 'reward', 'profit'])

                if(mode == 'Performance' or mode == 'Performance_Age'):
                    coalition_utility = sum(df["Shapley value"])
                    reward_pot_distributed = 60 * coalition_utility / 0.5
                    for j in range(6):
                        reward = df.iloc[j]["Shapley value"] * 60 / 0.5
                        profit = reward - reward_pot_distributed / 6
                        df_profits.loc[j] = [df.iloc[j].dataset, reward, profit]
                        
                df_profits.to_csv(path_reward)
            else:
                print("File does not exist:", path)
                break

            seed += 1
            print(seed)