import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

foldernames = ["plot_documents_reduced_maxtranslation_0.1final_as_is", "plot_documents_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.025", "plot_documents_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.05", "plot_documents_reduced_maxtranslation_0.1_rewardsystems_labelflip_0.075"]

filename_start = "Profits_performance_"
filename_starts = [filename_start, filename_start, filename_start, filename_start]
#filename_start = "Shapley_LR_method_predictive_performance_"
#filename_starts = ["Shapley_performance_", filename_start, filename_start, filename_start]
column_to_plot = "reward"
#column_to_plot = "Shapley value"


path = os.path.join(os.getcwd(), "results")

df_means = []
df_hs = []
for i, foldername in enumerate(foldernames):
    df = pd.read_csv(os.path.join(path, foldername, filename_starts[i]+"means.csv"))
    df_means.append(df)
    df = pd.read_csv(os.path.join(path, foldername, filename_starts[i]+"hs.csv"))
    df_hs.append(df)

# bar plot of dfs

import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 4
means_nih = []
means_chexpert = []
means_mimic = []

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

# transform df_means list to client specific lists
means_chexpert_1 = []
means_chexpert_2 = []
means_nih_1 = []
means_nih_2 = []
means_mimic_1 = []
means_mimic_2 = []
hs_chexpert_1 = []
hs_chexpert_2 = []
hs_nih_1 = []
hs_nih_2 = []
hs_mimic_1 = []
hs_mimic_2 = []

for i, df in enumerate(df_means):
    means_nih_1.append(df_means[i][column_to_plot][2])
    means_nih_2.append(df_means[i][column_to_plot][3])
    means_chexpert_1.append(df_means[i][column_to_plot][0])
    means_chexpert_2.append(df_means[i][column_to_plot][1])
    means_mimic_1.append(df_means[i][column_to_plot][4])
    means_mimic_2.append(df_means[i][column_to_plot][5])
    hs_nih_1.append(df_hs[i][column_to_plot][2])
    hs_nih_2.append(df_hs[i][column_to_plot][3])
    hs_chexpert_1.append(df_hs[i][column_to_plot][1])
    hs_chexpert_2.append(df_hs[i][column_to_plot][2])
    hs_mimic_1.append(df_hs[i][column_to_plot][4])
    hs_mimic_2.append(df_hs[i][column_to_plot][5])

rects1 = plt.bar(index+0*bar_width, means_chexpert_1, bar_width,
alpha=opacity,
color='#003f5c',
label='CXP, unchanged', yerr=hs_chexpert_1)

rects2 = plt.bar(index+1*bar_width, means_chexpert_2, bar_width,
alpha=opacity,
color='#003f5c',
label='CXP, labels flipped', yerr=hs_chexpert_2, hatch='//')

rects3 = plt.bar(index+2*bar_width, means_nih_1, bar_width,
alpha=opacity,
color='#ffa600',
label='NIH, unchanged', yerr=hs_nih_1)

rects4 = plt.bar(index+3*bar_width, means_nih_2, bar_width,
alpha=opacity,
color=['#ffa600'],
label='NIH, labels flipped', yerr=hs_nih_2, hatch='//')

rects5 = plt.bar(index+4*bar_width, means_mimic_1, bar_width,
alpha=opacity,
color='#bc5090',
label='CXR, unchanged', yerr=hs_mimic_1)

rects6 = plt.bar(index+5*bar_width, means_mimic_2, bar_width,
alpha=opacity,
color='#bc5090',
label='CXR, labels flipped', yerr=hs_mimic_2, hatch='//')

plt.xlabel('Share of flipped labels for specific clients [%]')
plt.ylabel('Reward [MUs]')
plt.title('Mean reward and 95% confidence intervals by client')
plt.xticks(index + bar_width * 3, ('0', '2.5', '5', '7.5'))
# show legend top left
plt.legend(loc='upper left')

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 13 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)

plt.tight_layout()
plt.show()

