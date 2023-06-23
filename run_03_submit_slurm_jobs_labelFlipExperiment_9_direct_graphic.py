import pandas as pd
import os
import numpy as np
import scipy.stats
import csv
from pathlib import Path

filenames = ["Shapley_federated_performance"]

path = os.path.join(os.getcwd(), "results", "plot_documents_reduced_maxtranslation_0.1_labelflip_40")

df_means = []
df_hs = []
for filename in filenames:
    df = pd.read_csv(os.path.join(path, filename+"_means.csv"))
    df_means.append(df)
    df = pd.read_csv(os.path.join(path, filename+"_hs.csv"))
    df_hs.append(df)

# bar plot of dfs

import matplotlib.pyplot as plt
import numpy as np

# data to plot
n_groups = 1
means_nih = []
means_chexpert = []
means_mimic = []

hs_nih = []
hs_chexpert = []
hs_mimic = []
for i in range(n_groups):
    means_nih.append(df_means[i].iloc[0]["operating point"])
    hs_nih.append(df_hs[i].iloc[0]["operating point"])
    means_nih.append(df_means[i].iloc[1]["operating point"])
    hs_nih.append(df_hs[i].iloc[1]["operating point"])
    means_nih.append(df_means[i].iloc[2]["operating point"])
    hs_nih.append(df_hs[i].iloc[2]["operating point"])
    means_nih.append(df_means[i].iloc[3]["operating point"])
    hs_nih.append(df_hs[i].iloc[3]["operating point"])
    means_nih.append(df_means[i].iloc[4]["operating point"])
    hs_nih.append(df_hs[i].iloc[4]["operating point"])
    means_nih.append(df_means[i].iloc[5]["operating point"])
    hs_nih.append(df_hs[i].iloc[5]["operating point"])

# create plot
fig, ax = plt.subplots()
index = np.arange(6)
bar_width = 0.9
opacity = 0.8

rects1 = plt.bar(index, means_nih, bar_width,
alpha=opacity,
color='b', yerr=hs_nih)

plt.xlabel('Client share of flipped labels [%]')
plt.ylabel('Shapley value [%]')
#plt.title('Mean shapley value and 95% confidence intervals\nby client')
plt.xticks(index, ('0', '5', '10', '15', '20', '25'))

# set size of plot
width_inch_from_cm = 13 / 2.54
height_inch_from_cm = 9 / 2.54
fig.set_size_inches(width_inch_from_cm, height_inch_from_cm)
plt.ylim(3, 7.2)

plt.tight_layout()
plt.show()

# compute steps in means_nih from one to the next
steps = []
for i in range(len(means_nih)-1):
    steps.append(means_nih[i+1]-means_nih[i])

# compute share of confidence intervals of means_nih
share_of_confidence_intervals = []
for i in range(len(hs_nih)):
    share_of_confidence_intervals.append(2*hs_nih[i]/means_nih[i])