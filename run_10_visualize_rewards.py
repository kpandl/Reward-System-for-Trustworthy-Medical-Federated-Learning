import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os
import copy

x = range(10)
y = range(10)

fig, axes = plt.subplots(nrows=1, ncols=3)

def plot_reward_and_profit_chart_as_is(axes_object, computation_name, title=None):

    name_of_results_folder = "plot_documents" + computation_name

    df_profit_performance_reward_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_performance_means.csv"))
    df_profit_performance_reward_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_performance_hs.csv"))

    df_profit_bias_reward_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_bias_means.csv"))
    df_profit_bias_reward_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_bias_hs.csv"))

    df_profit_age_bias_reward_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_age_bias_means.csv"))
    df_profit_age_bias_reward_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_age_bias_hs.csv"))

    index = ["NIH-1", "NIH-2", "CXP-1", "CXP-2", "CXR-1", "CXR-2"]
    x = np.arange(len(index))

    data_means_performance_reward = list(df_profit_performance_reward_means["reward"][0:6])
    data_hs_performance_reward = list(df_profit_performance_reward_hs["reward"][0:6])
    
    data_means_bias_reward = list(df_profit_bias_reward_means["reward"][0:6])
    data_hs_bias_reward = list(df_profit_bias_reward_hs["reward"][0:6])
    
    data_means_bias_age_reward = list(df_profit_age_bias_reward_means["reward"][0:6])
    data_hs_bias_age_reward = list(df_profit_age_bias_reward_hs["reward"][0:6])

    labels = ['G1']
    width = 0.3

    running_sum = 0.5

    data_means_performance_reward_additional = copy.deepcopy(data_means_performance_reward)
    data_means_bias_reward_additional = copy.deepcopy(data_means_bias_reward)
    data_means_bias_age_reward_additional = copy.deepcopy(data_means_bias_age_reward)

    axes_object.bar(x - width/2, data_means_performance_reward_additional, yerr=data_hs_performance_reward, label="Reward, pred. perfor.", align='edge', width=width, color='#003f5c')
    
    axes_object.bar(x + width/2, data_means_bias_reward_additional, yerr=data_hs_bias_reward, label="Reward, sex bias", align='edge',width=width, color='#ffa600')
    
    axes_object.bar(x + 3*width/2, data_means_bias_age_reward_additional, yerr=data_hs_bias_age_reward, label="Reward, age bias", align='edge',width=width, color='#bc5090')

    axes_object.set_xticks(x, index)

    #Start label loop
    loop = 0
    for i in range(6):

        axes_object.annotate("{:,.3f}".format(data_means_performance_reward_additional[loop])+"%±"+"{:,.3f}".format(data_hs_performance_reward[loop])+"%",(x[i] - 0/2,data_means_performance_reward_additional[loop]+data_hs_performance_reward[loop]+0.1),ha="center",rotation=90)
        axes_object.annotate("{:,.3f}".format(data_means_bias_reward_additional[loop])+"%±"+"{:,.3f}".format(data_hs_bias_reward[loop])+"%",(x[i] + width,data_means_bias_reward_additional[loop]+data_hs_bias_reward[loop]+0.1),ha="center",rotation=90)
        if(i==2 or i==3):
            axes_object.annotate("{:,.3f}".format(data_means_bias_age_reward_additional[loop])+"%\n±"+"{:,.3f}".format(data_hs_bias_age_reward[loop])+"%",(x[i] + 4*width/2,data_means_bias_age_reward_additional[loop]+data_hs_bias_age_reward[loop]+0.1),ha="center",rotation=90)
        else:
            axes_object.annotate("{:,.3f}".format(data_means_bias_age_reward_additional[loop])+"%±"+"{:,.3f}".format(data_hs_bias_age_reward[loop])+"%",(x[i] + 4*width/2,data_means_bias_age_reward_additional[loop]+data_hs_bias_age_reward[loop]+0.1),ha="center",rotation=90)


        loop+=1

    axes_object.set_ylabel('Reward [MU]')
    if(title!=None):
        axes_object.set_title(title)

def plot_reward_and_profit_chart(axes_object, computation_name, title=None, color_bias="#003f5c"):

    name_of_results_folder = "plot_documents" + computation_name

    df_profit_performance_reward_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_performance_means.csv"))
    df_profit_performance_reward_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_performance_hs.csv"))

    df_profit_bias_reward_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_bias_means.csv"))
    df_profit_bias_reward_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Profits_bias_hs.csv"))
    index = ["NIH-1", "NIH-2", "CXP-1", "CXP-2", "CXR-1", "CXR-2"]
    x = np.arange(len(index))

    data_means_performance_reward = list(df_profit_performance_reward_means["reward"][0:6])
    data_hs_performance_reward = list(df_profit_performance_reward_hs["reward"][0:6])
    
    data_means_bias_reward = list(df_profit_bias_reward_means["reward"][0:6])
    data_hs_bias_reward = list(df_profit_bias_reward_hs["reward"][0:6])

    a = 0
    labels = ['G1']
    width = 0.45

    running_sum = 0.5
    data_means_performance_reward_additional = copy.deepcopy(data_means_performance_reward)
    data_means_bias_reward_additional = copy.deepcopy(data_means_bias_reward)

    axes_object.bar(x - width/2, data_means_performance_reward, yerr=data_hs_performance_reward, label="Reward, pred. perfor.", align='edge', width=width, color='#003f5c')
    
    axes_object.bar(x + width/2, data_means_bias_reward, yerr=data_hs_bias_reward, label="Reward, sex bias", align='edge',width=width, color=color_bias)
    
    axes_object.set_xticks(x, index)

    loop = 0
    for i in range(6):
        
        axes_object.annotate("{:,.3f}".format(data_means_performance_reward[loop])+"%±\n"+"{:,.3f}".format(data_hs_performance_reward[loop])+"%",(x[i] - 0/2,data_means_performance_reward[loop]+data_hs_performance_reward[loop]),ha="center",rotation=90,linespacing=0.72)
        axes_object.annotate("{:,.3f}".format(data_means_bias_reward[loop])+"%±\n"+"{:,.3f}".format(data_hs_bias_reward[loop])+"%",(x[i] + width,data_means_bias_reward[loop]+data_hs_bias_reward[loop]),ha="center",rotation=90,linespacing=0.72)

        loop+=1

    axes_object.set_ylabel('Reward [MU]')
    if(title!=None):
        axes_object.set_title(title)

plot_reward_and_profit_chart_as_is(axes.flat[0], "_reduced_maxtranslation_0.1final_as_is", title="Sex-based 'as is' split")

plot_reward_and_profit_chart(axes.flat[1], "_reduced_maxtranslation_0.1final_100_0", title="Sex-based '100/0' split", color_bias="#ffa600")
plot_reward_and_profit_chart(axes.flat[2], "_reduced_maxtranslation_0.1final_age_quantile_100_0", title="Age-based '100/0' split", color_bias="#bc5090")

axes.flat[0].set_ylim([0, 26])
axes.flat[1].set_ylim([0, 22])
axes.flat[2].set_ylim([0, 22])

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)

plt.show()