from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import os
import copy

fig, axes = plt.subplots(nrows=4, ncols=2, gridspec_kw={'width_ratios': [1, 1]}, sharex=False, figsize=(11, 7))

def plot_shapley_value_chart(axes_object, computation_name, title=None, label_positions=[1, 1, 1, 1, 1, 1, 1]):

    name_of_results_folder = "plot_documents" + computation_name

    df_shapley_means = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Shapley_fairness_means.csv"))
    df_shapley_hs = pd.read_csv(os.path.join(os.getcwd(), "results", name_of_results_folder, "Shapley_fairness_hs.csv"))

    stdlist = list(df_shapley_hs["Shapley value"][0:6])
    stdlist.append(df_shapley_hs.iloc[-1]["Shapley value"])

    #Data to plot. Do not include a total, it will be calculated
    index = ["NIH-1", "NIH-2", "CXP-1", "CXP-2", "CXR-1", "CXR-2"]
    data = {'amount': list(df_shapley_means["Shapley value"][0:6])}

    #Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=data,index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)

    total = trans.sum().amount
    trans.loc["net"]= total
    blank.loc["net"] = total

    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    blank.loc["net"] = 0

    trans=trans.iloc[::-1]

    stdlist_plot = copy.deepcopy(stdlist)
    stdlist_plot.reverse()

    color_positive = "#003f5c"
    color_negative = "#ffa600"

    colors = []

    y = list(trans["amount"])
    x = list(trans.index.values)

    for val in y:
        if(val>=0):
            colors.append(color_positive)
        else:
            colors.append(color_negative)

    axes_object.barh(x, y, left=blank.iloc[::-1], xerr=stdlist_plot, height = 0.9, color=colors)

    if(computation_name=="_reduced_maxtranslation_0.1final_75_25"):
        a = 0

    loop = 0
    SV_sum = 0

    trans=trans.iloc[::-1]
    
    for index, row in trans.iterrows():
        if index == 'net':
            y = trans.iloc[loop].amount
        else:
            SV_sum += trans.iloc[loop].amount
            y = SV_sum
        
        if(label_positions[loop] == 1):
            if(row['amount'] > 0):
                axes_object.annotate("{:,.3f}".format(row['amount'])+"±"+"{:,.3f}".format(stdlist[loop])+"",(y+stdlist[loop],6-loop-0.2),ha="left")
            else:
                axes_object.annotate("{:,.3f}".format(row['amount'])+"±"+"{:,.3f}".format(stdlist[loop])+"",(max(y-row['amount'], y+stdlist[loop]),6-loop-0.2),ha="left")
        if(label_positions[loop] == -1):
            if (row['amount'] > 0):
                axes_object.annotate("{:,.3f}".format(row['amount'])+"±"+"{:,.3f}".format(stdlist[loop])+"",(y-row['amount'],6-loop-0.2),ha="right")
            else:
                axes_object.annotate("{:,.3f}".format(row['amount'])+"±"+"{:,.3f}".format(stdlist[loop])+"",(y-stdlist[loop],6-loop-0.2),ha="right")

        loop+=1

    axes_object.set_ylim(-0.04,7)

    if(title!=None):
        axes_object.set_title(title)

plot_shapley_value_chart(axes.flat[7], "_reduced_maxtranslation_0.1final_age_quantile_100_0", title=None)
plot_shapley_value_chart(axes.flat[5], "_reduced_maxtranslation_0.1final_age_quantile_75_25", title=None)
plot_shapley_value_chart(axes.flat[3], "_reduced_maxtranslation_0.1final_age_quantile_50_50", title=None)
plot_shapley_value_chart(axes.flat[1], "_reduced_maxtranslation_0.1final_age_as_is", title="Age-based splits")

plot_shapley_value_chart(axes.flat[6], "_reduced_maxtranslation_0.1final_100_0", title=None, label_positions=[1, 1, 1, 1, 1, 1, 1])
plot_shapley_value_chart(axes.flat[4], "_reduced_maxtranslation_0.1final_75_25", title=None, label_positions=[1, 1, 1, 1, 1, 1, 1])
plot_shapley_value_chart(axes.flat[2], "_reduced_maxtranslation_0.1final_50_50", title=None)
plot_shapley_value_chart(axes.flat[0], "_reduced_maxtranslation_0.1final_as_is", title="Sex-based splits")

axes.flat[0].get_shared_x_axes().join(axes.flat[0], axes.flat[2], axes.flat[4], axes.flat[6])
axes.flat[1].get_shared_x_axes().join(axes.flat[1], axes.flat[3], axes.flat[5], axes.flat[7])

axes.flat[0].set_xticklabels([])
axes.flat[2].set_xticklabels([])
axes.flat[4].set_xticklabels([])

axes.flat[1].set_xticklabels([])
axes.flat[3].set_xticklabels([])
axes.flat[5].set_xticklabels([])

axes.flat[1].set_yticklabels([])
axes.flat[3].set_yticklabels([])
axes.flat[5].set_yticklabels([])
axes.flat[7].set_yticklabels([])

for i in range(8):
    axes.flat[i].set_ylim([-0.5, 6.6])


axes.flat[0].set_xlim([-0.1, 0.9])
axes.flat[1].set_xlim([0, 7])

axes.flat[0].set_ylabel('as is')
axes.flat[2].set_ylabel('50/50')
axes.flat[4].set_ylabel('75/25')
axes.flat[6].set_ylabel('100/0')

axes.flat[6].set_xlabel('SV based on test AUROC female - test AUROC male [%]')
axes.flat[7].set_xlabel('SV based on test AUROC young - test AUROC old [%]')

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.show()