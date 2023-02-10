import matplotlib.pyplot as plt
import csv
import sys
import os
import pandas as pd 
from scipy.stats import ttest_ind
import numpy as np

file_name = 'data1.csv'

plain_data = pd.read_csv(file_name)
data = plain_data.groupby("down_level")
print(data)
mean_data = data.mean()
std_data = data.std()

fig, ax = plt.subplots()

# mm = list(zip(min_data[y_name], max_data[y_name]))
print(mean_data['test_bleu'])
print(mean_data['test_chrf'])
# ax.errorbar(mean_data.index, mean_data['test_bleu'], yerr=std_data['test_bleu'], fmt='o', linewidth=2, capsize=6)
# ax.fill_between(mean_data.index, mean_data[y_name] - std_data[y_name], mean_data[y_name] + std_data[y_name], alpha=.5, linewidth=0)
ax.plot(mean_data.index, mean_data['test_bleu'] * 100, linewidth=2, marker='o', label="BLEU")
ax.plot(mean_data.index, mean_data['test_chrf'], linewidth=2, marker='x', label="chrf")
plt.legend(loc = 'upper left', framealpha = 0.75)                

#Welch ttest
# ttest = ttest_ind(plain_data[plain_data['down_level'] == 0.0][y_name].to_numpy(), plain_data[plain_data['down_level'] == 1.0][y_name].to_numpy(), equal_var = False, alternative='less')
# print(ttest) #exact match Ttest_indResult(statistic=3.1980107453341544, pvalue=0.008130031236723555) - ge has statistically significant effect

plt.xticks(rotation = 25)
plt.axvline(x = 0, color = 'tab:gray', linestyle = 'dashed', label = 'GE enabled')
plt.axvline(x = 1, color = 'tab:gray', linestyle = 'dashed', label = 'GE disabled')
plt.xlabel('Down level')
plt.ylabel('Chrf and BLEU')
# plt.title('Weather Report', fontsize = 20)
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
# fig = plot.get_figure()  
fig.set_tight_layout(True)      
fig.savefig(f"{file_name.replace('.', '_')}-chrf-bleu.png", format='png')
