import matplotlib.pyplot as plt
import csv
import sys
import os 
# import numpy as np
import pandas as pd

plt.xticks(rotation = 25)

data = pd.read_csv('data.csv')
data = data.sort_values(by = ["down_level"])
# xfill = np.round(np.arange(-1, 2 + 0.05, 0.05), 2)
# data = data.set_index('down_level') #.reindex(xfill).reset_index()
data['test_correct_not_matched'] = data.loc[:, 'test_correct_percent'] - data.loc[:, 'test_exact_match']
data['test_incomplete_or_wrong'] = 1.0 - data.loc[:, 'test_correct_percent'] - data.loc[:, 'test_unparse_type_errors_percent']

data = data.rename(columns={'test_exact_match':'match', 'test_correct_not_matched':'correct', 'test_unparse_type_errors_percent':'type err', 'test_incomplete_or_wrong': 'err'})
plot = data.plot.area(stacked=True, x='down_level', y=['match', 'correct', 'type err', 'err'], linewidth=0, xlim=(-1.,1.5), ylim=(0.0, 1.0),
                        title='Result program quality', xlabel='Down level', ylabel='Test set, %', color=('#83e6d5aa', '#5266ebaa', '#e681d3aa', '#d9414eaa'))
plot.legend(loc = 'upper right', framealpha = 0.75)                

plot.axvline(x = 0, color = 'tab:gray', linestyle = 'dashed', label = 'GE enabled')
plot.axvline(x = 1, color = 'tab:gray', linestyle = 'dashed', label = 'GE disabled')
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
fig = plot.get_figure()  
# fig.set_tight_layout(True)      
fig.savefig(f"prog-quality.png", format='png')
