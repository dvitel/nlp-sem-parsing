import matplotlib
import matplotlib.pyplot as plt
import csv
import sys
import os 
# import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [ r"\usepackage{times}" ]
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']

plt.xticks(rotation = 25)

data = pd.read_csv('data1.csv')
data = data.sort_values(by = ["down_level"])
data = data.groupby('down_level').mean()
# xfill = np.round(np.arange(-1, 2 + 0.05, 0.05), 2)
# data = data.set_index('down_level') #.reindex(xfill).reset_index()
data['test_correct_not_matched'] = data.loc[:, 'test_correct_percent'] - data.loc[:, 'test_exact_match']
data['test_incomplete'] = data.loc[:, 'test_incomplete_progcount'] / 66
data['test_complete_wrong'] = 1.0 - data.loc[:, 'test_correct_percent'] - data.loc[:, 'test_incomplete']

data = data.rename(columns={'test_exact_match':'match', 'test_correct_not_matched':'correct', 'test_complete_wrong':'complete wrong', 'test_incomplete': 'incomplete'})
print(data)
data.loc[:, ['match', 'correct', 'complete wrong', 'incomplete']] *= 100
plot = data.plot.area(stacked=True, y=['match', 'correct', 'complete wrong', 'incomplete'], linewidth=0, xlim=(-0.2,1.2), ylim=(0, 100),
                        fontsize=14, color=('#C0C0C0', '#909090', '#606060', '#303030'))

# print(plot.patches)
# '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
# plot.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch='/'))
# plot.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch='-'))
# plot.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch='|'))
# plot.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch='*'))
plot.set_xlabel('Down level', fontsize=14)
plot.set_ylabel('Test set, \%', fontsize=14)
plot.legend(loc = 'upper left', framealpha = 0.75, fontsize=14)                

plot.axvline(x = 0, color = 'white', linewidth=0.5, linestyle = 'dashed', label = 'GE enabled')
plot.axvline(x = 1, color = 'white', linewidth=0.5, linestyle = 'dashed', label = 'GE disabled')
plot.tick_params(axis='both', which='major', labelsize=14)
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
fig = plot.get_figure()  
fig.set_tight_layout(True)      
fig.savefig(f"prog-quality1.pdf", format='pdf')
