import matplotlib.pyplot as plt
import csv
import sys
import os 
# import numpy as np
import pandas as pd
import matplotlib
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
data['test_match_count'] = data.loc[:, 'test_exact_match'] * 66
data['test_longer_complete'] = data.loc[:, 'test_longer_count'] - data.loc[:, 'test_incomplete_progcount']


data = data.rename(columns={'test_shorter_count':'shorter', 'test_match_count':'match', 'test_samesize_error_count':'samelen', 'test_longer_complete':'longer', 'test_incomplete_progcount':'incomplete'})
data.loc[:, ['shorter', 'match', 'samelen', 'longer', 'incomplete']] /= 66.0
print(data)
data.loc[:, ['shorter', 'match', 'samelen', 'longer', 'incomplete']] *= 100
plot = data.plot.area(stacked=True, y=['shorter', 'match', 'samelen', 'longer', 'incomplete'], linewidth=0, xlim=(-0.2,1.2), ylim=(0, 100),
                        color=('#808080', '#C0C0C0', '#A0A0A0', '#606060', '#303030'), fontsize=14)
plot.legend(loc = 'lower left', framealpha = 0.75, fontsize=14)                

plot.set_xlabel('Down level', fontsize=14)
plot.set_ylabel('Test set, \%', fontsize=14)
plot.axvline(x = 0, color = 'white', linewidth=0.5, linestyle = 'dashed', label = 'GE enabled')
plot.axvline(x = 1, color = 'white', linewidth=0.5, linestyle = 'dashed', label = 'GE disabled')
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
fig = plot.get_figure()  
fig.set_tight_layout(True)      
fig.savefig(f"prog-length1.pdf", format='pdf')
