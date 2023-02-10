import matplotlib
import matplotlib.pyplot as plt
from matplotlib.type1font import Type1Font
from matplotlib import font_manager
import csv
import sys
import os
import pandas as pd 
from scipy.stats import ttest_ind
import numpy as np
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [ r"\usepackage{times}" ]
# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# [f for f in font_manager.findSystemFonts(fontpaths=None, fontext="afm") if "Nimbus" in f])
# font = Type1Font('/usr/share/fonts/X11/Type1/n021003l.pfb')
# font_manager.FontManager().addfont('/usr/share/fonts/X11/Type1/n021003l.pfb')
# [f for f in font_manager.FontManager().get_font_names() if "9" in f ]
# print(font)
matplotlib.rcParams['font.family'] = ['Times']


file_name = sys.argv[1]
y_name = sys.argv[2]
y_title = sys.argv[3] if len(sys.argv) >= 4 else ''

# with open(file_name,'r') as csvfile:
#     lines = csv.DictReader(csvfile, delimiter=',')
#     for row in lines:
#         x.append(float(row['down_level']))
#         y.append(float(row[y_name]))

plain_data = pd.read_csv(file_name)
if y_name in ['test_shorter', 'test_longer', 'test_samesize']:
    plain_data.loc[:,y_name] = plain_data[y_name + '_count'] / 66.0
data = plain_data.groupby("down_level")
print(data)
mean_data = data.mean()
std_data = data.std()
min_data = data.min()
max_data = data.max()
# print(mean_data[y_name], std_data[y_name])

# plot = plt.plot(data.index, data[y_name], color = 'g', marker = 'o', label = y_name)[0]
# plot = data.(column=y_name, by='down_level')

fig, ax = plt.subplots()

# mm = list(zip(min_data[y_name], max_data[y_name]))
print(mean_data[y_name], std_data[y_name])
# yerr = [mean_data[y_name] - min_data[y_name], max_data[y_name] - mean_data[y_name]]
yerr = std_data[y_name]
plt.axvline(x = 0, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed', label = 'GE enabled')
plt.axvline(x = 1, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed', label = 'GE disabled')
plt.axhline(y = 30, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed')
plt.axhline(y = 25, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed')

ax.errorbar(mean_data.index, mean_data[y_name] * 100, yerr=yerr * 100, marker='o', color='k', linewidth=1, capsize=6)
# ax.fill_between(mean_data.index, mean_data[y_name] - std_data[y_name], mean_data[y_name] + std_data[y_name], alpha=.5, linewidth=0)
# ax.plot(mean_data.index, mean_data[y_name] * 100, linewidth=0.5, marker='o', color="k")

#Welch ttest
ttest = ttest_ind(plain_data[plain_data['down_level'] == 0.0][y_name].to_numpy(), plain_data[plain_data['down_level'] == 1.0][y_name].to_numpy(), equal_var = False, alternative='greater')
print(ttest) #exact match Ttest_indResult(statistic=3.1980107453341544, pvalue=0.008130031236723555) - ge has statistically significant effect

# plt.xticks(rotation = 25)
plt.xlabel('Down level', size=14)
plt.ylabel(y_title, size=14)
ax.tick_params(axis='both', which='major', labelsize=14)
# plt.title('Weather Report', fontsize = 20)
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
# fig = plot.get_figure()  
fig.set_tight_layout(True)      
fig.savefig(f"{file_name.replace('.', '_')}-{y_name}.pdf", format='pdf')
