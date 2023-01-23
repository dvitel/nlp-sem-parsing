import matplotlib.pyplot as plt
import csv
import sys
import os 

file_name = sys.argv[1]
y_name = sys.argv[2]
y_title = sys.argv[3] if len(sys.argv) >= 4 else ''

x = []
y = []

with open(file_name,'r') as csvfile:
    lines = csv.DictReader(csvfile, delimiter=',')
    for row in lines:
        x.append(float(row['down_level']))
        y.append(float(row[y_name]))

x, y = zip(*sorted(zip(x, y), key=lambda x: x[0]))
plot = plt.plot(x, y, color = 'g', marker = 'o', label = y_name)[0]

plt.xticks(rotation = 25)
plt.axvline(x = 0, color = 'tab:gray', linestyle = 'dashed', label = 'GE enabled')
plt.axvline(x = 1, color = 'tab:gray', linestyle = 'dashed', label = 'GE disabled')
plt.xlabel('Down level')
plt.ylabel(y_title)
# plt.title('Weather Report', fontsize = 20)
# legend = plot.legend("Exact match accuracy", fontsize=16)
# plot.xaxis.label.set_fontsize(18)
# plot.yaxis.label.set_fontsize(18)
fig = plot.get_figure()  
fig.set_tight_layout(True)      
fig.savefig(f"{file_name.replace('.', '_')}-{y_name}.png", format='png')
