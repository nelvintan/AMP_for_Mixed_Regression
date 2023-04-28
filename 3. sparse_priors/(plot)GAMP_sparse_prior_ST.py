import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
from numpy import load
import seaborn as sns

'''
heatmap matrix for:
p1 = 0.6
'''

min_corr_matrix = load('data_0.6p1_ST.npy')

sns.set(font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
x_axis_labels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] # labels for x-axis
y_axis_labels = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] # labels for y-axis
ax = sns.heatmap(min_corr_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu", vmin=0, vmax=1)
plt.xlabel(r'$\delta$') # y-axis label with fontsize 15
plt.ylabel(r'$\epsilon$') # x-axis label with fontsize 15
plt.savefig("heatmap_0.6p1_ST.pdf", bbox_inches='tight')

print('min_corr_matrix p1=0.6:\n', min_corr_matrix)

'''
heatmap matrix for:
p1 = 0.7
'''

min_corr_matrix = load('data_0.7p1_ST.npy')

plt.clf()
sns.set(font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
x_axis_labels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] # labels for x-axis
y_axis_labels = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01] # labels for y-axis
ax = sns.heatmap(min_corr_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap="YlGnBu", vmin=0, vmax=1)
plt.xlabel(r'$\delta$') # y-axis label with fontsize 15
plt.ylabel(r'$\epsilon$') # x-axis label with fontsize 15
plt.savefig("heatmap_0.7p1_ST.pdf", bbox_inches='tight')

print('min_corr_matrix p1=0.7:\n', min_corr_matrix)