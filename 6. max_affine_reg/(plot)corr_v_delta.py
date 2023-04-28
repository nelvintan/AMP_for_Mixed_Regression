import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_others(p, n_list):

  output_list = load('diff_mean_diff_inter_sig01.npy') # to change
  AM_output_list, GAMP_output_list, EMGAMP_output_list = output_list

  mean_corr1_list_AM = AM_output_list[0]
  mean_corr2_list_AM = AM_output_list[1]
  SD_corr1_list_AM = AM_output_list[2]
  SD_corr2_list_AM = AM_output_list[3]

  mean_corr1_list_GAMP = GAMP_output_list[0]
  mean_corr2_list_GAMP = GAMP_output_list[1]
  SD_corr1_list_GAMP = GAMP_output_list[2]
  SD_corr2_list_GAMP = GAMP_output_list[3]

  mean_corr1_list_EMGAMP = EMGAMP_output_list[0]
  mean_corr2_list_EMGAMP = EMGAMP_output_list[1]
  SD_corr1_list_EMGAMP = EMGAMP_output_list[2]
  SD_corr2_list_EMGAMP = EMGAMP_output_list[3]
  
  # plotting beta1 sq norm correlation vs delta
  size = len(mean_corr1_list_AM)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_corr1_list_AM, yerr=SD_corr1_list_AM, marker='v', color='red', ecolor='red', elinewidth=1, capsize=7, label="AM")
  plt.errorbar(delta_list, mean_corr1_list_GAMP, yerr=SD_corr1_list_GAMP, marker='o', color='blue', ecolor='blue', elinewidth=1, capsize=7, label="OR-AMP")
  plt.errorbar(delta_list, mean_corr1_list_EMGAMP, yerr=SD_corr1_list_EMGAMP, marker='x', color='green', ecolor='green', elinewidth=1, capsize=7, label="EM-AMP")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('diffmean_diffinter_sig01_beta1.pdf', bbox_inches='tight') # to change
  plt.show()

  # plotting beta2 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr2_list_AM, yerr=SD_corr2_list_AM, marker='v', color='red', ecolor='red', elinewidth=1, capsize=7, label="AM")
  plt.errorbar(delta_list, mean_corr2_list_GAMP, yerr=SD_corr2_list_GAMP, marker='o', color='blue', ecolor='blue', elinewidth=1, capsize=7, label="OR-AMP")
  plt.errorbar(delta_list, mean_corr2_list_EMGAMP, yerr=SD_corr2_list_EMGAMP, marker='x', color='green', ecolor='green', elinewidth=1, capsize=7, label="EM-AMP")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  # plt.legend(loc=(0.05,0.25), fontsize=16) # only for diff mean, diff intercpt
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('diffmean_diffinter_sig01_beta2.pdf', bbox_inches='tight') # to change
  plt.show()

  return

p = 500
n_list = [int(0.5*p), int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
plot_GAMP_v_others(p, n_list)