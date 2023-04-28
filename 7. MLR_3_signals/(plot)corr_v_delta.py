import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_corr_v_delta(p, n_list):
  
  output_list = load('diff_mean_same_prop.npy')

  mean_final_corr1_list = output_list[0]
  mean_final_corr2_list = output_list[1]
  mean_final_corr3_list = output_list[2]
  mean_final_corr1_list_SE = output_list[3]
  mean_final_corr2_list_SE = output_list[4]
  mean_final_corr3_list_SE = output_list[5]
  SD_final_corr1_list = output_list[6]
  SD_final_corr2_list = output_list[7]
  SD_final_corr3_list = output_list[8]

  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr1_list)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr1_list, yerr=SD_final_corr1_list, color='blue', ecolor='blue', elinewidth=3, capsize=10, label='AMP')
  plt.plot(delta_list, mean_final_corr1_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label='SE')
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('diff_mean_same_prop_beta1.pdf', bbox_inches='tight')
  plt.show()

  # plotting beta2 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_final_corr2_list, yerr=SD_final_corr2_list, color='blue', ecolor='blue', elinewidth=3, capsize=10, label='AMP')
  plt.plot(delta_list, mean_final_corr2_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label='SE')
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('diff_mean_same_prop_beta2.pdf', bbox_inches='tight')
  plt.show()

  # plotting beta3 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_final_corr3_list, yerr=SD_final_corr3_list, color='blue', ecolor='blue', elinewidth=3, capsize=10, label='AMP')
  plt.plot(delta_list, mean_final_corr3_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label='SE')
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower right", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('diff_mean_same_prop_beta3.pdf', bbox_inches='tight')
  plt.show()

p = 500
n_list = [int(5*p), int(5.5*p), int(6*p), int(6.5*p), int(7*p), int(7.5*p), int(8*p), int(8.5*p), int(9*p)]
plot_corr_v_delta(p, n_list)