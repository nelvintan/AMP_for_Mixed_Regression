import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
from numpy import load

def plot_GAMP_v_SE_multi_delta(p, n_list):

  output_list1 = load('output_list1_zero_mean.npy')

  mean_final_corr1_list1 = output_list1[0]
  mean_final_corr2_list1 = output_list1[1]
  mean_final_corr1_list_SE1 = output_list1[2]
  mean_final_corr2_list_SE1 = output_list1[3]
  SD_final_corr1_list1 = output_list1[4]
  SD_final_corr2_list1 = output_list1[5]
  SD_final_corr1_list_SE1 = output_list1[6]
  SD_final_corr2_list_SE1 = output_list1[7]

  output_list2 = load('output_list2_zero_mean.npy')

  mean_final_corr1_list2 = output_list2[0]
  mean_final_corr2_list2 = output_list2[1]
  mean_final_corr1_list_SE2 = output_list2[2]
  mean_final_corr2_list_SE2 = output_list2[3]
  SD_final_corr1_list2 = output_list2[4]
  SD_final_corr2_list2 = output_list2[5]
  SD_final_corr1_list_SE2 = output_list2[6]
  SD_final_corr2_list_SE2 = output_list2[7]

  output_list3 = load('output_list3_zero_mean.npy')

  mean_final_corr1_list3 = output_list3[0]
  mean_final_corr2_list3 = output_list3[1]
  mean_final_corr1_list_SE3 = output_list3[2]
  mean_final_corr2_list_SE3 = output_list3[3]
  SD_final_corr1_list3 = output_list3[4]
  SD_final_corr2_list3 = output_list3[5]
  SD_final_corr1_list_SE3 = output_list3[6]
  SD_final_corr2_list_SE3 = output_list3[7]
  
  # plotting beta1 sq norm correlation vs delta
  size = len(mean_final_corr1_list1)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_final_corr1_list1, yerr=SD_final_corr1_list1, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0$")
  plt.plot(delta_list, mean_final_corr1_list_SE1, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label=r"SE, $\sigma=0$")
  plt.errorbar(delta_list, mean_final_corr1_list2, yerr=SD_final_corr1_list2, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.2$")
  plt.plot(delta_list, mean_final_corr1_list_SE2, linestyle='None', marker='v', mfc='none', color='red', markersize=10, label=r"SE, $\sigma=0.2$")
  plt.errorbar(delta_list, mean_final_corr1_list3, yerr=SD_final_corr1_list3, color='green', ecolor='green', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.4$")
  plt.plot(delta_list, mean_final_corr1_list_SE3, linestyle='None', marker='s', mfc='none', color='green', markersize=10, label=r"SE, $\sigma=0.4$")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.legend(loc="lower right", fontsize=13)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig("corr_vs_delta_beta1_multi_sigma_0prior.pdf", bbox_inches='tight')
  plt.show()

  # plotting beta2 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_final_corr2_list1, yerr=SD_final_corr2_list1, color='blue', ecolor='blue', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0$")
  plt.plot(delta_list, mean_final_corr2_list_SE1, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label=r"SE, $\sigma=0$")
  plt.errorbar(delta_list, mean_final_corr2_list2, yerr=SD_final_corr2_list2, color='red', ecolor='red', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.2$")
  plt.plot(delta_list, mean_final_corr2_list_SE2, linestyle='None', marker='v', mfc='none', color='red', markersize=10, label=r"SE, $\sigma=0.2$")
  plt.errorbar(delta_list, mean_final_corr2_list3, yerr=SD_final_corr2_list3, color='green', ecolor='green', elinewidth=3, capsize=10, label=r"AMP, $\sigma=0.4$")
  plt.plot(delta_list, mean_final_corr2_list_SE3, linestyle='None', marker='s', mfc='none', color='green', markersize=10, label=r"SE, $\sigma=0.4$")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.legend(loc="upper left", fontsize=13)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig("corr_vs_delta_beta2_multi_sigma_0prior.pdf", bbox_inches='tight')
  plt.show()
  
  return

p = 500
n_list = [int(0.5*p), int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
plot_GAMP_v_SE_multi_delta(p, n_list)