import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_GAMP_v_others(p, n_list):

  output_list = load('GAMP_v_others.npy')
  Spec_output_list, EM_output_list, AM_output_list, GAMP_output_list = output_list

  mean_corr1_list_spec, mean_corr2_list_spec, mean_corr3_list_spec, SD_corr1_list_spec, SD_corr2_list_spec, SD_corr3_list_spec = Spec_output_list
  mean_corr1_list_EM, mean_corr2_list_EM, mean_corr3_list_EM, SD_corr1_list_EM, SD_corr2_list_EM, SD_corr3_list_EM = EM_output_list
  mean_corr1_list_AM, mean_corr2_list_AM, mean_corr3_list_AM, SD_corr1_list_AM, SD_corr2_list_AM, SD_corr3_list_AM = AM_output_list
  mean_corr1_list_GAMP, mean_corr2_list_GAMP, mean_corr3_list_GAMP, SD_corr1_list_GAMP, SD_corr2_list_GAMP, SD_corr3_list_GAMP = GAMP_output_list
  
  # plotting beta1 sq norm correlation vs delta
  size = len(mean_corr1_list_AM)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_corr1_list_spec, yerr=SD_corr1_list_spec, marker='v', color='black', ecolor='black', elinewidth=1, capsize=7, label="Spectral")
  plt.errorbar(delta_list, mean_corr1_list_EM, yerr=SD_corr1_list_EM, marker='s', color='green', ecolor='green', elinewidth=1, capsize=7, label="EM")
  plt.errorbar(delta_list, mean_corr1_list_AM, yerr=SD_corr1_list_AM, marker='x', color='red', ecolor='red', elinewidth=1, capsize=7, label="AM")
  plt.errorbar(delta_list, mean_corr1_list_GAMP, yerr=SD_corr1_list_GAMP, marker='o', color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc=(0.6, 0.4), fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('GAMP_v_others_beta1.pdf', bbox_inches='tight')
  plt.show()

  # plotting beta2 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr2_list_spec, yerr=SD_corr2_list_spec, marker='v', color='black', ecolor='black', elinewidth=1, capsize=7, label="Spectral")
  plt.errorbar(delta_list, mean_corr2_list_EM, yerr=SD_corr2_list_EM, marker='s', color='green', ecolor='green', elinewidth=1, capsize=7, label="EM")
  plt.errorbar(delta_list, mean_corr2_list_AM, yerr=SD_corr2_list_AM, marker='x', color='red', ecolor='red', elinewidth=1, capsize=7, label="AM")
  plt.errorbar(delta_list, mean_corr2_list_GAMP, yerr=SD_corr2_list_GAMP, marker='o', color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  # plt.legend(loc=(1.04,0), fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('GAMP_v_others_beta2.pdf', bbox_inches='tight')
  plt.show()

  # plotting beta3 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr3_list_spec, yerr=SD_corr3_list_spec, marker='v', color='black', ecolor='black', elinewidth=1, capsize=7, label="Spectral")
  plt.errorbar(delta_list, mean_corr3_list_EM, yerr=SD_corr3_list_EM, marker='s', color='green', ecolor='green', elinewidth=1, capsize=7, label="EM")
  plt.errorbar(delta_list, mean_corr3_list_AM, yerr=SD_corr3_list_AM, marker='x', color='red', ecolor='red', elinewidth=1, capsize=7, label="AM")
  plt.errorbar(delta_list, mean_corr3_list_GAMP, yerr=SD_corr3_list_GAMP, marker='o', color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  # plt.legend(loc=(1.04,0), fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('GAMP_v_others_beta3.pdf', bbox_inches='tight')
  plt.show()

  return

p = 500
n_list = [int(5*p), int(5.5*p), int(6*p), int(6.5*p), int(7*p), int(7.5*p), int(8*p), int(8.5*p), int(9*p)]
plot_GAMP_v_others(p, n_list)