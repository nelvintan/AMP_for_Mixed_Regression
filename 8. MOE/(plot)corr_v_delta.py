import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import numpy as np
from numpy import load

def plot_multi_delta(p, n_list):

  output_list = load('GAMP_corr_v_delta_4321_sig04.npy') # <= To change!

  mean_corr1_list_GAMP = output_list[0]
  mean_corr2_list_GAMP = output_list[1]
  mean_corr3_list_GAMP = output_list[2]
  mean_corr4_list_GAMP = output_list[3]
  SD_corr1_list_GAMP = output_list[4]
  SD_corr2_list_GAMP = output_list[5]
  SD_corr3_list_GAMP = output_list[6]
  SD_corr4_list_GAMP = output_list[7]
  mean_corr1_list_SE = output_list[8]
  mean_corr2_list_SE = output_list[9]
  mean_corr3_list_SE = output_list[10]
  mean_corr4_list_SE = output_list[11]
  
  # plotting beta1 sq norm correlation vs delta
  size = len(mean_corr1_list_GAMP)
  delta_list = np.array(n_list) / p
  plt.errorbar(delta_list, mean_corr1_list_GAMP, yerr=SD_corr1_list_GAMP, color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.plot(delta_list, mean_corr1_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label="SE")
  plt.xlabel(r"$\delta$", fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="upper left", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('multi_delta_beta1_4321_sig04.pdf', bbox_inches='tight') # <= To change!
  plt.show()

  # plotting beta2 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr2_list_GAMP, yerr=SD_corr2_list_GAMP, color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.plot(delta_list, mean_corr2_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label="SE")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="upper left", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('multi_delta_beta2_4321_sig04.pdf', bbox_inches='tight') # <= To change!
  plt.show()

  # plotting beta3 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr3_list_GAMP, yerr=SD_corr3_list_GAMP, color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.plot(delta_list, mean_corr3_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label="SE")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower left", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  ax = plt.gca()
  ax.set_ylim([0.8, 1.0])
  plt.savefig('multi_delta_gate1_4321_sig04.pdf', bbox_inches='tight') # <= To change!
  plt.show()

  # plotting beta4 sq norm correlation vs delta
  plt.clf()
  plt.errorbar(delta_list, mean_corr4_list_GAMP, yerr=SD_corr4_list_GAMP, color='blue', ecolor='blue', elinewidth=1, capsize=7, label="AMP")
  plt.plot(delta_list, mean_corr4_list_SE, linestyle='None', marker='o', mfc='none', color='blue', markersize=10, label="SE")
  plt.xlabel(r"$\delta$",fontsize=16)
  plt.ylabel("Correlation", fontsize=16)
  plt.legend(loc="lower left", fontsize=16)
  plt.grid(which='major', axis='both', zorder=-1.0)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  ax = plt.gca()
  ax.set_ylim([0.8, 1.0])
  plt.savefig('multi_delta_gate2_4321_sig04.pdf', bbox_inches='tight') # <= To change!
  plt.show()

  return

p = 500
n_list = [int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
plot_multi_delta(p, n_list)