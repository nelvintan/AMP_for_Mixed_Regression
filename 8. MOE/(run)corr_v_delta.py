import math

import numpy as np
from numpy import save
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import multinomial
from numpy.random import uniform

from scipy.stats import norm
from scipy.stats import multivariate_normal as multivariate_normal_sp
from scipy.linalg import eigh

# Copied this function over from scipy library
def _eigvalsh_to_eps(spectrum, cond=None, rcond=None):
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = spectrum.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
    eps = cond * np.max(abs(spectrum))
    return eps

# Copied this function over from scipy library
def is_pos_semi_def_scipy(matrix):
  s, u = eigh(matrix)
  eps = _eigvalsh_to_eps(s)
  if np.min(s) < -eps:
    print('the input matrix must be positive semidefinite')
    return False
  else:
    return True

def generate_Sigma_0(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov):
  
  Sigma_0 = np.zeros((8, 8))

  # Fill in all the diagonals.
  Sigma_0[0, 0] = B_bar_cov[0, 0] + B_bar_mean[0]**2
  Sigma_0[1, 1] = B_bar_cov[1, 1] + B_bar_mean[1]**2
  Sigma_0[2, 2] = B_bar_cov[2, 2] + B_bar_mean[2]**2
  Sigma_0[3, 3] = B_bar_cov[3, 3] + B_bar_mean[3]**2
  Sigma_0[4, 4] = B_hat_0_row_cov[0, 0] + B_hat_0_row_mean[0]**2
  Sigma_0[5, 5] = B_hat_0_row_cov[1, 1] + B_hat_0_row_mean[1]**2
  Sigma_0[6, 6] = B_hat_0_row_cov[2, 2] + B_hat_0_row_mean[2]**2
  Sigma_0[7, 7] = B_hat_0_row_cov[3, 3] + B_hat_0_row_mean[3]**2

  # Fill in all the off-diagonals (the upper triangle).
  # (We do this row by row below.)
  Sigma_0[0, 1] = B_bar_cov[0, 1] + B_bar_mean[0] * B_bar_mean[1]
  Sigma_0[0, 2] = B_bar_cov[0, 2] + B_bar_mean[0] * B_bar_mean[2]
  Sigma_0[0, 3] = B_bar_cov[0, 3] + B_bar_mean[0] * B_bar_mean[3]
  Sigma_0[0, 4] = B_bar_mean[0] * B_hat_0_row_mean[0]
  Sigma_0[0, 5] = B_bar_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[0, 6] = B_bar_mean[0] * B_hat_0_row_mean[2]
  Sigma_0[0, 7] = B_bar_mean[0] * B_hat_0_row_mean[3]

  Sigma_0[1, 2] = B_bar_cov[1, 2] + B_bar_mean[1] * B_bar_mean[2]
  Sigma_0[1, 3] = B_bar_cov[1, 3] + B_bar_mean[1] * B_bar_mean[3]
  Sigma_0[1, 4] = B_bar_mean[1] * B_hat_0_row_mean[0]
  Sigma_0[1, 5] = B_bar_mean[1] * B_hat_0_row_mean[1]
  Sigma_0[1, 6] = B_bar_mean[1] * B_hat_0_row_mean[2]
  Sigma_0[1, 6] = B_bar_mean[1] * B_hat_0_row_mean[3]

  Sigma_0[2, 3] = B_bar_cov[2, 3] + B_bar_mean[2] * B_bar_mean[3]
  Sigma_0[2, 4] = B_bar_mean[2] * B_hat_0_row_mean[0]
  Sigma_0[2, 5] = B_bar_mean[2] * B_hat_0_row_mean[1]
  Sigma_0[2, 6] = B_bar_mean[2] * B_hat_0_row_mean[2]
  Sigma_0[2, 7] = B_bar_mean[2] * B_hat_0_row_mean[3]

  Sigma_0[3, 4] = B_bar_mean[3] * B_hat_0_row_mean[0]
  Sigma_0[3, 5] = B_bar_mean[3] * B_hat_0_row_mean[1]
  Sigma_0[3, 6] = B_bar_mean[3] * B_hat_0_row_mean[2]
  Sigma_0[3, 7] = B_bar_mean[3] * B_hat_0_row_mean[3]

  Sigma_0[4, 5] = B_hat_0_row_cov[0, 1] + B_hat_0_row_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[4, 6] = B_hat_0_row_cov[0, 2] + B_hat_0_row_mean[0] * B_hat_0_row_mean[2]
  Sigma_0[4, 7] = B_hat_0_row_cov[0, 3] + B_hat_0_row_mean[0] * B_hat_0_row_mean[3]

  Sigma_0[5, 6] = B_hat_0_row_cov[1, 2] + B_hat_0_row_mean[1] * B_hat_0_row_mean[2]
  Sigma_0[5, 7] = B_hat_0_row_cov[1, 3] + B_hat_0_row_mean[1] * B_hat_0_row_mean[3]

  Sigma_0[6, 7] = B_hat_0_row_cov[2, 3] + B_hat_0_row_mean[2] * B_hat_0_row_mean[3]

  # Fill in all the off-diagonals (the lower triangle).

  Sigma_0[1, 0] = Sigma_0[0, 1]

  Sigma_0[2, 0] = Sigma_0[0, 2]
  Sigma_0[2, 1] = Sigma_0[1, 2]

  Sigma_0[3, 0] = Sigma_0[0, 3]
  Sigma_0[3, 1] = Sigma_0[1, 3]
  Sigma_0[3, 2] = Sigma_0[2, 3]

  Sigma_0[5, 4] = Sigma_0[4, 5]
  
  Sigma_0[6, 4] = Sigma_0[4, 6]
  Sigma_0[6, 5] = Sigma_0[5, 6]

  Sigma_0[7, 4] = Sigma_0[4, 7]
  Sigma_0[7, 5] = Sigma_0[5, 7]
  Sigma_0[7, 6] = Sigma_0[6, 7]

  Sigma_0[4:, :4] = Sigma_0[:4, 4:].T

  return Sigma_0 / delta

'''
Our GAMP functions below -- note that the inputs Z_k and Y_bar will be exchanged
for Theta^k_i and Y_i in our matrix-GAMP algorithm.
'''

def Var_Z_given_Zk(Sigma_k):
  return Sigma_k[0:4, 0:4] - np.dot(np.dot(Sigma_k[0:4, 4:8], linalg.pinv(Sigma_k[4:8, 4:8])), Sigma_k[4:8, 0:4])

def E_Z_given_Zk(Sigma_k, Z_k):
  return np.dot(np.dot(Sigma_k[0:4, 4:8], linalg.pinv(Sigma_k[4:8, 4:8])), Z_k)

def pdf_Y_bar_given_Z(Z, Y_bar, sigma):
  # Note: function requires sigma > 0.
  
  Z1 = Z[0]
  Z2 = Z[1]
  Z3 = Z[2]
  Z4 = Z[3]

  prob = np.exp(Z3) / (np.exp(Z3) + np.exp(Z4))
  output = prob * norm.pdf((Y_bar-Z1)/sigma) + (1 - prob) * norm.pdf((Y_bar-Z2)/sigma)

  return output

def E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, sigma):
  # NOTE: c1, c2 are the intercepts of max-affine reg.

  Sigma_11 = Sigma_k[:4,:4] 
  Sigma_12 = Sigma_k[:4,4:] 
  Sigma_21 = Sigma_k[4:,:4] 
  Sigma_22 = Sigma_k[4:,4:] 
  mean_Z_given_Zk = np.dot(Sigma_12, np.dot(linalg.pinv(Sigma_22), Z_k)) 
  cov_Z_given_Zk = Sigma_11 - np.dot(Sigma_12, np.dot(linalg.pinv(Sigma_22), Sigma_21))
  num_samples = 200
  Z_samples = multivariate_normal(mean_Z_given_Zk, cov_Z_given_Zk, num_samples)

  num_data = np.zeros((num_samples, 4))
  denom_sum = 0
  denom_count = 0
  for i in range(num_samples):
    Z_sample = Z_samples[i]
    p_Zk = multivariate_normal_sp.pdf(Z_k, mean=np.array([0,0,0,0]), cov=Sigma_22, allow_singular=True)
    p_Ybar_given_Z_Zk = pdf_Y_bar_given_Z(Z_sample, Y_bar, sigma)
    num_data[i] = Z_sample * p_Zk * p_Ybar_given_Z_Zk
    denom_sum += p_Zk * p_Ybar_given_Z_Zk
    denom_count += 1
  
  numerator = np.mean(num_data, axis=0)
  denominator = denom_sum / denom_count

  output = numerator / denominator

  return output

def g_k_bayes(Z_k, Y_bar, Sigma_k, sigma): 
  mat1 = Var_Z_given_Zk(Sigma_k)
  vec2 = E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, sigma)
  vec3 = E_Z_given_Zk(Sigma_k, Z_k)
  
  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_Y_bar, Sigma_k, sigma):
  Z_k = Z_k_and_Y_bar[:4]
  Y_bar = Z_k_and_Y_bar[4:]

  return g_k_bayes(Z_k, Y_bar, Sigma_k, sigma)

def f_k_bayes(B_bar_k, M_k_B, T_k_B, B_bar_mean, B_bar_cov):
  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  part2 = B_bar_k - np.dot(M_k_B, B_bar_mean)
  output = B_bar_mean + np.dot(np.dot(B_bar_cov, M_k_B.T), np.dot(part1, part2))

  return output

def compute_C_k(Theta_k, R_hat_k, Sigma_k):
  n = len(Theta_k)
  part1 = np.dot(Theta_k.T, R_hat_k)/n
  part2 = np.dot(Sigma_k[4:8,0:4], np.dot(R_hat_k.T, R_hat_k)/n)
  output = np.dot(linalg.pinv(Sigma_k[4:8,4:8]), part1 - part2)
  
  return output.T

# This only holds for jointly Gaussian priors.
def f_k_prime(M_k_B, T_k_B, B_bar_cov):
  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  output = np.dot(part1, np.dot(M_k_B, B_bar_cov))
  return output

def SE_norm_sq_corr(B_bar_mean, B_bar_cov, M_k_B, num_MC_samples):
  
  B_bar_samples = multivariate_normal(B_bar_mean, B_bar_cov, num_MC_samples)
  G_k_B_samples = multivariate_normal(np.zeros(4), M_k_B, num_MC_samples)
  
  # The four parts of the normalized sq correlation.
  E_B_bar1_sq = B_bar_cov[0,0] + B_bar_mean[0]**2
  E_f1_B_bar1 = 0
  E_f1_sq = 0
  E_B_bar2_sq = B_bar_cov[1,1] + B_bar_mean[1]**2
  E_f2_B_bar2 = 0
  E_f2_sq = 0
  E_B_bar3_sq = B_bar_cov[2,2] + B_bar_mean[2]**2
  E_f3_B_bar3 = 0
  E_f3_sq = 0
  E_B_bar4_sq = B_bar_cov[3,3] + B_bar_mean[3]**2
  E_f4_B_bar4 = 0
  E_f4_sq = 0
  for i in range(num_MC_samples):
    B_bar_sample = B_bar_samples[i]
    G_k_B_sample = G_k_B_samples[i]
    T_k_B = M_k_B
    s = np.dot(M_k_B, B_bar_sample) + G_k_B_sample
    f = f_k_bayes(s, M_k_B, T_k_B, B_bar_mean, B_bar_cov)
    E_f1_B_bar1 += f[0]*B_bar_sample[0]
    E_f1_sq += f[0]**2

    E_f2_B_bar2 += f[1]*B_bar_sample[1]
    E_f2_sq += f[1]**2
    
    E_f3_B_bar3 += f[2]*B_bar_sample[2]
    E_f3_sq += f[2]**2
    
    E_f4_B_bar4 += f[3]*B_bar_sample[3]
    E_f4_sq += f[3]**2
  E_f1_B_bar1 /= num_MC_samples
  E_f1_sq /= num_MC_samples
  E_f2_B_bar2 /= num_MC_samples
  E_f2_sq /= num_MC_samples
  E_f3_B_bar3 /= num_MC_samples
  E_f3_sq /= num_MC_samples
  E_f4_B_bar4 /= num_MC_samples
  E_f4_sq /= num_MC_samples

  SE_norm_sq_corr1 = (E_f1_B_bar1**2) / (E_f1_sq * E_B_bar1_sq)
  SE_norm_sq_corr2 = (E_f2_B_bar2**2) / (E_f2_sq * E_B_bar2_sq)
  SE_norm_sq_corr3 = (E_f3_B_bar3**2) / (E_f3_sq * E_B_bar3_sq)
  SE_norm_sq_corr4 = (E_f4_B_bar4**2) / (E_f4_sq * E_B_bar4_sq)

  return SE_norm_sq_corr1, SE_norm_sq_corr2, SE_norm_sq_corr3, SE_norm_sq_corr4


def norm_sq_corr(beta, beta_hat):
  num = np.square(np.dot(beta, beta_hat))
  denom = np.square(linalg.norm(beta)) * np.square(linalg.norm(beta_hat))

  return num / denom

def get_SD(var_corr_list, mean_corr_list, succ_run_list):
  
  num_iter = len(mean_corr_list)
  num_runs = len(var_corr_list)

  SD_list = np.zeros(num_iter)
  for iter in range(num_iter):
    var = 0
    for run in range(num_runs):
      corr = var_corr_list[run][iter]
      if corr > 0:
        var += (corr - mean_corr_list[iter])**2
    var = var / succ_run_list[iter]
    SD_list[iter] = np.sqrt(var)

  return SD_list

def run_matrix_GAMP(n, p, sigma, X, Y, B, B_bar_mean, B_bar_cov, B_hat_0, num_iter):

  delta = n / p
  B_hat_0_row_mean = B_bar_mean
  B_hat_0_row_cov = B_bar_cov

  # Matrix-GAMP initializations
  R_hat_minus_1 = np.zeros((n,4))
  F_0 = np.eye(4)

  Sigma_0 = generate_Sigma_0(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov)
  print('Sigma_0\n',Sigma_0)

  # Storage of GAMP variables from previous iteration
  Theta_k = np.zeros((n,4))
  R_hat_k_minus_1 = R_hat_minus_1
  B_hat_k = B_hat_0
  F_k = F_0

  # State evolution parameters
  M_k_B = np.zeros((4,4))
  T_k_B = M_k_B
  Sigma_k = Sigma_0

  # Storage of the estimate B_hat
  B_hat_storage = []
  B_hat_storage.append(B_hat_0)

  # Storage of the state evolution param M_k_B
  M_k_B_storage = []

  for k in range(num_iter):
    print("=== Running iteration: " + str(k+1) + " ===")
    
    # Computing Theta_k
    Theta_k = np.dot(X, B_hat_k) - np.dot(R_hat_k_minus_1, F_k.T)

    # Computing R_hat_k
    Theta_k_and_Y = np.concatenate((Theta_k,Y[:,None]), axis=1)

    R_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, Theta_k_and_Y, Sigma_k, sigma)

    if (np.isnan(R_hat_k).any()):
      print('=== EARLY STOPPAGE ===')
      break
    
    # Computing C_k
    C_k = compute_C_k(Theta_k, R_hat_k, Sigma_k)
    
    # Computing B_k_plus_1
    B_k_plus_1 = np.dot(X.T, R_hat_k) - np.dot(B_hat_k, C_k.T)

    # Computing state evolution for the (k+1)th iteration
    M_k_plus_1_B = np.dot(R_hat_k.T, R_hat_k) / n
    T_k_plus_1_B = M_k_plus_1_B
    
    # Computing B_hat_k_plus_1
    B_hat_k_plus_1 = np.apply_along_axis(f_k_bayes, 1, B_k_plus_1, M_k_plus_1_B, T_k_plus_1_B, B_bar_mean, B_bar_cov)

    # Computing F_k_plus_1
    F_k_plus_1 = (p / n) * f_k_prime(M_k_plus_1_B, T_k_plus_1_B, B_bar_cov)

    # Computing state evolution for the (k+1)th iteration
    Sigma_k_plus_1 = np.zeros((8,8))
    Sigma_k_plus_1[0:4,0:4] = Sigma_k[0:4,0:4]
    temp_matrix = np.dot(B_hat_k_plus_1.T, B_hat_k_plus_1) / p
    Sigma_k_plus_1[0:4,4:8] = temp_matrix / delta
    Sigma_k_plus_1[4:8,0:4] = temp_matrix / delta
    Sigma_k_plus_1[4:8,4:8] = temp_matrix / delta

    # Updating parameters and storing B_hat_k_plus_1 & M_k_plus_1_B
    B_hat_storage.append(B_hat_k_plus_1)
    R_hat_k_minus_1 = R_hat_k
    B_hat_k = B_hat_k_plus_1
    F_k = F_k_plus_1
    M_k_B_storage.append(M_k_plus_1_B)
    M_k_B = M_k_plus_1_B
    T_k_B = T_k_plus_1_B
    Sigma_k = Sigma_k_plus_1

    print('M_k_B\n',M_k_B) # Under bayes-optimal setting, M_k_B = T_k_B
    print('Sigma_k:\n',Sigma_k)

  return B_hat_storage, M_k_B_storage

def run_multi_delta(p, n_list, sigma, num_iter, num_runs, num_MC_samples):
  
  num_deltas = len(n_list)

  mean_corr1_list_GAMP = np.zeros(num_deltas)
  mean_corr2_list_GAMP = np.zeros(num_deltas)
  mean_corr3_list_GAMP = np.zeros(num_deltas)
  mean_corr4_list_GAMP = np.zeros(num_deltas)
  var_corr1_list_GAMP = np.zeros((num_runs, num_deltas))
  var_corr2_list_GAMP = np.zeros((num_runs, num_deltas))
  var_corr3_list_GAMP = np.zeros((num_runs, num_deltas))
  var_corr4_list_GAMP = np.zeros((num_runs, num_deltas))

  mean_corr1_list_SE = np.zeros(num_deltas)
  mean_corr2_list_SE = np.zeros(num_deltas)
  mean_corr3_list_SE = np.zeros(num_deltas)
  mean_corr4_list_SE = np.zeros(num_deltas)

  for n_index in range(len(n_list)):
    n = n_list[n_index]
    final_corr1 = 0
    final_corr2 = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible

      B_bar_mean = np.array([1, 2, 3, 4])
      B_bar_cov = np.eye(4)
      B = multivariate_normal(B_bar_mean, B_bar_cov, p)
      beta1 = B[:, 0]
      beta2 = B[:, 1]
      gate1 = B[:, 2]
      gate2 = B[:, 3]

      B_hat_0_row_mean = B_bar_mean
      B_hat_0_row_cov = B_bar_cov
      B_hat_0 = multivariate_normal(B_hat_0_row_mean, B_hat_0_row_cov, p)

      u_vec = uniform(0, 1, n)
      eps_vec = normal(0, sigma, n)
      X = normal(0, np.sqrt(1/n), (n, p))
      Y = np.zeros(n)
      for i in range(n):
        u_i = u_vec[i]
        eps_i = eps_vec[i]
        X_i = X[i]
        prob = np.exp(np.inner(X_i, gate1)) / (np.exp(np.inner(X_i, gate1)) + np.exp(np.inner(X_i, gate2)))
        if u_i <= prob:
          Y[i] = np.inner(X_i, beta1) + eps_i
        else:
          Y[i] = np.inner(X_i, beta2) + eps_i

      B_hat_storage, M_k_B_storage = run_matrix_GAMP(n, p, sigma, X, Y, B, B_bar_mean, B_bar_cov, B_hat_0, num_iter)
      M_k_B = M_k_B_storage[-1]
      SE_norm_sq_corr1, SE_norm_sq_corr2, SE_norm_sq_corr3, SE_norm_sq_corr4 = SE_norm_sq_corr(B_bar_mean, B_bar_cov, M_k_B, num_MC_samples)

      # For GAMP.
      B_hat_GAMP = B_hat_storage[-1]
      beta1_GAMP = B_hat_GAMP[:, 0]
      beta2_GAMP = B_hat_GAMP[:, 1]
      gate1_GAMP = B_hat_GAMP[:, 2]
      gate2_GAMP = B_hat_GAMP[:, 3]

      norm_sq_corr1_GAMP = norm_sq_corr(beta1, beta1_GAMP)
      mean_corr1_list_GAMP[n_index] += norm_sq_corr1_GAMP
      var_corr1_list_GAMP[run_num][n_index] = norm_sq_corr1_GAMP

      norm_sq_corr2_GAMP = norm_sq_corr(beta2, beta2_GAMP)
      mean_corr2_list_GAMP[n_index] += norm_sq_corr2_GAMP
      var_corr2_list_GAMP[run_num][n_index] = norm_sq_corr2_GAMP

      norm_sq_corr3_GAMP = norm_sq_corr(gate1, gate1_GAMP)
      mean_corr3_list_GAMP[n_index] += norm_sq_corr3_GAMP
      var_corr3_list_GAMP[run_num][n_index] = norm_sq_corr3_GAMP

      norm_sq_corr4_GAMP = norm_sq_corr(gate2, gate2_GAMP)
      mean_corr4_list_GAMP[n_index] += norm_sq_corr4_GAMP
      var_corr4_list_GAMP[run_num][n_index] = norm_sq_corr4_GAMP

      mean_corr1_list_SE[n_index] += SE_norm_sq_corr1
      mean_corr2_list_SE[n_index] += SE_norm_sq_corr2
      mean_corr3_list_SE[n_index] += SE_norm_sq_corr3
      mean_corr4_list_SE[n_index] += SE_norm_sq_corr4

  mean_corr1_list_GAMP = mean_corr1_list_GAMP / num_runs
  mean_corr2_list_GAMP = mean_corr2_list_GAMP / num_runs
  mean_corr3_list_GAMP = mean_corr3_list_GAMP / num_runs
  mean_corr4_list_GAMP = mean_corr4_list_GAMP / num_runs

  SD_corr1_list_GAMP = np.sqrt(np.sum(np.square(var_corr1_list_GAMP - mean_corr1_list_GAMP), axis=0) / num_runs)
  SD_corr2_list_GAMP = np.sqrt(np.sum(np.square(var_corr2_list_GAMP - mean_corr2_list_GAMP), axis=0) / num_runs)
  SD_corr3_list_GAMP = np.sqrt(np.sum(np.square(var_corr2_list_GAMP - mean_corr2_list_GAMP), axis=0) / num_runs)
  SD_corr4_list_GAMP = np.sqrt(np.sum(np.square(var_corr2_list_GAMP - mean_corr2_list_GAMP), axis=0) / num_runs)

  mean_corr1_list_SE = mean_corr1_list_SE / num_runs
  mean_corr2_list_SE = mean_corr2_list_SE / num_runs
  mean_corr3_list_SE = mean_corr3_list_SE / num_runs
  mean_corr4_list_SE = mean_corr4_list_SE / num_runs

  GAMP_output_list = [mean_corr1_list_GAMP, mean_corr2_list_GAMP, mean_corr3_list_GAMP, mean_corr4_list_GAMP, 
                      SD_corr1_list_GAMP, SD_corr2_list_GAMP, SD_corr3_list_GAMP, SD_corr4_list_GAMP,
                      mean_corr1_list_SE, mean_corr2_list_SE, mean_corr3_list_SE, mean_corr4_list_SE]

  return GAMP_output_list

p = 500
n_list = [int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
sigma = 0.1
num_iter = 5
num_runs = 5
num_MC_samples = 1000

output_list = run_multi_delta(p, n_list, sigma, num_iter, num_runs, num_MC_samples)
save('GAMP_corr_v_delta_1234_sig01', np.array(output_list))