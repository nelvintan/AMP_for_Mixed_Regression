import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

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
    #print('the input matrix must be positive semidefinite')
    return False
  else:
    return True

def norm_pdf(x, mean, var):
  first_part = 1 / np.sqrt(2 * np.pi * var)
  second_part = np.exp((-1/2) * ((x - mean) ** 2 / var))
  return first_part * second_part

def multi_norm_pdf(x, mean, cov): # multivariate normal distribution
  dimension = len(x)
  first_part = 1 / np.sqrt((2 * np.pi) ** dimension * linalg.det(cov))
  second_part = np.exp((-1/2) * np.dot(np.dot((x - mean).T, linalg.pinv(cov)), (x - mean)))
  return first_part * second_part

def generate_Sigma_0(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov):
  
  Sigma_0 = np.zeros((4, 4))
  
  Sigma_0[0, 0] = B_bar_cov[0, 0] + B_bar_mean[0]**2
  Sigma_0[0, 1] = B_bar_cov[0, 1] + B_bar_mean[0] * B_bar_mean[1]
  Sigma_0[1, 0] = Sigma_0[0, 1]
  Sigma_0[1, 1] = B_bar_cov[1, 1] + B_bar_mean[1]**2

  Sigma_0[0, 2] = B_bar_mean[0] * B_hat_0_row_mean[0]
  Sigma_0[0, 3] = B_bar_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[1, 2] = B_bar_mean[1] * B_hat_0_row_mean[0]
  Sigma_0[1, 3] = B_bar_mean[1] * B_hat_0_row_mean[1]

  Sigma_0[2, 2] = B_hat_0_row_cov[0, 0] + B_hat_0_row_mean[0]**2
  Sigma_0[2, 3] = B_hat_0_row_cov[0, 1] + B_hat_0_row_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[3, 2] = Sigma_0[2, 3]
  Sigma_0[3, 3] = B_hat_0_row_cov[1, 1] + B_hat_0_row_mean[1]**2

  Sigma_0[2:, :2] = Sigma_0[:2, 2:].T

  return Sigma_0 / delta

'''
Our GAMP functions below -- note that the inputs Z_k and Y_bar will be exchanged
for Theta^k_i and Y_i in our matrix-GAMP algorithm.
'''

def Var_Z_given_Zk(Sigma_k):
  return Sigma_k[0:2, 0:2] - np.dot(np.dot(Sigma_k[0:2, 2:4], linalg.pinv(Sigma_k[2:4, 2:4])), Sigma_k[2:4, 0:2])

def E_Z_given_Zk(Sigma_k, Z_k):
  return np.dot(np.dot(Sigma_k[0:2, 2:4], linalg.pinv(Sigma_k[2:4, 2:4])), Z_k)

def E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, p1, sigma):

  Sigma_k1_Y = np.zeros((5, 5))
  Sigma_k1_Y[:4, :4] = Sigma_k
  Sigma_k1_Y[4, :4] = Sigma_k[0, :]
  Sigma_k1_Y[:4, 4] = Sigma_k[0, :]
  Sigma_k1_Y[4, 4] = Sigma_k[0, 0] + sigma**2
  
  Sigma_k0_Y = np.zeros((5, 5))
  Sigma_k0_Y[:4, :4] = Sigma_k
  Sigma_k0_Y[4, :4] = Sigma_k[1, :]
  Sigma_k0_Y[:4, 4] = Sigma_k[1, :]
  Sigma_k0_Y[4, 4] = Sigma_k[1, 1] + sigma**2
  
  E_Z_given_Zk_Ybar_cbar1 = np.dot(Sigma_k1_Y[:2, 2:], np.dot(linalg.pinv(Sigma_k1_Y[2:, 2:]), np.concatenate((Z_k, Y_bar))))
  E_Z_given_Zk_Ybar_cbar0 = np.dot(Sigma_k0_Y[:2, 2:], np.dot(linalg.pinv(Sigma_k0_Y[2:, 2:]), np.concatenate((Z_k, Y_bar))))
  
  mean = np.zeros(3)
  cov1 = Sigma_k1_Y[2:, 2:]
  cov2 = Sigma_k0_Y[2:, 2:]

  if is_pos_semi_def_scipy(cov1) == False or is_pos_semi_def_scipy(cov2) == False:
    return np.array([np.nan, np.nan])

  P_Zk_Ybar_given_cbar1 = multivariate_normal_sp.pdf(np.concatenate((Z_k, Y_bar)), mean=mean, cov=cov1, allow_singular=True)
  P_Zk_Ybar_given_cbar0 = multivariate_normal_sp.pdf(np.concatenate((Z_k, Y_bar)), mean=mean, cov=cov2, allow_singular=True)

  P_cbar1_given_Zk_Ybar = (p1*P_Zk_Ybar_given_cbar1) / (p1*P_Zk_Ybar_given_cbar1 + (1 - p1)*P_Zk_Ybar_given_cbar0)
  P_cbar0_given_Zk_Ybar = ((1 - p1)*P_Zk_Ybar_given_cbar0) / (p1*P_Zk_Ybar_given_cbar1 + (1 - p1)*P_Zk_Ybar_given_cbar0)
  
  output = P_cbar1_given_Zk_Ybar * E_Z_given_Zk_Ybar_cbar1 + P_cbar0_given_Zk_Ybar * E_Z_given_Zk_Ybar_cbar0

  return output

def g_k_bayes(Z_k, Y_bar, Sigma_k, p1, sigma):
  
  mat1 = Var_Z_given_Zk(Sigma_k)
  vec2 = E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, p1, sigma)
  vec3 = E_Z_given_Zk(Sigma_k, Z_k)
  
  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_Y_bar, Sigma_k, p1, sigma):
  Z_k = Z_k_and_Y_bar[:2]
  Y_bar = Z_k_and_Y_bar[2:]
  return g_k_bayes(Z_k, Y_bar, Sigma_k, p1, sigma)
  
def f_k_bayes(B_bar_k, M_k_B, T_k_B, B_bar_mean, B_bar_cov):

  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  part2 = B_bar_k - np.dot(M_k_B, B_bar_mean)
  output = B_bar_mean + np.dot(np.dot(B_bar_cov, M_k_B.T), np.dot(part1, part2))

  return output

def compute_C_k(Theta_k, R_hat_k, Sigma_k):
  n = len(Theta_k)
  part1 = np.dot(Theta_k.T, R_hat_k)/n
  part2 = np.dot(Sigma_k[2:4,0:2], np.dot(R_hat_k.T, R_hat_k)/n)
  output = np.dot(linalg.pinv(Sigma_k[2:4,2:4]), part1 - part2)
  return output.T

# This only holds for jointly Gaussian priors.
def f_k_prime(M_k_B, T_k_B, B_bar_cov):
  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  output = np.dot(part1, np.dot(M_k_B, B_bar_cov))
  return output

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

'''
Spectral Initialization
http://proceedings.mlr.press/v32/yia14.pdf
the scalings of X and beta1 and beta2 are different from the paper when 
i compared the algo, but i dont think it matters? 
'''

def loss(Y, X, beta1, beta2):
  output = 0
  for i in range(len(Y)):
    loss1 = (Y[i] - np.dot(X[i, :], beta1))**2
    loss2 = (Y[i] - np.dot(X[i, :], beta2))**2
    output += min(loss1, loss2)
  return output

def spec_init_grid_search(Y, X, n, p, grid_param):

  # Compute matrix M
  M = np.zeros((p, p))
  for i in range(n):
    X_i = X[i, :]
    M = M + ((Y[i]**2) * np.outer(X_i, X_i))
  M = M / n

  # # Compute top 2 eigenvector of M
  eigenvectors = linalg.eigh(M)[1]
  eigenvector1 = eigenvectors[:, -1]
  eigenvector2 = eigenvectors[:, -2]

  # Make the grid points
  G = []
  for t in range(int(np.ceil((2 * np.pi) / grid_param))):
    u = eigenvector1 * np.cos(grid_param * t) + eigenvector2 * np.sin(grid_param * t)
    G.append(u)

  # Pick the pair that has the lowest loss
  current_min = loss(Y, X, G[0], G[1])
  beta1_0 = G[0]
  beta2_0 = G[1]
  for u1 in G:
    for u2 in G:
      current_loss = loss(Y, X, u1, u2)
      if current_loss < current_min:
        current_min = current_loss
        beta1_0 = u1
        beta2_0 = u2

  return beta1_0, beta2_0


''' 
Expectation Maximization (EM) algorithm 
from: EM: https://arxiv.org/pdf/1905.12106.pdf
'''

def run_EM(n, p, p1, sigma, X, Y, B_hat_0, num_iter):
  
  beta1_k = B_hat_0[:, 0]
  beta2_k = B_hat_0[:, 1]

  B_hat_storage = []
  B_hat_storage.append(B_hat_0)
  
  pi = np.array([0.5, 0.5]) # probability of latent variable being either of the signals
  w = np.zeros((n, 2))
  for k in range(num_iter):
    # === E step ===
    for i in range(n):
      X_i = X[i, :]
      Y_i = Y[i]
      num = pi[0] * np.exp(-1 * ((Y_i - np.dot(X_i, beta1_k))**2 / 2))
      denom = pi[0] * np.exp(-1 * ((Y_i - np.dot(X_i, beta1_k))**2 / 2)) + pi[1] * np.exp(-1 * ((Y_i - np.dot(X_i, beta2_k))**2 / 2))
      w[i, 0] = num / denom
      num = pi[1] * np.exp(-1 * ((Y_i - np.dot(X_i, beta2_k))**2 / 2))
      denom = pi[0] * np.exp(-1 * ((Y_i - np.dot(X_i, beta1_k))**2 / 2)) + pi[1] * np.exp(-1 * ((Y_i - np.dot(X_i, beta2_k))**2 / 2))
      w[i, 1] = num / denom

    # === M step ===
    part1_1 = np.zeros((p, p))
    part1_2 = np.zeros((p, p))
    for i in range(n):
      X_i = X[i, :]
      part1_1 += w[i, 0] * np.outer(X_i, X_i)
      part1_2 += w[i, 1] * np.outer(X_i, X_i)
    part1_1 = linalg.inv(part1_1 / n)
    part1_2 = linalg.inv(part1_2 / n)
    part2_1 = np.zeros(p)
    part2_2 = np.zeros(p)
    for i in range(n):
      X_i = X[i, :]
      Y_i = Y[i]
      part2_1 += w[i, 0] * X_i * Y_i
      part2_2 += w[i, 1] * X_i * Y_i
    part2_1 = part2_1 / n
    part2_2 = part2_2 / n
    beta1_k = np.dot(part1_1, part2_1)
    beta2_k = np.dot(part1_2, part2_2)
    pi[0] = np.mean(w[:, 0])
    pi[1] = np.mean(w[:, 1])

    # Storing estimate at this iteration:
    B_hat_k = np.column_stack((beta1_k, beta2_k))
    B_hat_storage.append(B_hat_k)

  return B_hat_storage

'''
ALternating minimization (AM) algorithm
'''

def run_AM(n, p, p1, sigma, X, Y, B_hat_0, num_iter):

  delta = n / p
  
  beta1_k = B_hat_0[:, 0]
  beta2_k = B_hat_0[:, 1]

  B_hat_storage = []
  B_hat_storage.append(B_hat_0)
  
  for k in range(num_iter):
    # AM part I: Guess the labels
    J1 = []
    J2 = []
    for i in range(n):
      diff1 = np.abs(Y[i] - np.dot(X[i, :], beta1_k)) 
      diff2 = np.abs(Y[i] - np.dot(X[i, :], beta2_k))
      if diff1 < diff2:
        J1.append(i)
      else:
        J2.append(i)

    # AM part II: Solve least squares
    beta1_k = np.linalg.lstsq(np.take(X, J1, axis=0), Y[J1], rcond=None)[0]
    beta2_k = np.linalg.lstsq(np.take(X, J2, axis=0), Y[J2], rcond=None)[0]

    B_hat_k = np.column_stack((beta1_k, beta2_k))
    B_hat_storage.append(B_hat_k)

  return B_hat_storage

'''
GAMP
'''

def run_matrix_GAMP(n, p, p1, sigma, X, Y, B_bar_mean, B_bar_cov, 
                                    B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter):

  delta = n / p

  # Matrix-GAMP initializations
  R_hat_minus_1 = np.zeros((n,2))
  F_0 = np.eye(2)

  Sigma_0 = generate_Sigma_0(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov)
  print('Sigma_0\n',Sigma_0)

  # Storage of GAMP variables from previous iteration
  Theta_k = np.zeros((n,2))
  R_hat_k_minus_1 = R_hat_minus_1
  B_hat_k = B_hat_0
  F_k = F_0

  # State evolution parameters
  M_k_B = np.zeros((2,2))
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
    R_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, Theta_k_and_Y, Sigma_k, p1, sigma)

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
    Sigma_k_plus_1 = np.zeros((4,4))
    Sigma_k_plus_1[0:2,0:2] = Sigma_k[0:2,0:2]
    temp_matrix = np.dot(B_hat_k_plus_1.T, B_hat_k_plus_1) / p
    Sigma_k_plus_1[0:2,2:4] = temp_matrix / delta
    Sigma_k_plus_1[2:4,0:2] = temp_matrix / delta
    Sigma_k_plus_1[2:4,2:4] = temp_matrix / delta

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

  return [B_hat_storage, M_k_B_storage]

def compare_algo_multi_delta(p, n_list, p1, sigma, num_iter, num_runs):
  
  num_deltas = len(n_list)

  mean_corr1_list_spec = np.zeros(num_deltas)
  mean_corr2_list_spec = np.zeros(num_deltas)
  var_corr1_list_spec = np.zeros((num_runs, num_deltas))
  var_corr2_list_spec = np.zeros((num_runs, num_deltas))

  mean_corr1_list_EM = np.zeros(num_deltas)
  mean_corr2_list_EM = np.zeros(num_deltas)
  var_corr1_list_EM = np.zeros((num_runs, num_deltas))
  var_corr2_list_EM = np.zeros((num_runs, num_deltas))

  mean_corr1_list_AM = np.zeros(num_deltas)
  mean_corr2_list_AM = np.zeros(num_deltas)
  var_corr1_list_AM = np.zeros((num_runs, num_deltas))
  var_corr2_list_AM = np.zeros((num_runs, num_deltas))

  mean_corr1_list_GAMP = np.zeros(num_deltas)
  mean_corr2_list_GAMP = np.zeros(num_deltas)
  var_corr1_list_GAMP = np.zeros((num_runs, num_deltas))
  var_corr2_list_GAMP = np.zeros((num_runs, num_deltas))

  for n_index in range(len(n_list)):
    n = n_list[n_index]
    final_corr1 = 0
    final_corr2 = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible

      B_bar_mean = np.array([0, 0])
      B_bar_cov = np.eye(2)
      B = multivariate_normal(B_bar_mean, B_bar_cov, p)

      B_hat_0_row_mean = np.array([0, 0])
      B_hat_0_row_cov = np.eye(2)
      B_hat_0 = multivariate_normal(B_hat_0_row_mean, B_hat_0_row_cov, p)

      X = normal(0, np.sqrt(1/n), (n, p))
      Theta = np.dot(X, B)

      # Generating Y: We used ome numpy operational trick to avoid writing 
      # a for loop (inefficient) to compute Y.
      c = binomial(1, p1, n)
      eps = normal(0, sigma, n)
      c = c[:, None]
      Y = (Theta * np.c_[c, 1-c]).sum(1) + eps

      grid_param = 0.1
      B_hat_spec = spec_init_grid_search(Y, X, n, p, grid_param)
      B_hat_storage_EM = run_EM(n, p, p1, sigma, X, Y, B_hat_0, 1) #iter one because we have checked that it stops improving after first iter.
      B_hat_storage_AM = run_AM(n, p, p1, sigma, X, Y, B_hat_0, 1) #iter one because we have checked that it stops improving after first iter.
      B_hat_storage_GAMP = run_matrix_GAMP(n, p, p1, sigma, X, Y, B_bar_mean, B_bar_cov, 
                                    B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)[0]

      beta1 = B[:, 0]
      beta2 = B[:, 1]

      # For Spectral initialization.
      beta1_hat_spec = B_hat_spec[0]
      beta2_hat_spec = B_hat_spec[1]

      norm_sq_corr1_spec = norm_sq_corr(beta1, beta1_hat_spec)
      mean_corr1_list_spec[n_index] += norm_sq_corr1_spec
      var_corr1_list_spec[run_num][n_index] = norm_sq_corr1_spec

      norm_sq_corr2_spec = norm_sq_corr(beta2, beta2_hat_spec)
      mean_corr2_list_spec[n_index] += norm_sq_corr2_spec
      var_corr2_list_spec[run_num][n_index] = norm_sq_corr2_spec

      # For EM.
      B_hat_EM = B_hat_storage_EM[-1]
      beta1_hat_EM = B_hat_EM[:, 0]
      beta2_hat_EM = B_hat_EM[:, 1]

      norm_sq_corr1_EM = norm_sq_corr(beta1, beta1_hat_EM)
      mean_corr1_list_EM[n_index] += norm_sq_corr1_EM
      var_corr1_list_EM[run_num][n_index] = norm_sq_corr1_EM

      norm_sq_corr2_EM = norm_sq_corr(beta2, beta2_hat_EM)
      mean_corr2_list_EM[n_index] += norm_sq_corr2_EM
      var_corr2_list_EM[run_num][n_index] = norm_sq_corr2_EM

      # For AM.
      B_hat_AM = B_hat_storage_AM[-1]
      beta1_hat_AM = B_hat_AM[:, 0]
      beta2_hat_AM = B_hat_AM[:, 1]

      norm_sq_corr1_AM = norm_sq_corr(beta1, beta1_hat_AM)
      mean_corr1_list_AM[n_index] += norm_sq_corr1_AM
      var_corr1_list_AM[run_num][n_index] = norm_sq_corr1_AM

      norm_sq_corr2_AM = norm_sq_corr(beta2, beta2_hat_AM)
      mean_corr2_list_AM[n_index] += norm_sq_corr2_AM
      var_corr2_list_AM[run_num][n_index] = norm_sq_corr2_AM

      # For GAMP.
      B_hat_GAMP = B_hat_storage_GAMP[-1]
      beta1_hat_GAMP = B_hat_GAMP[:, 0]
      beta2_hat_GAMP = B_hat_GAMP[:, 1]

      norm_sq_corr1_GAMP = norm_sq_corr(beta1, beta1_hat_GAMP)
      mean_corr1_list_GAMP[n_index] += norm_sq_corr1_GAMP
      var_corr1_list_GAMP[run_num][n_index] = norm_sq_corr1_GAMP

      norm_sq_corr2_GAMP = norm_sq_corr(beta2, beta2_hat_GAMP)
      mean_corr2_list_GAMP[n_index] += norm_sq_corr2_GAMP
      var_corr2_list_GAMP[run_num][n_index] = norm_sq_corr2_GAMP

  mean_corr1_list_spec = mean_corr1_list_spec / num_runs
  mean_corr2_list_spec = mean_corr2_list_spec / num_runs

  mean_corr1_list_EM = mean_corr1_list_EM / num_runs
  mean_corr2_list_EM = mean_corr2_list_EM / num_runs

  mean_corr1_list_AM = mean_corr1_list_AM / num_runs
  mean_corr2_list_AM = mean_corr2_list_AM / num_runs

  mean_corr1_list_GAMP = mean_corr1_list_GAMP / num_runs
  mean_corr2_list_GAMP = mean_corr2_list_GAMP / num_runs

  SD_corr1_list_spec = np.sqrt(np.sum(np.square(var_corr1_list_spec - mean_corr1_list_spec), axis=0) / num_runs)
  SD_corr2_list_spec = np.sqrt(np.sum(np.square(var_corr2_list_spec - mean_corr2_list_spec), axis=0) / num_runs)

  SD_corr1_list_EM = np.sqrt(np.sum(np.square(var_corr1_list_EM - mean_corr1_list_EM), axis=0) / num_runs)
  SD_corr2_list_EM = np.sqrt(np.sum(np.square(var_corr2_list_EM - mean_corr2_list_EM), axis=0) / num_runs)

  SD_corr1_list_AM = np.sqrt(np.sum(np.square(var_corr1_list_AM - mean_corr1_list_AM), axis=0) / num_runs)
  SD_corr2_list_AM = np.sqrt(np.sum(np.square(var_corr2_list_AM - mean_corr2_list_AM), axis=0) / num_runs)

  SD_corr1_list_GAMP = np.sqrt(np.sum(np.square(var_corr1_list_GAMP - mean_corr1_list_GAMP), axis=0) / num_runs)
  SD_corr2_list_GAMP = np.sqrt(np.sum(np.square(var_corr2_list_GAMP - mean_corr2_list_GAMP), axis=0) / num_runs)

  Spec_output_list = [mean_corr1_list_spec, mean_corr2_list_spec, SD_corr1_list_spec, SD_corr2_list_spec]
  EM_output_list = [mean_corr1_list_EM, mean_corr2_list_EM, SD_corr1_list_EM, SD_corr2_list_EM]
  AM_output_list = [mean_corr1_list_AM, mean_corr2_list_AM, SD_corr1_list_AM, SD_corr2_list_AM]
  GAMP_output_list = [mean_corr1_list_GAMP, mean_corr2_list_GAMP, SD_corr1_list_GAMP, SD_corr2_list_GAMP]

  return [Spec_output_list, EM_output_list, AM_output_list, GAMP_output_list]

p = 500
n_list = [int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
p1 = 0.6
sigma = 0
num_iter = 10
num_runs = 10

output_list = compare_algo_multi_delta(p, n_list, p1, sigma, num_iter, num_runs)
save('output_list_gau_zero_mean', np.array(output_list))