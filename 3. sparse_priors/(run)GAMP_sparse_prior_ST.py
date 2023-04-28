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

''' Some helper functions '''

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

def sparse_pmf(beta, eps, alpha):
  if beta == 1:
    return (eps / 2) * (1 + alpha)
  elif beta == -1:
    return (eps / 2) * (1 - alpha)
  elif beta == 0:
    return 1 - eps
  else:
    return 0

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

def soft_threshold(input, threshold):
  if input > threshold:
    return input - threshold
  elif input < -1 * threshold:
    return input + threshold
  else:
    return 0

def f_k_ST(B_bar_k, M_k_B, T_k_B, ST_param): 
  # Note that ST_param is usually alpha but here our alpha is used as a 
  # parameter to control the prior.

  inv_M_k_B = linalg.pinv(M_k_B)
  modified_B_bar_k = np.dot(inv_M_k_B, B_bar_k)
  noise_cov = np.dot(inv_M_k_B, np.dot(T_k_B, inv_M_k_B.T))
  output = np.zeros(2)
  output[0] = soft_threshold(modified_B_bar_k[0], ST_param * np.sqrt(noise_cov[0, 0]))
  output[1] = soft_threshold(modified_B_bar_k[1], ST_param * np.sqrt(noise_cov[1, 1]))

  return output

def compute_C_k(Theta_k, R_hat_k, Sigma_k):
  n = len(Theta_k)
  part1 = np.dot(Theta_k.T, R_hat_k)/n
  part2 = np.dot(Sigma_k[2:4,0:2], np.dot(R_hat_k.T, R_hat_k)/n)
  output = np.dot(linalg.pinv(Sigma_k[2:4,2:4]), part1 - part2)
  return output.T

def f_k_prime(B_bar_k, M_k_B, T_k_B, ST_param):

  output = np.zeros((2, 2))
  inv_M_k_B = linalg.pinv(M_k_B)
  modified_B_bar_k = np.dot(inv_M_k_B, B_bar_k)
  noise_cov = np.dot(inv_M_k_B, np.dot(T_k_B, inv_M_k_B))
  if modified_B_bar_k[0] > ST_param * np.sqrt(noise_cov[0, 0]):
    output[0, 0] = inv_M_k_B[0,0]
    output[0, 1] = inv_M_k_B[0,1]
  elif modified_B_bar_k[0] < -1 * ST_param * np.sqrt(noise_cov[0, 0]):
    output[0, 0] = inv_M_k_B[0,0]
    output[0, 1] = inv_M_k_B[0,1]

  if modified_B_bar_k[1] > ST_param * np.sqrt(noise_cov[1, 1]):
    output[1, 0] = inv_M_k_B[1,0]
    output[1, 1] = inv_M_k_B[1,1]
  elif modified_B_bar_k[1] < -1 * ST_param * np.sqrt(noise_cov[1, 1]):
    output[1, 0] = inv_M_k_B[1,0]
    output[1, 1] = inv_M_k_B[1,1]

  return output

# Specific to the prior
def SE_norm_sq_corr(M_k_B, eps_vec, alpha, num_MC_samples):
  eps1 = eps_vec[0]
  eps2 = eps_vec[1]
  beta1 = np.random.choice(np.array([-1, 0, 1]), size=num_MC_samples, p=[(eps1/2)*(1-alpha), 1-eps1, (eps1/2)*(1+alpha)])
  beta2 = np.random.choice(np.array([-1, 0, 1]), size=num_MC_samples, p=[(eps2/2)*(1-alpha), 1-eps2, (eps2/2)*(1+alpha)])
  beta1 = beta1[:, None]
  beta2 = beta2[:, None]
  B_bar_samples = np.concatenate((beta1, beta2), axis=1)
  G_k_B_samples = multivariate_normal([0,0], M_k_B, num_MC_samples)
  
  E_beta1bar_sq = eps1
  E_f1_beta1bar = 0
  E_f1_sq = 0
  E_beta2bar_sq = eps2
  E_f2_beta2bar = 0
  E_f2_sq = 0
  for i in range(num_MC_samples):
    B_bar_sample = B_bar_samples[i]
    G_k_B_sample = G_k_B_samples[i]
    T_k_B = M_k_B
    s = np.dot(M_k_B, B_bar_sample) + G_k_B_sample
    f = f_k_bayes(s, M_k_B, T_k_B, eps_vec, alpha)
    E_f1_beta1bar += f[0]*B_bar_sample[0]
    E_f1_sq += f[0]**2
    E_f2_beta2bar += f[1]*B_bar_sample[1]
    E_f2_sq += f[1]**2
  E_f1_beta1bar /= num_MC_samples
  E_f1_sq /= num_MC_samples
  E_f2_beta2bar /= num_MC_samples
  E_f2_sq /= num_MC_samples

  SE_norm_sq_corr1 = (E_f1_beta1bar**2) / (E_f1_sq * E_beta1bar_sq)
  SE_norm_sq_corr2 = (E_f2_beta2bar**2) / (E_f2_sq * E_beta2bar_sq)
  return SE_norm_sq_corr1, SE_norm_sq_corr2

def norm_sq_corr(beta, beta_hat):
  num = np.square(np.dot(beta, beta_hat))
  denom = np.square(linalg.norm(beta)) * np.square(linalg.norm(beta_hat))
  return num / denom

def MSE(beta, beta_hat):
  output = np.mean(np.square(beta - beta_hat))
  return output

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

def run_matrix_GAMP(n, p, p1, sigma, ST_param, B, B_bar_mean, B_bar_cov, 
                                    B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter):

  delta = n / p

  X = normal(0, np.sqrt(1/n), (n, p))
  Theta = np.dot(X, B)

  # Generating Y: We used ome numpy operational trick to avoid writing 
  # a for loop (inefficient) to compute Y.
  c = binomial(1, p1, n)
  eps = normal(0, sigma, n)
  c = c[:, None]
  Y = (Theta * np.c_[c, 1-c]).sum(1) + eps

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

  prev_min_corr = 0
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
    B_hat_k_plus_1 = np.apply_along_axis(f_k_ST, 1, B_k_plus_1, M_k_plus_1_B, T_k_plus_1_B, ST_param)

    if (np.isnan(B_hat_k_plus_1).any()):
      print('=== EARLY STOPPAGE ===')
      break

    # Computing F_k_plus_1
    F_k_plus_1 = np.zeros((2, 2))
    for j in range(p):
      F_k_plus_1 += f_k_prime(B_k_plus_1[j], M_k_plus_1_B, T_k_plus_1_B, ST_param)
    F_k_plus_1 = F_k_plus_1 / n

    # Computing state evolution for the (k+1)th iteration
    Sigma_k_plus_1 = np.zeros((4,4))
    Sigma_k_plus_1[0:2,0:2] = Sigma_k[0:2,0:2]
    temp_matrix = np.dot(B_hat_k_plus_1.T, B_hat_k_plus_1) / p
    Sigma_k_plus_1[0:2,2:4] = temp_matrix / delta
    Sigma_k_plus_1[2:4,0:2] = temp_matrix / delta
    Sigma_k_plus_1[2:4,2:4] = temp_matrix / delta

    if (np.isnan(Sigma_k_plus_1).any()):
      print('=== EARLY STOPPAGE ===')
      break
    
    # deciding termination of algorithm
    beta1_hat = B_hat_k_plus_1[:, 0]
    beta2_hat = B_hat_k_plus_1[:, 1]
    beta1 = B[:, 0]
    beta2 = B[:, 1]

    current_min_corr = min(norm_sq_corr(beta1, beta1_hat), norm_sq_corr(beta2, beta2_hat))
    if (prev_min_corr >= current_min_corr):
      print('=== EARLY STOPPAGE ===')
      break
    else:
      prev_min_corr = current_min_corr

    # Updating parameters and storing B_hat_k_plus_1 & M_k_plus_1_B
    B_hat_storage.append(B_hat_k_plus_1)
    R_hat_k_minus_1 = R_hat_k
    B_hat_k = B_hat_k_plus_1
    F_k = F_k_plus_1
    M_k_B_storage.append(M_k_plus_1_B)
    M_k_B = M_k_plus_1_B
    T_k_B = T_k_plus_1_B
    Sigma_k = Sigma_k_plus_1

    print('M_k_B\n',M_k_B)
    print('Sigma_k:\n',Sigma_k)

  return B_hat_storage, M_k_B_storage

def get_heatmap_points(p, p1, sigma, eps_vec, alpha, ST_param, delta, num_iter, num_runs):
  n = int(delta * p)
  eps1 = eps_vec[0]
  eps2 = eps_vec[1]
  final_mean_corr1 = 0
  final_mean_corr2 = 0

  for run_num in range(num_runs):
    print('=== Run number: ' + str(run_num + 1) + ' ===')

    np.random.seed(run_num) # so that result is reproducible

    B_bar_mean = np.array([eps1*alpha, eps2*alpha])
    B_bar_cov = np.array([
                          [eps1-(eps1*alpha)**2,0],
                          [0,eps2-(eps2*alpha)**2]
    ])
    beta1 = np.random.choice(np.array([-1, 0, 1]), size=p, p=[(eps1/2)*(1-alpha), 1-eps1, (eps1/2)*(1+alpha)])
    beta2 = np.random.choice(np.array([-1, 0, 1]), size=p, p=[(eps2/2)*(1-alpha), 1-eps2, (eps2/2)*(1+alpha)])
    beta1 = beta1[:, None]
    beta2 = beta2[:, None]
    B = np.concatenate((beta1, beta2), axis=1)

    B_hat_0_row_mean = np.array([eps1*alpha, eps2*alpha])
    B_hat_0_row_cov = np.array([
                          [eps1-(eps1*alpha)**2,0],
                          [0,eps2-(eps2*alpha)**2]
    ])
    beta1 = np.random.choice(np.array([-1, 0, 1]), size=p, p=[(eps1/2)*(1-alpha), 1-eps1, (eps1/2)*(1+alpha)])
    beta2 = np.random.choice(np.array([-1, 0, 1]), size=p, p=[(eps2/2)*(1-alpha), 1-eps2, (eps2/2)*(1+alpha)])
    beta1 = beta1[:, None]
    beta2 = beta2[:, None]
    B_hat_0 = np.concatenate((beta1, beta2), axis=1)

    B_hat_storage, M_k_B_storage = run_matrix_GAMP(n, p, p1, sigma, ST_param, B, B_bar_mean, B_bar_cov, 
                                    B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)
    beta1 = B[:, 0] 
    beta2 = B[:, 1]
    B_hat_final = B_hat_storage[-1]
    beta1_hat_final = B_hat_final[:, 0]
    beta2_hat_final = B_hat_final[:, 1]
      
    norm_sq_corr1 = norm_sq_corr(beta1, beta1_hat_final)
    final_mean_corr1 += norm_sq_corr1

    norm_sq_corr2 = norm_sq_corr(beta2, beta2_hat_final)
    final_mean_corr2 += norm_sq_corr2

  final_mean_corr1 /= num_runs
  final_mean_corr2 /= num_runs
  min_final_mean_corr = min(final_mean_corr1, final_mean_corr2)

  print('final_mean_corr1\n',final_mean_corr1)
  print('final_mean_corr2\n',final_mean_corr2)
  return min_final_mean_corr

p = 1000
p1 = 0.7
sigma = 0
alpha = 0
ST_param = 1.1402

num_iter = 10
num_runs = 10

''' Going row by row.'''
# eps = 1
# delta_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# data = np.zeros(len(delta_list))
# for delta_index in range(len(delta_list)):
  # delta = delta_list[delta_index]
  # eps_vec = np.array([eps, eps])
  # min_corr = get_heatmap_points(p, p1, sigma, eps_vec, alpha, delta, num_iter, num_runs)
  # data[delta_index] = min_corr

# print('data\n', data)

''' Going for entire matrix.'''
eps_list = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
delta_list = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

data = np.zeros((len(eps_list), len(delta_list)))
for eps_index in range(len(eps_list)):
  for delta_index in range(len(delta_list)):
    eps = eps_list[eps_index]
    delta = delta_list[delta_index]
    eps_vec = np.array([eps, eps])
    min_corr = get_heatmap_points(p, p1, sigma, eps_vec, alpha, ST_param, delta, num_iter, num_runs)
    data[eps_index][delta_index] = min_corr

save('data_0.7p1_ST', data)