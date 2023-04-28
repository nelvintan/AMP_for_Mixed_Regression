import numpy as np
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import uniform
from numpy import save

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

''' Some helper functions '''
# We don't use the premade pdf functions from scipy because
# then we wouldn't be able to use jit for parallelism.

def norm_pdf(x, mean, var):
  first_part = 1 / np.sqrt(2 * np.pi * var)
  second_part = np.exp((-1/2) * ((x - mean) ** 2 / var))
  return first_part * second_part

def multi_norm_pdf(x, mean, cov): # multivariate normal distribution
  dimension = len(x)
  if linalg.det(cov) == 0:
    print('linalg.det(cov) == 0')
  first_part = 1 / np.sqrt((2 * np.pi) ** dimension * linalg.det(cov))
  if np.isnan(first_part):
    print('first_part is nan.')
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

''' Below is the computation of E[Z|Z^k,bar{Y}] using importance sampling '''

def eqv_indicator(input1, input2):

  if input1 == input2:
    return 1
  else:
    return 0

def pdf_Y_bar_given_Z(Z, Y_bar, sigma, c1, c2):
  
  Z1 = Z[0]
  Z2 = Z[1]
  if sigma == 0:
    if Z1 + c1 > Z2 + c2:
      return eqv_indicator(Y_bar, Z1+c1)
    else:
      return 1 - eqv_indicator(Y_bar, Z1+c1)
  elif sigma > 0:
    if Z1 + c1 > Z2 + c2:
      return norm.pdf((Y_bar - Z1 - c1) / sigma)
    else:
      return norm.pdf((Y_bar - Z2 - c2) / sigma)

def E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, c1, c2, sigma):
  # NOTE: c1, c2 are the intercepts of max-affine reg.

  Sigma_11 = Sigma_k[:2,:2] 
  Sigma_12 = Sigma_k[:2,2:] 
  Sigma_21 = Sigma_k[2:,:2] 
  Sigma_22 = Sigma_k[2:,2:] 
  mean_Z_given_Zk = np.dot(Sigma_12, np.dot(linalg.pinv(Sigma_22), Z_k)) 
  cov_Z_given_Zk = Sigma_11 - np.dot(Sigma_12, np.dot(linalg.pinv(Sigma_22), Sigma_21))
  num_samples = 100
  Z_samples = multivariate_normal(mean_Z_given_Zk, cov_Z_given_Zk, num_samples)

  num_data = np.zeros((num_samples, 2))
  denom_sum = 0
  denom_count = 0
  for i in range(num_samples):
    Z_sample = Z_samples[i]
    p_Zk = multivariate_normal_sp.pdf(Z_k, mean=np.array([0,0]), cov=Sigma_k[2:,2:], allow_singular=True)
    p_Ybar_given_Z_Zk = pdf_Y_bar_given_Z(Z_sample, Y_bar, sigma, c1, c2)
    num_data[i] = Z_sample * p_Zk * p_Ybar_given_Z_Zk
    denom_sum += p_Zk * p_Ybar_given_Z_Zk
    denom_count += 1
  
  numerator = np.mean(num_data, axis=0)
  denominator = denom_sum / denom_count

  output = numerator / denominator

  return output

def E_Z_given_Ybar(Y_bar, Sigma_k, c1, c2, sigma):
  # This function is for EM-AMP.
  # NOTE: c1, c2 are the intercepts of max-affine reg.

  num_samples = 100
  Sigma_11 = Sigma_k[:2,:2]
  Z_samples = multivariate_normal(np.array([0,0]), Sigma_11, num_samples)

  num_data = np.zeros((num_samples, 2))
  denom_sum = 0
  denom_count = 0
  for i in range(num_samples):
    Z_sample = Z_samples[i]
    p_Ybar_given_Z = pdf_Y_bar_given_Z(Z_sample, Y_bar, sigma, c1, c2)
    num_data[i] = Z_sample * p_Ybar_given_Z
    denom_sum += p_Ybar_given_Z
    denom_count += 1
  
  numerator = np.mean(num_data, axis=0)
  denominator = denom_sum / denom_count

  output = numerator / denominator

  return output

def g_k_bayes(Z_k, Y_bar, Sigma_k, c1, c2, sigma):
  
  mat1 = Var_Z_given_Zk(Sigma_k)
  vec2 = E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, c1, c2, sigma)
  vec3 = E_Z_given_Zk(Sigma_k, Z_k)
  
  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_Y_bar, Sigma_k, c1, c2, sigma):
  Z_k = Z_k_and_Y_bar[:2]
  Y_bar = Z_k_and_Y_bar[2:]
  return g_k_bayes(Z_k, Y_bar, Sigma_k, c1, c2, sigma)

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

def MSE_beta1_SE(M_k_B, B_bar_mean, B_bar_cov):
  T_k_B = M_k_B
  C_1 = np.dot(M_k_B.T, linalg.pinv(np.dot(M_k_B, M_k_B.T) + T_k_B))
  C_2 = np.dot(np.eye(2) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)
  C_2_1 = C_2[0]
  C_3_11 = C_3[0, 0]
  C_3_12 = C_3[0, 1]
  C_1_11 = C_1[0, 0]
  C_1_12 = C_1[0, 1]

  part1 = C_2_1**2+2*C_2_1*(C_3_11-1)*B_bar_mean[0]+2*C_2_1*C_3_12*B_bar_mean[1]
  part2 = (C_3_11-1)**2*(B_bar_cov[0,0]+B_bar_mean[0]**2)+2*(C_3_11-1)*C_3_12*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])
  part3 = (C_3_12**2)*(B_bar_cov[1,1]+B_bar_mean[1]**2)
  part4 = (C_1_11**2)*T_k_B[0,0]+2*C_1_11*C_1_12*T_k_B[0,1]+(C_1_12**2)*T_k_B[1,1]

  return part1 + part2 + part3 + part4

def MSE_beta2_SE(M_k_B, B_bar_mean, B_bar_cov):
  T_k_B = M_k_B
  C_1 = np.dot(M_k_B.T, linalg.pinv(np.dot(M_k_B, M_k_B.T) + T_k_B))
  C_2 = np.dot(np.eye(2) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)
  C_2_2 = C_2[1]
  C_3_21 = C_3[1, 0]
  C_3_22 = C_3[1, 1]
  C_1_21 = C_1[1, 0]
  C_1_22 = C_1[1, 1]

  part1 = C_2_2**2+2*C_2_2*C_3_21*B_bar_mean[0]+2*C_2_2*(C_3_22-1)*B_bar_mean[1]
  part2 = (C_3_21**2)*(B_bar_cov[0,0]+B_bar_mean[0]**2)+2*C_3_21*(C_3_22-1)*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])
  part3 = ((C_3_22-1)**2)*(B_bar_cov[1,1]+B_bar_mean[1]**2)
  part4 = (C_1_21**2)*T_k_B[0,0]+2*C_1_21*C_1_22*T_k_B[0,1]+(C_1_22**2)*T_k_B[1,1]

  return part1 + part2 + part3 + part4

def norm_sq_corr1_SE(M_k_B, B_bar_mean, B_bar_cov):
  T_k_B = M_k_B
  C_1 = np.dot(M_k_B.T, linalg.pinv(np.dot(M_k_B, M_k_B.T) + T_k_B))
  C_2 = np.dot(np.eye(2) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)
  C_2_1 = C_2[0]
  C_3_11 = C_3[0, 0]
  C_3_12 = C_3[0, 1]
  C_1_11 = C_1[0, 0]
  C_1_12 = C_1[0, 1]

  num = C_2_1*B_bar_mean[0]+C_3_11*(B_bar_cov[0,0]+B_bar_mean[0]**2)+C_3_12*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])

  part1 = C_2_1**2+2*C_2_1*C_3_11*B_bar_mean[0]+2*C_2_1*C_3_12*B_bar_mean[1]+2*C_3_11*C_3_12*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])
  part2 = (C_3_11**2)*(B_bar_cov[0,0]+B_bar_mean[0]**2)+(C_3_12**2)*(B_bar_cov[1,1]+B_bar_mean[1]**2)
  part3 = (C_1_11**2)*T_k_B[0,0]+2*C_1_11*C_1_12*T_k_B[0,1]+(C_1_12**2)*T_k_B[1,1]

  part4 = B_bar_cov[0,0]+B_bar_mean[0]**2

  return (num**2) / (part4 * (part1 + part2 + part3))

def norm_sq_corr2_SE(M_k_B, B_bar_mean, B_bar_cov):
  T_k_B = M_k_B
  C_1 = np.dot(M_k_B.T, linalg.pinv(np.dot(M_k_B, M_k_B.T) + T_k_B))
  C_2 = np.dot(np.eye(2) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)
  C_2_2 = C_2[1]
  C_3_21 = C_3[1, 0]
  C_3_22 = C_3[1, 1]
  C_1_21 = C_1[1, 0]
  C_1_22 = C_1[1, 1]

  num = C_2_2*B_bar_mean[1]+C_3_21*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])+C_3_22*(B_bar_cov[1,1]+B_bar_mean[1]**2)

  part1 = C_2_2**2+2*C_2_2*C_3_21*B_bar_mean[0]+2*C_2_2*C_3_22*B_bar_mean[1]+2*C_3_21*C_3_22*(B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1])
  part2 = (C_3_21**2)*(B_bar_cov[0,0]+B_bar_mean[0]**2)+(C_3_22**2)*(B_bar_cov[1,1]+B_bar_mean[1]**2)
  part3 = (C_1_21**2)*T_k_B[0,0]+2*C_1_21*C_1_22*T_k_B[0,1]+(C_1_22**2)*T_k_B[1,1]

  part4 = B_bar_cov[1,1]+B_bar_mean[1]**2

  return (num**2) / (part4 * (part1 + part2 + part3))

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

def run_matrix_GAMP(n, p, c1, c2, sigma, X, Y, B, B_bar_mean, B_bar_cov, 
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

  # Storage of the estimate B_hat and Theta_k
  B_hat_storage = []
  B_hat_storage.append(B_hat_0)

  # Storage of the state evolution param M_k_B
  M_k_B_storage = []

  # Required output for EM-algo
  E_Z_given_Ybar_emp = np.zeros(2)

  prev_min_corr = 0
  for k in range(num_iter):
    print("=== Running iteration: " + str(k+1) + " ===")
    
    # Computing Theta_k
    Theta_k = np.dot(X, B_hat_k) - np.dot(R_hat_k_minus_1, F_k.T)

    # Computing R_hat_k
    Theta_k_and_Y = np.concatenate((Theta_k,Y[:,None]), axis=1)
    R_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, Theta_k_and_Y, Sigma_k, c1, c2, sigma)

    if np.all(E_Z_given_Ybar_emp == np.zeros(2)):
      E_Z_given_Ybar_emp = np.mean(np.apply_along_axis(E_Z_given_Ybar, 1, Y[:,None], Sigma_k, c1, c2, sigma), axis=0)

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

  return B_hat_storage, M_k_B_storage, E_Z_given_Ybar_emp

def compute_c(Y, Theta_hat_m, E_Z_given_Ybar_emp, c1_m, c2_m):
  size1 = 0
  sum1 = 0
  size2 = 0
  sum2 = 0
  for i in range(len(Theta_hat_m)):
    Theta_hat_m_i = Theta_hat_m[i]
    Theta_hat_m_i1 = Theta_hat_m_i[0]
    Theta_hat_m_i2 = Theta_hat_m_i[1]
    if Theta_hat_m_i1 + c1_m > Theta_hat_m_i2 + c2_m:
      size1 += 1
      sum1 += Y[i]
    else:
      size2 += 1
      sum2 += Y[i]
  c1_est = 0
  c2_est = 0
  if size1 != 0: 
    c1_est = (sum1 / size1) - E_Z_given_Ybar_emp[0]
  else: 
    c1_est = 0 - E_Z_given_Ybar_emp[0]
  if size2 != 0: 
    c2_est = (sum2 / size2) - E_Z_given_Ybar_emp[1]
  else: 
    c2_est = 0 - E_Z_given_Ybar_emp[1]
  return c1_est, c2_est

def run_EM_GAMP(n, p, c1, c2, c1_0, c2_0, sigma, X, Y, B, iter_num_EM, iter_num_GAMP,
                B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov):
  
  beta1_hat_m = B_hat_0[:, 0]
  beta2_hat_m = B_hat_0[:, 1]

  # we call our iterations m here.
  Theta_hat_m = np.zeros((n, 2))
  B_hat_m = np.zeros((p, 2))
  c1_m = c1_0
  c2_m = c2_0
  beta1_hat_full_m = np.append(beta1_hat_m, c1_m)
  beta2_hat_full_m = np.append(beta2_hat_m, c2_m)
  beta1_hat_full_list = [beta1_hat_full_m,]
  beta2_hat_full_list = [beta2_hat_full_m,]

  prev_min_corr = 0
  beta1_full = np.append(B[:,0], c1)
  beta2_full = np.append(B[:,1], c2)
  for m in range(iter_num_EM):
    B_hat_storage, M_k_B_storage, E_Z_given_Ybar_emp = run_matrix_GAMP(n, p, c1_m, c2_m, sigma, X, Y, B, B_bar_mean, B_bar_cov, 
                                                   B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, iter_num_GAMP)
    B_hat_m = B_hat_storage[-1]
    Theta_hat_m = np.dot(X, B_hat_m)
    c1_m, c2_m = compute_c(Y, Theta_hat_m, E_Z_given_Ybar_emp, c1_m, c2_m)

    beta1_hat_m = B_hat_m[:, 0]
    beta2_hat_m = B_hat_m[:, 1]
    beta1_hat_full_m = np.append(beta1_hat_m, c1_m)
    beta2_hat_full_m = np.append(beta2_hat_m, c2_m)

    # deciding termination of algorithm
    current_min_corr = min(norm_sq_corr(beta1_full, beta1_hat_full_m), norm_sq_corr(beta2_full, beta2_hat_full_m))
    if (prev_min_corr >= current_min_corr):
      print('=== EARLY STOPPAGE ===')
      break
    else:
      prev_min_corr = current_min_corr

    beta1_hat_full_list.append(beta1_hat_full_m)
    beta2_hat_full_list.append(beta2_hat_full_m)
  
  print('======================> final c1_m, c2_m:', c1_m, c2_m)
  return beta1_hat_full_list, beta2_hat_full_list

'''
ALternating minimization (AM) algorithm
Link: https://arxiv.org/abs/1906.09255
Notations follow from the above paper.
'''

def run_AM(n, p, sigma, X_full, Y, beta1_full, beta2_full, beta1_full_0, beta2_full_0, c1_0, c2_0, num_iter):

  delta = n / p

  B_hat_0 = np.column_stack((beta1_full_0, beta2_full_0))
  B_hat_storage = []
  B_hat_storage.append(B_hat_0)
  
  beta1_full_t = beta1_full_0
  beta2_full_t = beta2_full_0

  prev_min_corr = 0
  for t in range(num_iter):
    # AM part I: Guess the labels
    S1 = []
    S2 = []
    for i in range(n):
      if np.dot(X_full[i,:],beta1_full_t) >= np.dot(X_full[i,:],beta2_full_t):
        S1.append(i)
      else:
        S2.append(i)

    # AM part II: Solve least squares
    beta1_full_t = np.linalg.lstsq(np.take(X_full, S1, axis=0), Y[S1], rcond=None)[0]
    beta2_full_t = np.linalg.lstsq(np.take(X_full, S2, axis=0), Y[S2], rcond=None)[0]

    # deciding termination of algorithm
    current_min_corr = min(norm_sq_corr(beta1_full, beta1_full_t), norm_sq_corr(beta2_full, beta2_full_t))
    if (prev_min_corr >= current_min_corr):
      print('=== EARLY STOPPAGE ===')
      break
    else:
      prev_min_corr = current_min_corr

    B_hat_t = np.column_stack((beta1_full_t, beta2_full_t))
    B_hat_storage.append(B_hat_t)

  return B_hat_storage

def compare_algo_multi_delta(p, n_list, sigma, num_iter, num_runs):
  
  num_deltas = len(n_list)

  mean_corr1_list_AM = np.zeros(num_deltas)
  mean_corr2_list_AM = np.zeros(num_deltas)
  var_corr1_list_AM = np.zeros((num_runs, num_deltas))
  var_corr2_list_AM = np.zeros((num_runs, num_deltas))

  mean_corr1_list_GAMP = np.zeros(num_deltas)
  mean_corr2_list_GAMP = np.zeros(num_deltas)
  var_corr1_list_GAMP = np.zeros((num_runs, num_deltas))
  var_corr2_list_GAMP = np.zeros((num_runs, num_deltas))

  mean_corr1_list_EMGAMP = np.zeros(num_deltas)
  mean_corr2_list_EMGAMP = np.zeros(num_deltas)
  var_corr1_list_EMGAMP = np.zeros((num_runs, num_deltas))
  var_corr2_list_EMGAMP = np.zeros((num_runs, num_deltas))

  for n_index in range(len(n_list)):
    n = n_list[n_index]
    print('------------> dealing with n:', n)
    final_corr1 = 0
    final_corr2 = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible

      c1, c2 = 1, 1
      c1_0, c2_0 = 0, 0

      B_bar_mean = np.array([0, 1])
      B_bar_cov = np.eye(2)
      B = multivariate_normal(B_bar_mean, B_bar_cov, p)
      beta1 = B[:, 0]
      beta2 = B[:, 1]

      B_hat_0_row_mean = B_bar_mean
      B_hat_0_row_cov = B_bar_cov
      B_hat_0 = multivariate_normal(B_hat_0_row_mean, B_hat_0_row_cov, p)
      beta1_full_0 = np.append(B_hat_0[:,0], c1_0)
      beta2_full_0 = np.append(B_hat_0[:,1], c2_0)

      X = normal(0, np.sqrt(1/n), (n, p))
      Theta = np.dot(X, B)
      c1_vec = np.full(n, c1)
      c2_vec = np.full(n, c2)
      eps = normal(0, sigma, n)
      Theta1 = Theta[:,0] + c1_vec
      Theta2 = Theta[:,1] + c2_vec
      Y = np.maximum(Theta1, Theta2) + eps

      X_full = np.column_stack([X, np.ones(n)])
      beta1_full = np.append(beta1, c1)
      beta2_full = np.append(beta2, c2)

      iter_num_EM = 5
      iter_num_GAMP = num_iter
      B_hat_storage_AM = run_AM(n, p, sigma, X_full, Y, beta1_full, beta2_full, beta1_full_0, beta2_full_0, c1_0, c2_0, num_iter)
      B_hat_storage_GAMP, M_k_B_storage, E_Z_given_Ybar_emp = run_matrix_GAMP(n, p, c1, c2, sigma, X, Y, B, B_bar_mean, B_bar_cov, 
                                                                              B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, iter_num_GAMP)
      beta1_hat_full_list, beta2_hat_full_list = run_EM_GAMP(n, p, c1, c2, c1_0, c2_0, sigma, X, Y, B, iter_num_EM, iter_num_GAMP, 
                                                            B_bar_mean, B_bar_cov, B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov)

      # For AM.
      B_hat_AM = B_hat_storage_AM[-1]
      beta1_full_AM = B_hat_AM[:, 0]
      beta2_full_AM = B_hat_AM[:, 1]

      norm_sq_corr1_AM = norm_sq_corr(beta1_full, beta1_full_AM)
      mean_corr1_list_AM[n_index] += norm_sq_corr1_AM
      var_corr1_list_AM[run_num][n_index] = norm_sq_corr1_AM

      norm_sq_corr2_AM = norm_sq_corr(beta2_full, beta2_full_AM)
      mean_corr2_list_AM[n_index] += norm_sq_corr2_AM
      var_corr2_list_AM[run_num][n_index] = norm_sq_corr2_AM

      # For GAMP.
      B_hat_GAMP = B_hat_storage_GAMP[-1]
      beta1_GAMP = B_hat_GAMP[:, 0]
      beta2_GAMP = B_hat_GAMP[:, 1]
      beta1_full_GAMP = np.append(beta1_GAMP, c1)
      beta2_full_GAMP = np.append(beta2_GAMP, c2)

      norm_sq_corr1_GAMP = norm_sq_corr(beta1_full, beta1_full_GAMP)
      mean_corr1_list_GAMP[n_index] += norm_sq_corr1_GAMP
      var_corr1_list_GAMP[run_num][n_index] = norm_sq_corr1_GAMP

      norm_sq_corr2_GAMP = norm_sq_corr(beta2_full, beta2_full_GAMP)
      mean_corr2_list_GAMP[n_index] += norm_sq_corr2_GAMP
      var_corr2_list_GAMP[run_num][n_index] = norm_sq_corr2_GAMP

      # For EM-GAMP.
      beta1_full_EMGAMP = beta1_hat_full_list[-1]
      beta2_full_EMGAMP = beta2_hat_full_list[-1]

      norm_sq_corr1_EMGAMP = norm_sq_corr(beta1_full, beta1_full_EMGAMP)
      mean_corr1_list_EMGAMP[n_index] += norm_sq_corr1_EMGAMP
      var_corr1_list_EMGAMP[run_num][n_index] = norm_sq_corr1_EMGAMP

      norm_sq_corr2_EMGAMP = norm_sq_corr(beta2_full, beta2_full_EMGAMP)
      mean_corr2_list_EMGAMP[n_index] += norm_sq_corr2_EMGAMP
      var_corr2_list_EMGAMP[run_num][n_index] = norm_sq_corr2_EMGAMP
  
  mean_corr1_list_AM = mean_corr1_list_AM / num_runs
  mean_corr2_list_AM = mean_corr2_list_AM / num_runs

  SD_corr1_list_AM = np.sqrt(np.sum(np.square(var_corr1_list_AM - mean_corr1_list_AM), axis=0) / num_runs)
  SD_corr2_list_AM = np.sqrt(np.sum(np.square(var_corr2_list_AM - mean_corr2_list_AM), axis=0) / num_runs)

  AM_output_list = [mean_corr1_list_AM, mean_corr2_list_AM, SD_corr1_list_AM, SD_corr2_list_AM]

  mean_corr1_list_GAMP = mean_corr1_list_GAMP / num_runs
  mean_corr2_list_GAMP = mean_corr2_list_GAMP / num_runs

  SD_corr1_list_GAMP = np.sqrt(np.sum(np.square(var_corr1_list_GAMP - mean_corr1_list_GAMP), axis=0) / num_runs)
  SD_corr2_list_GAMP = np.sqrt(np.sum(np.square(var_corr2_list_GAMP - mean_corr2_list_GAMP), axis=0) / num_runs)

  GAMP_output_list = [mean_corr1_list_GAMP, mean_corr2_list_GAMP, SD_corr1_list_GAMP, SD_corr2_list_GAMP]

  mean_corr1_list_EMGAMP = mean_corr1_list_EMGAMP / num_runs
  mean_corr2_list_EMGAMP = mean_corr2_list_EMGAMP / num_runs

  SD_corr1_list_EMGAMP = np.sqrt(np.sum(np.square(var_corr1_list_EMGAMP - mean_corr1_list_EMGAMP), axis=0) / num_runs)
  SD_corr2_list_EMGAMP = np.sqrt(np.sum(np.square(var_corr2_list_EMGAMP - mean_corr2_list_EMGAMP), axis=0) / num_runs)

  EMGAMP_output_list = [mean_corr1_list_EMGAMP, mean_corr2_list_EMGAMP, SD_corr1_list_EMGAMP, SD_corr2_list_EMGAMP]

  return [AM_output_list, GAMP_output_list, EMGAMP_output_list]

p = 500
n_list = [int(0.5*p), int(1*p), int(1.5*p), int(2*p), int(2.5*p), int(3*p), int(3.5*p), int(4*p), int(4.5*p), int(5*p)]
sigma = 0.1
num_iter = 5
num_runs = 5

output_list = compare_algo_multi_delta(p, n_list, sigma, num_iter, num_runs)
save('diff_mean_same_inter_sig01', np.array(output_list))