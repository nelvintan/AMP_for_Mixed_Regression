import numpy as np
from numpy import save
from numpy import linalg
from numpy.random import multivariate_normal
from numpy.random import normal
from numpy.random import binomial
from numpy.random import multinomial
from numpy.random import uniform

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
  
  Sigma_0 = np.zeros((6, 6))

  # Fill in all the diagonals.
  Sigma_0[0, 0] = B_bar_cov[0, 0] + B_bar_mean[0]**2
  Sigma_0[1, 1] = B_bar_cov[1, 1] + B_bar_mean[1]**2
  Sigma_0[2, 2] = B_bar_cov[2, 2] + B_bar_mean[2]**2
  Sigma_0[3, 3] = B_hat_0_row_cov[0, 0] + B_hat_0_row_mean[0]**2
  Sigma_0[4, 4] = B_hat_0_row_cov[1, 1] + B_hat_0_row_mean[1]**2
  Sigma_0[5, 5] = B_hat_0_row_cov[2, 2] + B_hat_0_row_mean[2]**2

  # Fill in all the off-diagonals (the upper triangle).
  Sigma_0[0, 1] = B_bar_cov[0, 1] + B_bar_mean[0] * B_bar_mean[1]
  Sigma_0[0, 2] = B_bar_cov[0, 2] + B_bar_mean[0] * B_bar_mean[2]
  Sigma_0[0, 3] = B_bar_mean[0] * B_hat_0_row_mean[0]
  Sigma_0[0, 4] = B_bar_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[0, 5] = B_bar_mean[0] * B_hat_0_row_mean[2]

  Sigma_0[1, 2] = B_bar_cov[1, 2] + B_bar_mean[1] * B_bar_mean[2]
  Sigma_0[1, 3] = B_bar_mean[1] * B_hat_0_row_mean[0]
  Sigma_0[1, 4] = B_bar_mean[1] * B_hat_0_row_mean[1]
  Sigma_0[1, 5] = B_bar_mean[1] * B_hat_0_row_mean[2]

  Sigma_0[2, 3] = B_bar_mean[2] * B_hat_0_row_mean[0]
  Sigma_0[2, 4] = B_bar_mean[2] * B_hat_0_row_mean[1]
  Sigma_0[2, 5] = B_bar_mean[2] * B_hat_0_row_mean[2]

  Sigma_0[3, 4] = B_hat_0_row_cov[0, 1] + B_hat_0_row_mean[0] * B_hat_0_row_mean[1]
  Sigma_0[3, 5] = B_hat_0_row_cov[0, 2] + B_hat_0_row_mean[0] * B_hat_0_row_mean[2]

  Sigma_0[4, 5] = B_hat_0_row_cov[1, 2] + B_hat_0_row_mean[1] * B_hat_0_row_mean[2]

  # Fill in all the off-diagonals (the lower triangle).

  Sigma_0[1, 0] = Sigma_0[0, 1]
  Sigma_0[2, 0] = Sigma_0[0, 2]
  Sigma_0[2, 1] = Sigma_0[1, 2]
  Sigma_0[4, 3] = Sigma_0[3, 4]
  Sigma_0[5, 3] = Sigma_0[3, 5]
  Sigma_0[5, 4] = Sigma_0[4, 5]

  Sigma_0[3:, :3] = Sigma_0[:3, 3:].T

  return Sigma_0 / delta

'''
Our GAMP functions below -- note that the inputs Z_k and Y_bar will be exchanged
for Theta^k_i and Y_i in our matrix-GAMP algorithm.
'''

def Var_Z_given_Zk(Sigma_k):
  return Sigma_k[0:3, 0:3] - np.dot(np.dot(Sigma_k[0:3, 3:6], linalg.pinv(Sigma_k[3:6, 3:6])), Sigma_k[3:6, 0:3])

def E_Z_given_Zk(Sigma_k, Z_k):
  return np.dot(np.dot(Sigma_k[0:3, 3:6], linalg.pinv(Sigma_k[3:6, 3:6])), Z_k)

def E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, alpha_vec, sigma):
  Sigma_k1_Y = np.zeros((7, 7))
  Sigma_k1_Y[:6, :6] = Sigma_k
  Sigma_k1_Y[6, :6] = Sigma_k[:, 0]
  Sigma_k1_Y[:6, 6] = Sigma_k[0, :]
  Sigma_k1_Y[6, 6] = Sigma_k[0, 0] + sigma**2
  
  Sigma_k2_Y = np.zeros((7, 7))
  Sigma_k2_Y[:6, :6] = Sigma_k
  Sigma_k2_Y[6, :6] = Sigma_k[:, 1]
  Sigma_k2_Y[:6, 6] = Sigma_k[1, :]
  Sigma_k2_Y[6, 6] = Sigma_k[1, 1] + sigma**2

  Sigma_k3_Y = np.zeros((7, 7))
  Sigma_k3_Y[:6, :6] = Sigma_k
  Sigma_k3_Y[6, :6] = Sigma_k[:, 2]
  Sigma_k3_Y[:6, 6] = Sigma_k[2, :]
  Sigma_k3_Y[6, 6] = Sigma_k[2, 2] + sigma**2
  
  E_Z_given_Zk_Ybar_cbar1 = np.dot(Sigma_k1_Y[:3, 3:], np.dot(linalg.pinv(Sigma_k1_Y[3:, 3:]), np.concatenate((Z_k, Y_bar))))
  E_Z_given_Zk_Ybar_cbar2 = np.dot(Sigma_k2_Y[:3, 3:], np.dot(linalg.pinv(Sigma_k2_Y[3:, 3:]), np.concatenate((Z_k, Y_bar))))
  E_Z_given_Zk_Ybar_cbar3 = np.dot(Sigma_k3_Y[:3, 3:], np.dot(linalg.pinv(Sigma_k3_Y[3:, 3:]), np.concatenate((Z_k, Y_bar))))
  
  mean = np.zeros(4)
  cov1 = Sigma_k1_Y[3:, 3:]
  cov2 = Sigma_k2_Y[3:, 3:]
  cov3 = Sigma_k3_Y[3:, 3:]

  alpha1, alpha2, alpha3 = alpha_vec[0], alpha_vec[1], alpha_vec[2]

  if is_pos_semi_def_scipy(cov1) == False or is_pos_semi_def_scipy(cov2) == False or is_pos_semi_def_scipy(cov3) == False:
    return np.array([np.nan, np.nan, np.nan])

  P_Zk_Ybar_given_cbar1 = multivariate_normal_sp.pdf(np.concatenate((Z_k, Y_bar)), mean=mean, cov=cov1, allow_singular=True)
  P_Zk_Ybar_given_cbar2 = multivariate_normal_sp.pdf(np.concatenate((Z_k, Y_bar)), mean=mean, cov=cov2, allow_singular=True)
  P_Zk_Ybar_given_cbar3 = multivariate_normal_sp.pdf(np.concatenate((Z_k, Y_bar)), mean=mean, cov=cov3, allow_singular=True)

  denom = alpha1*P_Zk_Ybar_given_cbar1 + alpha2*P_Zk_Ybar_given_cbar2 + alpha3*P_Zk_Ybar_given_cbar3
  P_cbar1_given_Zk_Ybar = (alpha1*P_Zk_Ybar_given_cbar1) / denom
  P_cbar2_given_Zk_Ybar = (alpha2*P_Zk_Ybar_given_cbar2) / denom
  P_cbar3_given_Zk_Ybar = (alpha3*P_Zk_Ybar_given_cbar3) / denom
  
  output = P_cbar1_given_Zk_Ybar*E_Z_given_Zk_Ybar_cbar1 + P_cbar2_given_Zk_Ybar*E_Z_given_Zk_Ybar_cbar2 + P_cbar3_given_Zk_Ybar*E_Z_given_Zk_Ybar_cbar3

  return output

def g_k_bayes(Z_k, Y_bar, Sigma_k, alpha_vec, sigma): 
  mat1 = Var_Z_given_Zk(Sigma_k)
  vec2 = E_Z_given_Zk_Ybar(Z_k, Y_bar, Sigma_k, alpha_vec, sigma)
  vec3 = E_Z_given_Zk(Sigma_k, Z_k)
  
  return np.dot(linalg.pinv(mat1), vec2 - vec3)

# wrapper function so that it fits into the requirement of np.apply_along_axis().
def g_k_bayes_wrapper(Z_k_and_Y_bar, Sigma_k, alpha_vec, sigma):
  Z_k = Z_k_and_Y_bar[:3]
  Y_bar = Z_k_and_Y_bar[3:]

  return g_k_bayes(Z_k, Y_bar, Sigma_k, alpha_vec, sigma)

def f_k_bayes(B_bar_k, M_k_B, T_k_B, B_bar_mean, B_bar_cov):
  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  part2 = B_bar_k - np.dot(M_k_B, B_bar_mean)
  output = B_bar_mean + np.dot(np.dot(B_bar_cov, M_k_B.T), np.dot(part1, part2))

  return output

def compute_C_k(Theta_k, R_hat_k, Sigma_k):
  n = len(Theta_k)
  part1 = np.dot(Theta_k.T, R_hat_k)/n
  part2 = np.dot(Sigma_k[3:6,0:3], np.dot(R_hat_k.T, R_hat_k)/n)
  output = np.dot(linalg.pinv(Sigma_k[3:6,3:6]), part1 - part2)
  
  return output.T

# This only holds for jointly Gaussian priors.
def f_k_prime(M_k_B, T_k_B, B_bar_cov):
  part1 = linalg.pinv(np.dot(M_k_B, np.dot(B_bar_cov, M_k_B.T)) + T_k_B)
  output = np.dot(part1, np.dot(M_k_B, B_bar_cov))
  return output

def norm_sq_corr1_SE(M_k_B, B_bar_mean, B_bar_cov):
  '''These are computed from the state evolution parameters'''
  T_k_B = M_k_B
  T_11 = T_k_B[0,0]
  T_12 = T_k_B[0,1]
  T_13 = T_k_B[0,2]
  T_21 = T_k_B[1,0]
  T_22 = T_k_B[1,1]
  T_23 = T_k_B[1,2]
  T_31 = T_k_B[2,0]
  T_32 = T_k_B[2,1]
  T_33 = T_k_B[2,2]

  C_1 = np.dot(B_bar_cov,np.dot(M_k_B.T,linalg.pinv(np.dot(np.dot(M_k_B,B_bar_cov), M_k_B.T)+T_k_B)))
  C_2 = np.dot(np.eye(3) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)

  C_2_1 = C_2[0]
  C_2_2 = C_2[1]
  C_2_3 = C_2[2]

  C_3_11 = C_3[0,0]
  C_3_12 = C_3[0,1]
  C_3_13 = C_3[0,2]
  C_3_21 = C_3[1,0]
  C_3_22 = C_3[1,1]
  C_3_23 = C_3[1,2]
  C_3_31 = C_3[2,0]
  C_3_32 = C_3[2,1]
  C_3_33 = C_3[2,2]

  C_1_11 = C_1[0,0]
  C_1_12 = C_1[0,1]
  C_1_13 = C_1[0,2]
  C_1_21 = C_1[1,0]
  C_1_22 = C_1[1,1]
  C_1_23 = C_1[1,2]
  C_1_31 = C_1[2,0]
  C_1_32 = C_1[2,1]
  C_1_33 = C_1[2,2]

  E_beta1_bar_sq = B_bar_cov[0,0]+B_bar_mean[0]**2
  E_beta2_bar_sq = B_bar_cov[1,1]+B_bar_mean[1]**2
  E_beta3_bar_sq = B_bar_cov[2,2]+B_bar_mean[2]**2
  E_beta1_bar_beta2_bar = B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1]
  E_beta1_bar_beta3_bar = B_bar_cov[0,2]+B_bar_mean[0]*B_bar_mean[2]
  E_beta2_bar_beta3_bar = B_bar_cov[1,2]+B_bar_mean[1]*B_bar_mean[2]

  num = C_2_1*B_bar_mean[0]+C_3_11*E_beta1_bar_sq+C_3_12*E_beta1_bar_beta2_bar+C_3_13*E_beta1_bar_beta3_bar

  part1 = C_2_1**2+2*C_2_1*C_3_11*B_bar_mean[0]+2*C_2_1*C_3_12*B_bar_mean[1]+2*C_2_1*C_3_13*B_bar_mean[2]
  part2 = (C_3_11**2)*E_beta1_bar_sq+2*C_3_11*C_3_12*E_beta1_bar_beta2_bar+2*C_3_11*C_3_13*E_beta1_bar_beta3_bar
  part3 = (C_3_12**2)*E_beta2_bar_sq+2*C_3_12*C_3_13*E_beta2_bar_beta3_bar+(C_3_13**2)*E_beta3_bar_sq
  part4 = (C_1_11**2)*T_11+2*C_1_11*C_1_12*T_12+2*C_1_11*C_1_13*T_13+(C_1_12**2)*T_22+2*C_1_12*C_1_13*T_23+(C_1_13**2)*T_33

  return (num**2) / (E_beta1_bar_sq * (part1+part2+part3+part4))

def norm_sq_corr2_SE(M_k_B, B_bar_mean, B_bar_cov):
  '''These are computed from the state evolution parameters'''
  T_k_B = M_k_B
  T_11 = T_k_B[0,0]
  T_12 = T_k_B[0,1]
  T_13 = T_k_B[0,2]
  T_21 = T_k_B[1,0]
  T_22 = T_k_B[1,1]
  T_23 = T_k_B[1,2]
  T_31 = T_k_B[2,0]
  T_32 = T_k_B[2,1]
  T_33 = T_k_B[2,2]

  C_1 = np.dot(B_bar_cov,np.dot(M_k_B.T,linalg.pinv(np.dot(np.dot(M_k_B,B_bar_cov), M_k_B.T)+T_k_B)))
  C_2 = np.dot(np.eye(3) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)

  C_2_1 = C_2[0]
  C_2_2 = C_2[1]
  C_2_3 = C_2[2]

  C_3_11 = C_3[0,0]
  C_3_12 = C_3[0,1]
  C_3_13 = C_3[0,2]
  C_3_21 = C_3[1,0]
  C_3_22 = C_3[1,1]
  C_3_23 = C_3[1,2]
  C_3_31 = C_3[2,0]
  C_3_32 = C_3[2,1]
  C_3_33 = C_3[2,2]

  C_1_11 = C_1[0,0]
  C_1_12 = C_1[0,1]
  C_1_13 = C_1[0,2]
  C_1_21 = C_1[1,0]
  C_1_22 = C_1[1,1]
  C_1_23 = C_1[1,2]
  C_1_31 = C_1[2,0]
  C_1_32 = C_1[2,1]
  C_1_33 = C_1[2,2]

  E_beta1_bar_sq = B_bar_cov[0,0]+B_bar_mean[0]**2
  E_beta2_bar_sq = B_bar_cov[1,1]+B_bar_mean[1]**2
  E_beta3_bar_sq = B_bar_cov[2,2]+B_bar_mean[2]**2
  E_beta1_bar_beta2_bar = B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1]
  E_beta1_bar_beta3_bar = B_bar_cov[0,2]+B_bar_mean[0]*B_bar_mean[2]
  E_beta2_bar_beta3_bar = B_bar_cov[1,2]+B_bar_mean[1]*B_bar_mean[2]

  num = C_2_2*B_bar_mean[1]+C_3_21*E_beta1_bar_beta2_bar+C_3_22*E_beta2_bar_sq+C_3_23*E_beta2_bar_beta3_bar

  part1 = C_2_2**2+2*C_2_2*C_3_21*B_bar_mean[0]+2*C_2_2*C_3_22*B_bar_mean[1]+2*C_2_2*C_3_23*B_bar_mean[2]
  part2 = (C_3_21**2)*E_beta1_bar_sq+2*C_3_21*C_3_22*E_beta1_bar_beta2_bar+2*C_3_21*C_3_23*E_beta1_bar_beta3_bar
  part3 = (C_3_22**2)*E_beta2_bar_sq+2*C_3_22*C_3_23*E_beta2_bar_beta3_bar+(C_3_23**2)*E_beta3_bar_sq
  part4 = (C_1_21**2)*T_11+2*C_1_21*C_1_22*T_12+2*C_1_21*C_1_23*T_13+(C_1_22**2)*T_22+2*C_1_22*C_1_23*T_23+(C_1_23**2)*T_33

  return (num**2) / (E_beta2_bar_sq * (part1+part2+part3+part4))

def norm_sq_corr3_SE(M_k_B, B_bar_mean, B_bar_cov):
  '''These are computed from the state evolution parameters'''
  T_k_B = M_k_B
  T_11 = T_k_B[0,0]
  T_12 = T_k_B[0,1]
  T_13 = T_k_B[0,2]
  T_21 = T_k_B[1,0]
  T_22 = T_k_B[1,1]
  T_23 = T_k_B[1,2]
  T_31 = T_k_B[2,0]
  T_32 = T_k_B[2,1]
  T_33 = T_k_B[2,2]

  C_1 = np.dot(B_bar_cov,np.dot(M_k_B.T,linalg.pinv(np.dot(np.dot(M_k_B,B_bar_cov), M_k_B.T)+T_k_B)))
  C_2 = np.dot(np.eye(3) - np.dot(C_1, M_k_B), B_bar_mean)
  C_3 = np.dot(C_1, M_k_B)

  C_2_1 = C_2[0]
  C_2_2 = C_2[1]
  C_2_3 = C_2[2]

  C_3_11 = C_3[0,0]
  C_3_12 = C_3[0,1]
  C_3_13 = C_3[0,2]
  C_3_21 = C_3[1,0]
  C_3_22 = C_3[1,1]
  C_3_23 = C_3[1,2]
  C_3_31 = C_3[2,0]
  C_3_32 = C_3[2,1]
  C_3_33 = C_3[2,2]

  C_1_11 = C_1[0,0]
  C_1_12 = C_1[0,1]
  C_1_13 = C_1[0,2]
  C_1_21 = C_1[1,0]
  C_1_22 = C_1[1,1]
  C_1_23 = C_1[1,2]
  C_1_31 = C_1[2,0]
  C_1_32 = C_1[2,1]
  C_1_33 = C_1[2,2]

  E_beta1_bar_sq = B_bar_cov[0,0]+B_bar_mean[0]**2
  E_beta2_bar_sq = B_bar_cov[1,1]+B_bar_mean[1]**2
  E_beta3_bar_sq = B_bar_cov[2,2]+B_bar_mean[2]**2
  E_beta1_bar_beta2_bar = B_bar_cov[0,1]+B_bar_mean[0]*B_bar_mean[1]
  E_beta1_bar_beta3_bar = B_bar_cov[0,2]+B_bar_mean[0]*B_bar_mean[2]
  E_beta2_bar_beta3_bar = B_bar_cov[1,2]+B_bar_mean[1]*B_bar_mean[2]

  num = C_2_3*B_bar_mean[2]+C_3_31*E_beta1_bar_beta3_bar+C_3_32*E_beta2_bar_beta3_bar+C_3_33*E_beta3_bar_sq

  part1 = C_2_3**2+2*C_2_3*C_3_31*B_bar_mean[0]+2*C_2_3*C_3_32*B_bar_mean[1]+2*C_2_3*C_3_33*B_bar_mean[2]
  part2 = (C_3_31**2)*E_beta1_bar_sq+2*C_3_31*C_3_32*E_beta1_bar_beta2_bar+2*C_3_31*C_3_33*E_beta1_bar_beta3_bar
  part3 = (C_3_32**2)*E_beta2_bar_sq+2*C_3_32*C_3_33*E_beta2_bar_beta3_bar+(C_3_33**2)*E_beta3_bar_sq
  part4 = (C_1_31**2)*T_11+2*C_1_31*C_1_32*T_12+2*C_1_31*C_1_33*T_13+(C_1_32**2)*T_22+2*C_1_32*C_1_33*T_23+(C_1_33**2)*T_33

  return (num**2) / (E_beta3_bar_sq * (part1+part2+part3+part4))

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

def run_matrix_GAMP(n, p, alpha_vec, sigma, X, Y, B, B_bar_mean, B_bar_cov, 
                    B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter):

  delta = n / p

  # Matrix-GAMP initializations
  R_hat_minus_1 = np.zeros((n,3))
  F_0 = np.eye(3)

  Sigma_0 = generate_Sigma_0(delta, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov)
  print('Sigma_0\n',Sigma_0)

  # Storage of GAMP variables from previous iteration
  Theta_k = np.zeros((n,3))
  R_hat_k_minus_1 = R_hat_minus_1
  B_hat_k = B_hat_0
  F_k = F_0

  # State evolution parameters
  M_k_B = np.zeros((3,3))
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
    R_hat_k = np.apply_along_axis(g_k_bayes_wrapper, 1, Theta_k_and_Y, Sigma_k, alpha_vec, sigma)

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
    Sigma_k_plus_1 = np.zeros((6,6))
    Sigma_k_plus_1[0:3,0:3] = Sigma_k[0:3,0:3]
    temp_matrix = np.dot(B_hat_k_plus_1.T, B_hat_k_plus_1) / p
    Sigma_k_plus_1[0:3,3:6] = temp_matrix / delta
    Sigma_k_plus_1[3:6,0:3] = temp_matrix / delta
    Sigma_k_plus_1[3:6,3:6] = temp_matrix / delta

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

''' Plotting norm sq corr vs delta (GAMP vs SE) for covariances for prior '''
def run_GAMP_v_SE_multi_delta_multi_cov(p, n_list, alpha_vec, B_bar_mean, B_bar_cov, 
                                        B_hat_0_row_mean, B_hat_0_row_cov, sigma, num_iter, num_runs):
  
  num_deltas = len(n_list)

  mean_final_corr1_list = np.zeros(num_deltas)
  mean_final_corr2_list = np.zeros(num_deltas)
  mean_final_corr3_list = np.zeros(num_deltas)
  var_final_corr1_list = np.zeros((num_runs, num_deltas))
  var_final_corr2_list = np.zeros((num_runs, num_deltas))
  var_final_corr3_list = np.zeros((num_runs, num_deltas))

  mean_final_corr1_list_SE = np.zeros(num_deltas)
  mean_final_corr2_list_SE = np.zeros(num_deltas)
  mean_final_corr3_list_SE = np.zeros(num_deltas)
  var_final_corr1_list_SE = np.zeros((num_runs, num_deltas))
  var_final_corr2_list_SE = np.zeros((num_runs, num_deltas))
  var_final_corr3_list_SE = np.zeros((num_runs, num_deltas))

  for n_index in range(len(n_list)):
    n = n_list[n_index]
    final_corr1 = 0
    final_corr2 = 0
    for run_num in range(num_runs):
      print('=== Run number: ' + str(run_num + 1) + ' ===')

      np.random.seed(run_num) # so that result is reproducible
      
      B = multivariate_normal(B_bar_mean, B_bar_cov, p)
      B_hat_0 = multivariate_normal(B_hat_0_row_mean, B_hat_0_row_cov, p)

      X = normal(0, np.sqrt(1/n), (n, p))
      Theta = np.dot(X, B)

      # Generating Y: We used some numpy operational trick to avoid writing 
      # a for loop (inefficient) to compute Y.
      c = multinomial(1, alpha_vec, n)
      eps = normal(0, sigma, n)
      Y = (Theta * np.c_[c[:,0], c[:,1], c[:,2]]).sum(1) + eps

      B_hat_storage, M_k_B_storage = run_matrix_GAMP(n, p, alpha_vec, sigma, X, Y, B, B_bar_mean, B_bar_cov, 
                                                     B_hat_0, B_hat_0_row_mean, B_hat_0_row_cov, num_iter)
      num_iter_ran = len(B_hat_storage)

      # GAMP
      beta1 = B[:, 0]
      beta2 = B[:, 1]
      beta3 = B[:, 2]
      B_hat = B_hat_storage[num_iter_ran - 1]
      beta1_hat = B_hat[:, 0]
      beta2_hat = B_hat[:, 1]
      beta3_hat = B_hat[:, 2]

      norm_sq_corr1 = norm_sq_corr(beta1, beta1_hat)
      mean_final_corr1_list[n_index] += norm_sq_corr1
      var_final_corr1_list[run_num][n_index] = norm_sq_corr1

      norm_sq_corr2 = norm_sq_corr(beta2, beta2_hat)
      mean_final_corr2_list[n_index] += norm_sq_corr2
      var_final_corr2_list[run_num][n_index] = norm_sq_corr2

      norm_sq_corr3 = norm_sq_corr(beta3, beta3_hat)
      mean_final_corr3_list[n_index] += norm_sq_corr3
      var_final_corr3_list[run_num][n_index] = norm_sq_corr3

      # State evolution
      M_k_B = M_k_B_storage[num_iter_ran - 2] # -2 because there is one less M_k_B than B_hat (due to initialization)
      norm_sq_corr1 = norm_sq_corr1_SE(M_k_B, B_bar_mean, B_bar_cov)
      mean_final_corr1_list_SE[n_index] += norm_sq_corr1
      var_final_corr1_list_SE[run_num][n_index] = norm_sq_corr1

      norm_sq_corr2 = norm_sq_corr2_SE(M_k_B, B_bar_mean, B_bar_cov)
      mean_final_corr2_list_SE[n_index] += norm_sq_corr2
      var_final_corr2_list_SE[run_num][n_index] = norm_sq_corr2

      norm_sq_corr3 = norm_sq_corr3_SE(M_k_B, B_bar_mean, B_bar_cov)
      mean_final_corr3_list_SE[n_index] += norm_sq_corr3
      var_final_corr3_list_SE[run_num][n_index] = norm_sq_corr3

  mean_final_corr1_list = mean_final_corr1_list / num_runs
  mean_final_corr2_list = mean_final_corr2_list / num_runs
  mean_final_corr3_list = mean_final_corr3_list / num_runs

  mean_final_corr1_list_SE = mean_final_corr1_list_SE / num_runs
  mean_final_corr2_list_SE = mean_final_corr2_list_SE / num_runs
  mean_final_corr3_list_SE = mean_final_corr3_list_SE / num_runs

  print('mean_final_corr1_list\n',mean_final_corr1_list)
  print('mean_final_corr2_list\n',mean_final_corr2_list)
  print('mean_final_corr3_list\n',mean_final_corr3_list)

  print('mean_final_corr1_list_SE\n',mean_final_corr1_list_SE)
  print('mean_final_corr2_list_SE\n',mean_final_corr2_list_SE)
  print('mean_final_corr3_list_SE\n',mean_final_corr3_list_SE)

  SD_final_corr1_list = np.sqrt(np.sum(np.square(var_final_corr1_list - mean_final_corr1_list), axis=0) / num_runs)
  SD_final_corr2_list = np.sqrt(np.sum(np.square(var_final_corr2_list - mean_final_corr2_list), axis=0) / num_runs)
  SD_final_corr3_list = np.sqrt(np.sum(np.square(var_final_corr3_list - mean_final_corr3_list), axis=0) / num_runs)

  SD_final_corr1_list_SE = np.sqrt(np.sum(np.square(var_final_corr1_list_SE - mean_final_corr1_list_SE), axis=0) / num_runs)
  SD_final_corr2_list_SE = np.sqrt(np.sum(np.square(var_final_corr2_list_SE - mean_final_corr2_list_SE), axis=0) / num_runs)
  SD_final_corr3_list_SE = np.sqrt(np.sum(np.square(var_final_corr3_list_SE - mean_final_corr3_list_SE), axis=0) / num_runs)

  print('SD_final_corr1_list\n',SD_final_corr1_list)
  print('SD_final_corr2_list\n',SD_final_corr2_list)
  print('SD_final_corr3_list\n',SD_final_corr3_list)

  print('SD_final_corr1_list_SE\n',SD_final_corr1_list_SE)
  print('SD_final_corr2_list_SE\n',SD_final_corr2_list_SE)
  print('SD_final_corr3_list_SE\n',SD_final_corr3_list_SE)

  return [mean_final_corr1_list, mean_final_corr2_list, mean_final_corr3_list, 
          mean_final_corr1_list_SE, mean_final_corr2_list_SE, mean_final_corr3_list_SE, 
          SD_final_corr1_list, SD_final_corr2_list, SD_final_corr3_list]

p = 500
n_list = [int(5*p), int(5.5*p), int(6*p), int(6.5*p), int(7*p), int(7.5*p), int(8*p), int(8.5*p), int(9*p)]
alpha_vec = [1/3, 1/3, 1/3]
num_iter = 10
num_runs = 10

sigma = 0
B_bar_mean = np.array([0, 0.5, 1])
B_bar_cov = np.eye(3)
B_hat_0_row_mean = B_bar_mean
B_hat_0_row_cov = B_bar_cov

output_list = run_GAMP_v_SE_multi_delta_multi_cov(p, n_list, alpha_vec, B_bar_mean, B_bar_cov, B_hat_0_row_mean, B_hat_0_row_cov, sigma, num_iter, num_runs)
save('diff_mean_same_prop', np.array(output_list))