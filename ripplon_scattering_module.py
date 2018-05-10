from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy    

unit_M = 1
unit_D = 1
unit_E = 1
unit_t = 1
e_charge = 1
initialized = False

def __init__():
    """
    Initialize units
    """
    pass

def init(unit_M_, unit_D_, unit_E_):
    """
    Initialize units
    """
    global initialized
    global unit_M, unit_D, unit_E, unit_t, e_charge
    unit_M = unit_M_
    unit_D = unit_D_
    unit_E = unit_E_
    unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # in s
    #Charge through Gaussian units:
    unit_Q = np.sqrt(unit_E*1e7*unit_D*1e2) # Coulombs
    unit_Qe = unit_Q/4.8032068e-10 # e, unit charge in units of elementary charge e
    e_charge = 1/unit_Qe # electron charge in units of unit_Q
    initialized = True


def phi_1(x):
    """Function phi(x) for x < 1
    """
    if len(np.where(x >= 1)[0]) > 0:
        raise ValueError("argument of phi_1 must be x < 1")
    A = 1 - x**2
    return -1/A + A**(-1.5)*np.log((1 + A**0.5)/x)

def phi_2(x):
    """Function phi(x) for x > 1
    """
    if len(np.where(x <= 1)[0]) > 0:
        raise ValueError("argument of phi_2 must be x > 1")
    B = x**2 - 1
    return 1/B - B**(-1.5) * np.arctan(np.sqrt(B))
    
def phi_x(x):
    """
    Function phi(x) from Dykman, M. I., et.al.
    Phys. Rev. B 55.24 (1997): 16249.
    Equations (89)-(90)
    """
    res = np.zeros(x.shape)
    g1_ind = np.where(x > 1)[0]
    l1_ind = np.where(x < 1)[0]
    eq1_ind = np.where(x == 1)[0]
    res[l1_ind] = phi_1(x[l1_ind])
    res[g1_ind] = phi_2(x[g1_ind])
    res[eq1_ind] = phi_1(0.9999)*np.ones(eq1_ind.shape)
    return res


def w_theta(N_theta, T, k):
    """ Calculate scattering rate vs angle w(\theta)
    \param N_theta number of theta points spanning [0, 2\pi]
    \param T temperature in hoomd units (K)
    \param k electron wavevector in hoomd units (1/micron)
    
    return w(theta) and theta arrays, each of size N_theta
    
    Only polarization part of the interaction is taken. Pressing field leads to forward scattering, so we want to ignore it.
    Dykman, M. I., et.al. Phys. Rev. B 55.24 (1997): 16249.
    Equations (89)-(90)
    """
    if not initialized:
        raise RuntimeError('RSM module not initialized')
    #theta_arr = np.linspace(2*np.pi/N_theta, 2*np.pi*(1 - 1/N_theta), N_theta)
    theta_arr = np.linspace(0, 2*np.pi, N_theta)
    
    #prm = 1.057 #permittivity of helium
    prm = 1.06
    m_e = 1
    hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
    alpha = 0.37*1e-3/unit_E*unit_D**2 # 0.37 erg/cm^2
    Lambda_0 = 0.25*e_charge**2*(prm - 1)/(prm + 1)
    lmbd = hbar**2/m_e/Lambda_0
    q_arr = k*np.sqrt(2*(1 - np.cos(theta_arr)))
    w_arr_res = np.zeros(theta_arr.shape)
    # exclude theta=0 and 2\pi to avoid dividing by zero in calculating w(\theta)
    w_arr_res[1:-1] = T*hbar/(8*np.pi*alpha*m_e*lmbd**2)*(q_arr[1:-1]*phi_x(0.5*q_arr[1:-1]*lmbd))**2
    return w_arr_res, theta_arr


def compute_w_k(k_arr, T, N_theta):
    """
    Compute scattering probability for each k from k_arr
    \param T temperature in hoomd units
    \param N_theta number of theta-points
    
    return w_k_res - total scattering rate vs k (array of size N_k)
           w_k_theta - 2D array, each row w_k_theta[i,:] is w(theta) distribution for k_i
    """
    if not initialized:
        raise RuntimeError('RSM module not initialized')
    N_k = len(k_arr)
    w_k_res = np.zeros(k_arr.shape)
    w_k_theta = np.zeros((N_k, N_theta)) # w_theta_k[i,j] = w(k_i, \theta_j) 
    for i,k in enumerate(k_arr):
        w_arr, theta_arr = w_theta(N_theta, T, k)
        w_k_res[i] = np.sum(w_arr)*2*np.pi/N_theta
        w_k_theta[i, :] = w_arr
    return w_k_res, w_k_theta

def compute_cumul_W(w_k_theta, w_k, N_W):
    """Compute cumulative scattering probability W(\theta, k) vs angles for all k, and resample W and \theta
    so that W is uniformly sampled from on [0,1]
    
    \param w_k_theta 2D array of shape (N_k, N_theta), with i-th row representing w_{k[i]}(theta)
    \param w_k 1D array of size N_k, total scattering rate for each k
    \param N_W Number of points for uniform resampling of W
    
    return W_resampled array, shape (N_W,), resampling points from 0 to 1 (including ends)
           theta_resampled array, shape (N_k, N_W), each row i being an array of new theta sampling for k[i]
           W_cumul array (N_k, N_theta), cumulative distribution W before resampling 
    """
    if not initialized:
        raise RuntimeError('RSM module not initialized')
    theta_arr = np.linspace(0, 2*np.pi, w_k_theta.shape[1])
    # Essentially simps integration:
    w_k_theta_mid = 0.5*(w_k_theta + np.roll(w_k_theta, 1, axis=1))
    W_cumul = np.cumsum(w_k_theta_mid, axis = 1)/np.reshape(w_k, (len(w_k), 1))*2*np.pi/w_k_theta.shape[1]
    #resample W_cumul uniformly from 0 to 1 to ease the function inversion:
    W_resampled = np.linspace(0, 1, N_W)
    theta_resampled = np.zeros((w_k_theta.shape[0], N_W))
    for i in range(w_k_theta.shape[0]):
        theta_resampled[i, :] = np.interp(W_resampled, W_cumul[i,:], theta_arr)
    
    return W_resampled, theta_resampled, W_cumul

def scattering_parameters(T, k_min, k_max, N_k, N_W, N_theta):
    if not initialized:
        raise RuntimeError("Units not initialized (use init to initialize)")
    m_e = 1
    hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
    vmin = hbar*k_min/m_e
    vmax = hbar*k_max/m_e
    
    k_arr = np.linspace(k_min, k_max, N_k) # k=1--500 correspoonding to T from 0.0005 to 100 (k=1 is quantum threshold as well)
    w_k, w_k_theta = compute_w_k(k_arr, T, N_theta)
    W_resampled, theta_resampled, W_cumul = compute_cumul_W(w_k_theta, w_k, N_W)
    return w_k, theta_resampled, W_cumul, vmin, vmax

def tau_rec_many_e(T, Nq=1000):
    """ Calculate \tau^-1, one over relaxation time, for many-electron correlated liquid
    \param T temperature in hoomd units (K)
    \param k electron wavevector in hoomd units (1/micron)
    
    return w(theta) and theta arrays, each of size N_theta
    
    Only polarization part of the interaction is taken. Pressing field leads to forward scattering, so we want to ignore it.
    Dykman, M. I., et.al. Phys. Rev. B 55.24 (1997): 16249.
    Equations (89)-(90)
    """
    #theta_arr = np.linspace(2*np.pi/N_theta, 2*np.pi*(1 - 1/N_theta), N_theta)
    #theta_arr = np.linspace(0, 2*np.pi, N_theta)
    
    #prm = 1.057 #permittivity of helium
    prm = 1.06
    m_e = 1
    hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
    alpha = 0.37*1e-3/unit_E*unit_D**2 # 0.37 erg/cm^2
    Lambda_0 = 0.25*e_charge**2*(prm - 1)/(prm + 1)
    lmbd = hbar**2/m_e/Lambda_0
    q_arr = np.linspace(0.000001, 600, Nq)
    
    integrand = q_arr**4*(phi_x(0.5*q_arr*lmbd))**2*np.exp(-hbar**2*q_arr**2/8/m_e/T)
    tau_rec = hbar**4/(16*alpha*m_e**2*lmbd**2*np.sqrt(2*np.pi*m_e*T))*scipy.integrate.simps(integrand, x=q_arr)
    return tau_rec
    
def relaxation_tau_rec(k_arr, T, N_theta):
    """
    Inverse relaxation time for each k in k_arr in one-electron picture.
    \param T temperature in hoomd units
    \param N_theta number of theta-points
    
    return tau^-1(k_arr)
    """
    N_k = len(k_arr)
    tau_rec = np.zeros(k_arr.shape) 
    for i,k in enumerate(k_arr):
        w_arr, theta_arr = w_theta(N_theta, T, k)
        tau_rec[i] = np.sum(w_arr*(1 - np.cos(theta_arr)))*2*np.pi/N_theta
    return tau_rec
    
def mu_many_e(T):
    """ Mobility of correlated many-electron system
    """
    m_e = 1
    tau_rec_me = tau_rec_many_e(T, Nq=5000)
    mu_me = e_charge/tau_rec_me/m_e
    return mu_me

def mu_one_e(T):
    """ Mobility of in a single-electron (nonintaracting) picture
    """
    Nk = 1000
    N_theta = 400
    hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
    k_arr = np.linspace(0.01, 400, Nk)
    tau_rec = relaxation_tau_rec(k_arr, T, N_theta)
    p_arr = hbar*k_arr
    dp = p_arr[1] - p_arr[0]
    integral = np.sum(p_arr**3/tau_rec*np.exp(-p_arr**2/(2*T)))*dp
    mu = e_charge*integral/(2*T**2)
    return mu
    
    
def write_to_file(k_min, k_max, w_k, N_W, theta_resampled, path = ''):
    """ Write k_min, k_max, w_k, N_W, theta_resampled to a file for HOOMD custom_scatter integrator.
    
    Not implemented
    """
    raise NotImplementedError("write_to_file is not implemented")