from __future__ import division
from scipy.special import kn
import numpy as np
import scipy
import pickle
import os
import time


unit_M = 1
unit_D = 1
unit_E = 1
unit_t = 1
e_charge = 1
hbar = 1
m_e = 1
Lambda_0 = 1
r_b = 1
vs = 1
rho = 1
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
    global unit_M, unit_D, unit_E, unit_t, e_charge, hbar, m_e
    global Lambda_0, r_b, vs, rho
    unit_M = unit_M_
    unit_D = unit_D_
    unit_E = unit_E_
    unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # in s
    #Charge through Gaussian units:
    unit_Q = np.sqrt(unit_E*1e7*unit_D*1e2) # Coulombs
    unit_Qe = unit_Q/4.8032068e-10 # e, unit charge in units of elementary charge e
    e_charge = 1/unit_Qe # electron charge in units of unit_Q
    hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
    m_e = 9.10938356e-31/unit_M
    prm = 1.057 #permittivity of helium
    Lambda_0 = 0.25*e_charge**2*(prm - 1)/(prm + 1)
    r_b = hbar**2/m_e/Lambda_0
    vs = 2.4e2/unit_D*unit_t
    rho = 0.145e3/unit_M*unit_D**3
    
    initialized = True
    

def zmelement(q, Qz, r_b, z_max, zp_max, N_z=500):
    hz = z_max/(N_z - 1)
    hzp = zp_max/(N_z - 1)
    epz = hz/1000 #starting point for z-integration
    epzp = hzp/1000 #starting point for z-prime integration

    z = np.linspace(epz, z_max, N_z)
    zp = np.linspace(epzp, zp_max, N_z)

    int_p = np.zeros(zp.shape)
    for i, zpi in enumerate(zp):
        #int_p[i] = np.sum(z**2/(z + zpi)*np.exp(-2*z/r_b)*np.sin(Qz*zpi)*kn(1, q*(z + zpi)) )*hz
        int_p[i] = scipy.integrate.simps(z**2/(z + zpi)*np.exp(-2*z/r_b)*np.sin(Qz*zpi)*kn(1, q*(z + zpi)), z)

    #melement = 4/r_b**3*np.sum(int_p)*hzp
    melement = 4/r_b**3*scipy.integrate.simps(int_p, zp)
    return melement


def bare_rate_theta(k, kp, Ntheta, N_z):
    """ Calculate phonon scattering rate matrix element (scattering rate without thermal factors NQ of NQ+1)
    vs scattering angle theta    ***for a given pair***    of k and k'.
    
    k magnitude of initial vector
    kp magnitude of scattering vector k'
    Ntheta number of points to sample theta, sampling includes ends 0 and 2\pi
    
    return array of shape (Ntheta,)
    """
    theta = np.linspace(0, 2*np.pi, Ntheta)
    kpx = kp*np.cos(theta)
    kpy = kp*np.sin(theta)
    qx = kpx - k
    qy = kpy

    q_arr = np.sqrt(qx**2 + qy**2)
    Q_arr = 0.5*hbar/vs/m_e*np.abs(k**2 - kpx**2 - kpy**2)
    ind1 = np.where(Q_arr**2 - q_arr**2 > 0)[0]
    ind2 = np.where(Q_arr**2 - q_arr**2 <= 0)[0]
    Qz_arr = np.zeros(Q_arr.shape)
    Qz_arr[ind1] = np.sqrt(Q_arr[ind1]**2 - q_arr[ind1]**2)
    Qz_arr[ind2] = 0

    melement_arr = np.zeros(theta.shape)
    for i in ind1:
        melement_arr[i] = zmelement(q_arr[i], Qz_arr[i], r_b, z_max = 5*r_b, zp_max = 5/(q_arr[i] + k*0.1), N_z=N_z)
    coeff = Lambda_0**2/(2*np.pi**2*rho*hbar*vs**2)
    dw_bare = coeff*Q_arr*q_arr**2*melement_arr**2
    return dw_bare

def compute_bare_fkkp_t(k_arr, Ntheta, N_z):
    """ Calculate phonon scattering rate matrix element (scattering rate without thermal factors NQ of NQ+1)
    vs scattering angle theta    ***for all***   k and k' in k_arr.
    """
    if not initialized:
        raise RuntimeError("Units not initialized")
    Nk = len(k_arr)
    fkkpt = np.zeros((Nk, Nk, Ntheta))
    start = time.time()
    for i,k in enumerate(k_arr):
        for j,kp in enumerate(k_arr):
            print('Calculating i,j=({},{}), Nk = {}, t = {} s'.format(i,j,Nk, time.time()))
            fkkpt[i,j, :] = bare_rate_theta(k, kp, Ntheta, N_z)
    end = time.time()
    print('Elapsed time = {} s'.format(end - start))
    print('{} s per point'.format((end-start)/Nk**2))
    return fkkpt

def compute_bare_fkkp(fkkp_t, k_arr):
    """ Integrate ffkp_t over angles to get fkkp
    """
    Ntheta = fkkp_t.shape[2]
    dtheta = 2*np.pi/(Ntheta - 1)
    fkkp_t_mid = 0.5*(fkkp_t + np.roll(fkkp_t, 1, axis= 2))
    fkkp_t_mid[:,:,0] = 0
    fkkp = np.sum(fkkp_t_mid, axis = 2)*dtheta*np.reshape(k_arr, (1,-1))
    # Symmetrize fkkp artificially, because it is not symmetric due to small numerical errors
    k_arr_col = np.reshape(k_arr, (-1,1))
    k_arr_row = np.reshape(k_arr, (1,-1))
    Omega_matrix = np.dot(1/k_arr_col, k_arr_row)
    rhs = np.transpose(fkkp)*Omega_matrix
    upper_ind = np.triu_indices(fkkp.shape[0])
    fkkp[upper_ind] = rhs[upper_ind]
    return fkkp

def compute_Yinv(fkkp_t, NY):
    """ Integrate ffkp_t over angles to find the cumulative angle distributions Y and their inverse Yinv.
    NY number of points to interpolate Yinv.
    """
    Ntheta = fkkp_t.shape[2]
    Nk = fkkp_t.shape[0]
    dtheta = 2*np.pi/(Ntheta - 1)
    theta = np.linspace(0, 2*np.pi, Ntheta)
    fkkp_t_mid = 0.5*(fkkp_t + np.roll(fkkp_t, 1, axis= 2))
    fkkp_t_mid[:,:,0] = 0
    fkkp = np.sum(fkkp_t_mid, axis = 2)*dtheta
    ind_nonzero = np.where(fkkp != 0)
    ind_zero = np.where(fkkp == 0)
    fkkp = np.reshape(fkkp, fkkp.shape + (1,))
    #Y = np.zeros(fkkp_t.shape)
    
    #Uniform distribution for points with zero scattering rate
    uniform_y = np.linspace(0, 1, Ntheta)
    uniform_y = np.reshape(uniform_y, (1,1) + uniform_y.shape)
    Y = np.tile(uniform_y, fkkp.shape)

    Y[ind_nonzero[0], ind_nonzero[1], :] = np.cumsum(fkkp_t_mid, axis = 2)[ind_nonzero[0], ind_nonzero[1], :]*dtheta\
    /fkkp[ind_nonzero[0], ind_nonzero[1], :]
    
    
    resampling_arr = np.linspace(0, 1, NY)
    Y_inv = np.zeros(fkkp.shape + (NY,))
    for i in range(Nk):
        for j in range(Nk):
            Y_inv[i, j, :] = np.interp(resampling_arr, Y[i,j,:], theta)
    return Y_inv
    

def dress_fkkp(fkkp, k_arr, T):
    """Multiply fkkpt with NQ if phonon absorbed or NQ+1 if phonon emitted
    """
    if not initialized:
        raise RuntimeError("Units not initialized")
    Nk = len(k_arr)
    
    KP, K = np.meshgrid(k_arr, k_arr)
    ksqd = (KP**2 - K**2)
    ind_nonzero = np.where(ksqd != 0)
    EQ = hbar**2/(2*m_e)*np.abs(ksqd)
    NQ = np.zeros(EQ.shape)
    NQ[ind_nonzero[0], ind_nonzero[1]] = 1/(np.exp(EQ[ind_nonzero[0], ind_nonzero[1]]/T) - 1)
    ind_emit = np.where(ksqd < 0)
    ind_absorb = np.where(ksqd > 0)
    wkkp = np.zeros(fkkp.shape)
    wkkp[ind_emit[0], ind_emit[1]] = fkkp[ind_emit[0], ind_emit[1]] * (NQ[ind_emit[0], ind_emit[1]] + 1)
    wkkp[ind_absorb[0], ind_absorb[1]] = fkkp[ind_absorb[0], ind_absorb[1]] * NQ[ind_absorb[0], ind_absorb[1]]
    return wkkp


def compute_total_wk(fkkp, k_arr):
    """ Compute total scattering rate from each k into all other states, by integrating fkkp over kp
    
    return: wk of the same size as k_arr
    """
    wk = np.zeros(k_arr.shape)
    for i,k in enumerate(k_arr):
        wk[i] = scipy.integrate.simps(fkkp[i,:], k_arr)
    return wk
    
def compute_cumulative_Fkkp_inv(fkkp, k_arr, tol = 0.01):
    """ Compute cumulative distributions over kp for scattering from k to kp.
    fkkp scattering rates table for scattering from any k to any kp
    tol*delta(F_inv_sampling) is the tolerance for the tail of distribution
    return: F_inv of shape fkkp.shape, where the cumulative distribution for scattering from k_i is F_inv[i,:]
    """
    F = np.zeros(fkkp.shape)
    F_inv = np.zeros(fkkp.shape)
    # Cumulatively integrate fkkp over kp
    fkkp_mid = 0.5*(fkkp + np.roll(fkkp, 1, axis= 1))
    fkkp_mid[:,0] = 0
    F = np.cumsum(fkkp_mid, axis = 1)
    #Normalize F
    F /= np.reshape(F[:,-1], (-1,1))
    F_inv_sampling = np.linspace(0,1, len(k_arr))
    
    for i in range(len(k_arr)):
        F_inv[i,:] = np.interp(F_inv_sampling, F[i,:], k_arr)
        #Cut off the vertical line at the edge of F_inv
        threshold = np.where(F[i,:] > 1 - F_inv_sampling[1]*tol)[0][0]
        F_inv[i,-1] = k_arr[threshold]
    return F_inv
        
    
def write_fkkp_t_to_file(fkkp_t, kmin, kmax, Nk, Ntheta, Nz, dir_path):
    data_dict = {'fkkp_t':fkkp_t, 'kmin':kmin, 'kmax':kmax, 'Nk':Nk, 'Ntheta':Ntheta, 'Nz':Nz}
    #Check consistency
    if fkkp_t.shape != (Nk, Nk, Ntheta):
        raise RuntimeError("fkkp_t.shape not consistent with provided Nk and Ntheta")
    fname = 'fkkpt_kmin{:.0f}_kmax{:.0f}_Nk{:d}_Ntheta{:d}.dat'.format(kmin, kmax, Nk, Ntheta)
    
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'   
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    with open(dir_path + fname, 'wb') as f:
        pickle.dump(data_dict, f)
    print('Data saved to {}'.format(dir_path + fname))

def read_fkkp_t_from_file(fname):
    with open(fname, 'rb') as f:
        data_dict = pickle.load(f)
    fkkp_t = data_dict['fkkp_t']
    kmin = data_dict['kmin']
    kmax = data_dict['kmax']
    print('Read successfully from {}'.format(fname))
    return fkkp_t, kmin, kmax
            
