from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import gsd
import gsd.fl
import numpy as np
import scipy.integrate
import os
import sys
import datetime
import time
import pickle
from shutil import copyfile
from hoomd.data import boxdim
import md_tools27 as md_tools
import inspect
import ripplon_scattering_module as rsm

"""
This is a version of plotIV.py that also plots the fraction of 6-neighbored particles as a function of A.
"""


def relaxation_tau(k_arr, T, N_theta):
    """
    Compute scattering probability for each k from k_arr
    \param T temperature in hoomd units
    \param N_theta number of theta-points
    
    return w_k_res - total scattering rate vs k (array of size N_k)
           w_k_theta - 2D array, each row w_k_theta[i,:] is w(theta) distribution for k_i
    """
    N_k = len(k_arr)
    tau_rec = np.zeros(k_arr.shape) 
    for i,k in enumerate(k_arr):
        w_arr, theta_arr = rsm.w_theta(N_theta, T, k)
        tau_rec[i] = np.sum(w_arr*(1 - np.cos(theta_arr)))*2*np.pi/N_theta
    return tau_rec

def read_log(path):
    coulomb_status = ''
    T_gamma = None
    with open(path + '/log.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                timestamp = line.rstrip()
            if line[:10] == '# Periodic':
                words = line.split(' ')
                p = int(words[9])
                A = float(words[6])
            if line[:4] == '# a ':
                words = line.split(' ')
                repeat_x = int(words[6])
                repeat_y = int(words[9])
                Np = 2*repeat_x*repeat_y
            if line[:7] == '# Gamma':
                words = line.split(' ')
                try:
                    gamma = float(words[2])
                    T = float(words[5])
                except:
                    gamma = '--'
                    T = '--'
            if line[:9] == '# Coulomb':
                words = line.split(' ')
                coulomb_status = words[-1]
            if line[:9] == '# T_gamma':
                words = line.split(' ')
                T_gamma = float(words[3])
        if gamma != '--' and T != '--' and T_gamma == None:
            T_gamma = T*gamma
    return {'timestamp': timestamp, 'p':p, 'A':A, 'Np': Np, 'gamma':gamma, 'T':T, 'coulomb_status':coulomb_status, \
            'T_gamma':T_gamma}

def fraction_six(fpath):
    '''
    Calculate the fraction of 6-coordinated particles (averaged over useframes), and average psi_6 order parameter (for the first frame).
    '''
    print(fpath)
    useframes = 10
    psiframes = 2
    psi6_av = 0
    num6 = 0
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])
        N = f_gsd.read_chunk(frame=0, name='particles/N')
        n_frames_total = f_gsd.nframes
        for iframe in range(useframes):
            pos = f_gsd.read_chunk(frame = iframe, name = 'particles/position')
            neighbor_list, neighbor_num = md_tools.find_neighbors_delone(pos, box)
            num6 += len(np.where(neighbor_num == 6)[0])
            if iframe < psiframes:
                psi6_av += np.mean(np.real(md_tools.psi_order_delone(pos, box)))
    psi6_av /= psiframes
    return num6/N/useframes, psi6_av


def OLS(x, y):
    '''OLS: x must be a vertical two-dimensional array'''
    X = np.hstack((np.reshape(np.ones(x.shape[0]), (-1,1)), x))#.transpose()
    Xpr = X.transpose()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(Xpr, X)), Xpr), y)
    #Estimate errors
    sigma_sq = np.dot(y - np.dot(X, beta), y - np.dot(X, beta))/(len(y) - 1.)
    sigma_beta_sq = sigma_sq*np.linalg.inv(np.dot(Xpr, X))
    return beta, sigma_beta_sq # = [f_0, df/d(A^2)]

def OLS_1(x, y):
    '''OLS: x must be a vertical two-dimensional array'''
    X = x#.transpose()
    Xpr = X.transpose()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(Xpr, X)), Xpr), y)
    #Estimate errors
    sigma_sq = np.dot(y - np.dot(X, beta), y - np.dot(X, beta))/(len(y) - 1.)
    sigma_beta_sq = sigma_sq*np.linalg.inv(np.dot(Xpr, X))
    return beta, sigma_beta_sq # = [f_0, df/d(A^2)]


def bootstrap_sterr(x, B=100):
    """
    Estimate standard error of \bar{x} by drawing B samples with replacement,
    calculating \bar{x}_b for each sample, and then calculating variance of them.
    """
    N = len(x)
    samples = np.zeros((B, N))
    mus = np.zeros((B,))
    for b in range(B):
        samples[b,:] = np.random.choice(x, N, replace=True)
        mus[b] = np.mean(samples[b,:])
    return np.std(mus)

def print_help():
    print('This script plots transport vs A, p, or T for data taken in transport measurements.')
    print('===========================================================')
    print('Usage: python plotIV.py mobility_data/a32x32_* [--options]')
    print('This will process all folders that match mobility_data/a32x32_*')
    print('===========================================================')
    print('Options:')
    print('\t--xonly will show only v_x on v-E curves')
    print('\t--showtext will print text info on the plots')
    print('\t--help or -h will print this help')


#Scattering parameters, for theoretical mobility
kmin = 1
kmax = 150
Nk = 200
k_arr = np.linspace(kmin, kmax, Nk)
Ntheta = 500

curr_fname = inspect.getfile(inspect.currentframe())
curr_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

## ==========================================
# Parse args

   
## =======================================================================
# Units
unit_M = 9.10938356e-31 # kg, electron mass
unit_D = 1e-6 # m, micron
unit_E = 1.38064852e-23 # m^2*kg/s^2
unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # = 2.568638150515e-10 s
epsilon_0 = 8.854187817e-12 # F/m = C^2/(J*m), vacuum permittivity
hbar = 1.0545726e-27/(unit_E*1e7)/unit_t
m_e = 9.10938356e-31/unit_M
unit_Q = np.sqrt(unit_E*1e7*unit_D*1e2) # Coulombs
unit_Qe = unit_Q/4.8032068e-10 # e, unit charge in units of elementary charge e
e_charge = 1/unit_Qe # electron charge in units of unit_Q

rsm.init(unit_M, unit_D, unit_E)
##=======================================================================
# Make a list of folders we want to process
folder_list = []
plot_xonly = False
show_text = False
plot_xonly_str = ''
for i in range(len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        folder_list.append(sys.argv[i])
    if sys.argv[i] == '--xonly':
        plot_xonly = True
        plot_xonly_str = '_xonly_'
    if sys.argv[i] == '--showtext':
        show_text = True
    if sys.argv[i] == '--help' or sys.argv[i] == '-h':
        print_help()
        exit()


# Make a list of subfolders A0.00... in each folders
subfolder_lists = []
for folder in folder_list:
    sf_list = []
    for item in os.walk(folder):
        # subfolder name and contained files
        sf_list.append((item[0], item[2]))
    sf_list = sf_list[1:]
    sf_list.sort()
    subfolder_lists.append(sf_list)

mode = ''
##=======================================================================
#calculate current and Teff for all files and plot the results:
for ifold, folder in enumerate(folder_list):
    print('==========================================================')
    print(folder)
    print('==========================================================')
    mu_arr = np.zeros(len(subfolder_lists[ifold]))
    sigma_mu_arr = np.zeros(len(subfolder_lists[ifold]))
    # X may be A, p, or G
    X_arr = np.zeros(len(subfolder_lists[ifold]))
    frac6_Xlist = [] #list of arrays for fraction of 6-coordinated particles
    psi6_Xlist = [] #list of arrays for psi6 order parameter
    for isf, sf in enumerate(subfolder_lists[ifold]):
        sf_words = sf[0].split('/')
        print(sf_words[-1])
        X = float(sf_words[-1][1:])
        X_arr[isf] = X
        log_data = read_log(sf[0])
        A = log_data['A']
        p = log_data['p']
        T_gamma = log_data['T_gamma']
        gamma = log_data['gamma']
        if sf_words[-1][0] == 'A':
            mode = 'A'
            A = X
        elif sf_words[-1][0] == 'p':
            mode = 'p'
            p = X
        elif sf_words[-1][0] == 'G':
            mode = 'G'
            gamma = X
        else:
            raise ValueError("Expected subfolder name to start with `A`, `p`, or `G`, in {}".format(fname))
        NE = 0
        fnames = []
        for fname in sf[1]:
            if fname[0] == 'E':
                fnames.append(fname)
                NE += 1
        fnames.sort()
        print(fnames)
        E_list = np.zeros(NE)
        v_cmx = np.zeros(NE)
        v_cmy = np.zeros(NE)
        sigma_vcmx = np.zeros(NE)
        sigma_vcmy = np.zeros(NE)
        T_eff = np.zeros(NE)
        frac6_E = np.zeros(NE)
        psi6_E = np.zeros(NE)
        for i, fname in enumerate(fnames):
            if fname[1] == '_':
                fname_words = fname.split('_')
                E_list[i] = float(fname_words[2])
                Eaxis = fname_words[1]
            else:
                E_list[i] = float(fname[1:-4])
                Eaxis = 'x'

            if Eaxis == 'x':
                ax_ind = 0
            elif Eaxis == 'y':
                ax_ind = 1
            else:
                raise ValueError('Eaxis must be x or y ({} given)'.format(Eaxis))

            # Calculate v_cm and effective temperature
            with gsd.fl.GSDFile(sf[0] + '/' + fname, 'rb') as f:
                n_frames = f.nframes
                try:
                    N = f.read_chunk(frame=0, name='particles/N')
                except:
                    print('could not read N from {}, using N=1024'.format(fname))
                    N = 1024
                v = np.zeros((n_frames, int(N), 2))
                for t in range(n_frames):
                    v_t = f.read_chunk(frame=t, name='particles/velocity')
                    v[t, :, 0] = v_t[:,0]
                    v[t, :, 1] = v_t[:,1]
            (frac6_E[i], psi6_E[i]) = fraction_six(sf[0] + '/' + fname)
            v_cm_data = np.mean(v, axis=1)
            v_cmx[i] = np.mean(v_cm_data[:,0])
            v_cmy[i] = np.mean(v_cm_data[:,1])
            ## =================================================
            #autocorrelation as a function of time interval
            Nt = 50
            t_arr = np.arange(Nt)
            v_cm_av = np.array([[v_cmx[i], v_cmy[i]]])
            rho_arr = np.zeros((Nt, 2))
            for i_t,delta in enumerate(t_arr):
                rho_arr[i_t,:] = np.mean((v_cm_data - v_cm_av)*np.roll((v_cm_data - v_cm_av), delta))
            sigmasq_from_rho = 2/n_frames*np.sum(rho_arr, axis = 0)
            sigma_vcmx = 2*np.sqrt(sigmasq_from_rho[0]) # factor of two is for 2\sigma error bars
            sigma_vcmy = 2*np.sqrt(sigmasq_from_rho[1])

            #sigma_vcmx[i] = np.sqrt(np.mean((v_cm_data[:,0] - v_cmx[i])**2))/np.sqrt(n_frames)
            #sigma_vcmy[i] = np.sqrt(np.mean((v_cm_data[:,1] - v_cmy[i])**2))/np.sqrt(n_frames)
            #sigma_vcmx[i] = bootstrap_sterr(v_cm_data[:,0], B=100)
            #sigma_vcmy[i] = bootstrap_sterr(v_cm_data[:,1], B=100)
            
            v_rel = np.swapaxes(v, 0,1) - v_cm_data
            T_data = 0.5*np.mean(v_rel[:,:,0]**2 + v_rel[:,:,1]**2, axis = 0)
            T_eff[i] = np.mean(T_data)
            # Save the results
            data = {'E_list':E_list, 'v_cmx':v_cmx, 'v_cmy':v_cmy, 'sigma_vcmx':sigma_vcmx, 'sigma_vcmy':sigma_vcmy, 'T_eff':T_eff}
            with open(sf[0] + '/tdata_{}{:2.6f}_'.format(mode, X) + '.dat', 'wb') as f:
                pickle.dump(data, f)
        ## ============================================================================
        # Estimate mobility for given X
        E_list_SI = E_list*unit_M*unit_D/unit_t**2/unit_Q/3.335e-10
        if Eaxis == 'x':
            #[v_cm] = unit_D/unit_t
            beta, beta_sigma = OLS_1(np.reshape(E_list_SI, (-1,1)), v_cmx*unit_D/unit_t)
        else:
            beta, beta_sigma = OLS_1(np.reshape(E_list_SI, (-1,1)), v_cmy*unit_D/unit_t)
        mu_arr[isf] = beta[0] #in SI units: m^2/Vs
        sigma_mu_arr[isf] = np.sqrt(beta_sigma[0,0])
        
        ## =======================================================================
        # Calculate theoretical mobility
        T = T_gamma/gamma
        tau_rec = relaxation_tau(k_arr, T, Ntheta)
        p_arr = hbar*k_arr
        dp = p_arr[1] - p_arr[0]
        #integral = np.sum(p_arr**3/tau_rec*np.exp(-p_arr**2/(2*T)))*dp
        integral = scipy.integrate.simps(p_arr**3/tau_rec*np.exp(-p_arr**2/(2*T)), dx=dp)
        mu = integral/(2*T**2)*e_charge
        v_theory = mu*E_list

        ## =======================================================================
        #Plot v_drift vs E in native units
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,6))
        ax1.errorbar(E_list, v_cmx, yerr=sigma_vcmx, fmt='o', capsize=2, label='$v_x$')
        ax1.errorbar(E_list, v_cmy, yerr=sigma_vcmy, fmt='o', capsize=2, label='$v_y$')
        ax1.set_xlabel('$E$')
        ax1.set_ylabel('$v_{drift}$')
        ax1.set_xlim(0, E_list[-1]*1.1)
        ax1.legend(loc=7)
        ax2.scatter(E_list, T_eff)
        ax2.set_xlabel('$E$')
        ax2.set_ylabel('$T_{eff}$')
        ax2.set_xlim(0, np.max(E_list)*1.1)

        fig.patch.set_alpha(alpha=1)
        plt.tight_layout()
        if folder[-1] == '/':
            folder_name = folder.split('/')[-2]
        else:
            folder_name = folder.split('/')[-1]
        
        fig.savefig(sf[0] + '/' + folder_name +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.png')
        ax1.plot(E_list, mu*E_list, label='e-gas')
        #Place text
        text_list = [log_data['timestamp'], '$p = {}$'.format(p), '$A = {}$'.format(A),\
                'axis={}'.format(Eaxis), 'Coulomb: {}'.format(log_data['coulomb_status'].rstrip()), \
                '$\\Gamma= {:.1f}$'.format(gamma)]
        y_lim = ax1.get_ylim()
        x_lim = ax1.get_xlim()
        h = y_lim[1] - y_lim[0]
        w = x_lim[1] - x_lim[0]
        text_x = x_lim[0] + 0.1*w
        text_y = y_lim[1] - 0.05*h

        if type(text_list) == list: 
            n_str = len(text_list)
            for i_fig in range(n_str):
                ax1.text(text_x, text_y - 0.05*h*i_fig, text_list[i_fig])
        elif type(text_list) == str:
            ax1.text(text_x, text_y, text_list)
        else:
            raise TypeError('text_list must be a list of strings or a string')
        plt.tight_layout()
        ax1.legend(loc=7)
        fig.savefig(folder + '/' + folder_name + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        fig.savefig(sf[0] + '/' + folder_name + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        plt.close('all')

        
        ## =======================================================================
        #Plot v_drift vs E in Gaussian units
        labelfont = 22
        tickfont = labelfont - 8
        legendfont = labelfont - 4
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,6))
        coeff_E_native_to_mVcm = unit_M*unit_D/unit_t**2*e_charge/1.60217e-19/100*1e3
        coeff_v_native_to_cms = unit_D*100/unit_t
        #E_list_mVcm = E_list*coeff_E_native_to_mVcm #in mV/cm
        E_list_mVcm = E_list_SI*10
        v_cmx_cms = v_cmx*coeff_v_native_to_cms # in cm/s
        v_cmy_cms = v_cmy*coeff_v_native_to_cms # in cm/s
        sigma_vcmx_cms = sigma_vcmx*coeff_v_native_to_cms
        sigma_vcmy_cms = sigma_vcmy*coeff_v_native_to_cms
        ax1.errorbar(E_list_mVcm, v_cmx_cms, yerr=sigma_vcmx_cms, fmt='o', capsize=2, label='$v_x$')
        if not plot_xonly:
            ax1.errorbar(E_list_mVcm, v_cmy_cms, yerr=sigma_vcmy_cms, fmt='o', capsize=2, label ='$v_y$')
        ax1.plot(E_list_mVcm, E_list_mVcm*10*mu_arr[isf], color ='red', label = 'fit', lw = 1.5)
        ax1.set_xlabel('$E$ [mV/cm]', fontsize=labelfont)
        ax1.set_ylabel('$v_{\\mathrm{drift}}$ [cm/s]', fontsize = labelfont)
        ax1.set_xlim(0, E_list_mVcm[-1]*1.1)
        ax1.legend(loc=2, fontsize=legendfont)
        ax1.tick_params(labelsize=tickfont)

        formatter = mticker.ScalarFormatter(useMathText = True)
        formatter.set_powerlimits((-3,2))
        ax1.yaxis.set_major_formatter(formatter)
        ax1.yaxis.offsetText.set_fontsize(tickfont)

        ax2.scatter(E_list_mVcm, T_eff)
        ax2.set_xlabel('$E$ [mV/cm]', fontsize = labelfont)
        ax2.set_ylabel('$T_{\\mathrm{eff}}$ [K]', fontsize = labelfont)
        ax2.set_xlim(0, np.max(E_list_mVcm)*1.1)
        ax2.tick_params(labelsize = tickfont)
        
        fig.patch.set_alpha(alpha=1)
        plt.tight_layout()
        if folder[-1] == '/':
            folder_name = folder.split('/')[-2]
        else:
            folder_name = folder.split('/')[-1]
        
        fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.png')
        fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.eps')
        fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.pdf')
        
        fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.png')
        fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.eps')
        fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.pdf')
        ax1.plot(E_list_mVcm, mu/(unit_M*unit_D/unit_t**2/unit_Q/3.335e-10)*unit_D/unit_t*10*E_list_mVcm, label='theory', lw=1.5)
        #Place text
        text_list = [log_data['timestamp'], '$p = {}$'.format(p), '$A = {}$'.format(A),\
                'axis={}'.format(Eaxis), 'Coulomb: {}'.format(log_data['coulomb_status'].rstrip()), \
                '$\\Gamma= {:.1f}$'.format(gamma)]
        y_lim = ax1.get_ylim()
        x_lim = ax1.get_xlim()
        h = y_lim[1] - y_lim[0]
        w = x_lim[1] - x_lim[0]
        text_x = x_lim[0] + 0.1*w
        text_y = y_lim[1] - 0.05*h
        
        if show_text:
            if type(text_list) == list: 
                n_str = len(text_list)
                for i_fig in range(n_str):
                    ax1.text(text_x, text_y - 0.05*h*i_fig, text_list[i_fig])
            elif type(text_list) == str:
                ax1.text(text_x, text_y, text_list)
            else:
                raise TypeError('text_list must be a list of strings or a string')
        ax1.legend(loc=2, fontsize = legendfont)
        fig.patch.set_alpha(alpha=1)
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        #fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.eps')
        #fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        plt.tight_layout()
        fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        #fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.eps')
        #fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        plt.close('all')

        ##===================================================================
        #Plot fraction of 6-coordinated particles vs E
        frac6_Xlist.append(frac6_E)
        
        labelfont = 22
        tickfont = labelfont - 8
        legendfont = labelfont - 4
        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        E_list_mVcm = E_list_SI*10
        ax1.scatter(E_list_mVcm, frac6_E)
        ax1.set_xlabel('$E$ [mV/cm]', fontsize = labelfont)
        ax1.set_ylabel('$\\alpha_6$', fontsize = labelfont)
        ax1.tick_params(labelsize = tickfont)
        fig.patch.set_alpha(alpha=1)
        plt.tight_layout()
        fig.savefig(folder + '/' + folder_name +'_fraction6' + '_{}{:2.6f}'.format(mode, X) + '.png')
        #fig.savefig(folder + '/' + folder_name +'_fraction6' + '_{}{:2.6f}'.format(mode, X) + '.eps')
        #fig.savefig(folder + '/' + folder_name +'_fraction6' + '_{}{:2.6f}'.format(mode, X) + '.pdf')
        

        ##===================================================================
        #Plot psi6 vs E
        psi6_Xlist.append(psi6_E)
        
        labelfont = 22
        tickfont = labelfont - 8
        legendfont = labelfont - 4
        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        E_list_mVcm = E_list_SI*10
        ax1.scatter(E_list_mVcm, psi6_E)
        ax1.set_xlabel('$E$ [mV/cm]', fontsize = labelfont)
        ax1.set_ylabel('$\\psi_6$', fontsize = labelfont)
        ax1.tick_params(labelsize = tickfont)
        fig.patch.set_alpha(alpha=1)
        plt.tight_layout()
        fig.savefig(folder + '/' + folder_name +'_psi6' + '_{}{:2.6f}'.format(mode, X) + '.png')
        #fig.savefig(folder + '/' + folder_name +'_fraction6' + '_{}{:2.6f}'.format(mode, X) + '.eps')
        #fig.savefig(folder + '/' + folder_name +'_fraction6' + '_{}{:2.6f}'.format(mode, X) + '.pdf')

        
        
    if len(X_arr) <= 1:
        print('Skip mu-vs-X plot since there is just one point')
    else:
        if mode == 'A':
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 2
            fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(7,5))
            # mu_arr is in m^2/Vs
            ax1.errorbar(X_arr, mu_arr*1e4, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax1.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            y_lim = ax1.get_ylim()
            #ax1.set_ylim([0, y_lim[1]*1.1])
            
            ax2.scatter(X_arr, np.log(np.abs(1e4*mu_arr)), label='$\\log(\\mu)$')
            ax2.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax2.set_ylabel('$\\log(\\mu)$', fontsize = labelfont)
            ax2.tick_params(labelsize=tickfont)
            #compute log slope
            beta_log, beta_log_sigma_sq = OLS(np.reshape(X_arr, (-1,1)), np.log(np.abs(1e4*mu_arr)))
            #plot fit line
            ax2.plot(X_arr, beta_log[0] + X_arr*beta_log[1], label = 'fit')
            y_lim = ax2.get_ylim()
            #ax2.set_ylim([0, y_lim[1]*1.1])
            if show_text:
                y_lim = ax2.get_ylim()
                x_lim = ax2.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.5*w
                text_y = y_lim[1] - 0.05*h
                ax2.text(text_x, text_y, 'slope = {:.3f}'.format(beta_log[1]))
            #Place text
            if show_text:
                text_list = [log_data['timestamp'], '$p = {}$'.format(log_data['p']),\
                        'axis={}'.format(Eaxis), 'Coulomb: {}'.format(log_data['coulomb_status'].rstrip()), 'T = {:.3f} K'.format(T)]
                y_lim = ax1.get_ylim()
                x_lim = ax1.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.5*w
                text_y = y_lim[1] - 0.05*h
                if type(text_list) == list: 
                    n_str = len(text_list)
                    for i_fig in range(n_str):
                        ax1.text(text_x, text_y - 0.05*h*i_fig, text_list[i_fig])
                elif type(text_list) == str:
                    ax1.text(text_x, text_y, text_list)
                else:
                    raise TypeError('text_list must be a list of strings or a string')
            #ax1.legend(loc=7)
            ax2.legend(loc=7)
            ##===================================================================
            #Plot fraction of 6-coordinated particles vs A
            frac6_A = np.zeros(X_arr.shape)
            for i_f6A, f6arr in enumerate(frac6_Xlist):
                frac6_A[i_f6A] = f6arr[0]
            
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 4
            fig_f6, (ax1_f6, ax2_f6) = plt.subplots(1,2, figsize=(7,5))
            ax1_f6.scatter(X_arr, frac6_A)
            ax1_f6.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax1_f6.set_ylabel('$\\alpha_6$', fontsize = labelfont)
            ax1_f6.tick_params(labelsize = tickfont)
           
            ax2_f6.scatter(frac6_A, 1e4*mu_arr)
            ax2_f6.set_xlabel('$\\alpha_6$', fontsize = labelfont)
            ax2_f6.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            fig_f6.patch.set_alpha(alpha=1)
            plt.tight_layout()
            fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.png')
            #fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.eps')
            #fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.pdf')
        
            ##===================================================================
            #Plot psi6 vs A
            psi6_A = np.zeros(X_arr.shape)
            for i_f6A, psi6arr in enumerate(psi6_Xlist):
                psi6_A[i_f6A] = psi6arr[0]
            
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 4
            fig_f6, (ax1_f6, ax2_f6) = plt.subplots(1,2, figsize=(7,5))
            ax1_f6.scatter(X_arr, psi6_A)
            ax1_f6.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax1_f6.set_ylabel('$\\psi_6$', fontsize = labelfont)
            ax1_f6.tick_params(labelsize = tickfont)
           
            ax2_f6.scatter(psi6_A, 1e4*mu_arr)
            ax2_f6.set_xlabel('$\\psi_6$', fontsize = labelfont)
            ax2_f6.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            fig_f6.patch.set_alpha(alpha=1)
            plt.tight_layout()
            fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.png')
            #fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.eps')
            #fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.pdf')



        elif mode == 'p':
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 2
            
            ## figure out system size to calculate normalized p
            sys_period = int(folder_name[1:3])
            X_arr /= sys_period
            #fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(7,5))
            fig, ax1  = plt.subplots(1,1, figsize=(6,6))
            ax1.errorbar(X_arr, 1e4*mu_arr, yerr=1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$p$', fontsize = labelfont)
            ax1.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            
            #Show zero line
            xlims = ax1.get_xlim()
            ax1.plot(np.array(xlims), np.array([0,0]), '--', label = 'zero level', color='black') 
            ax1.set_xlim(xlims)
            #Place text
            if show_text:
                text_list = [log_data['timestamp'], '$A={}$'.format(A),\
                        'axis={}'.format(Eaxis), 'Coulomb: {}'.format(log_data['coulomb_status'].rstrip()), 'T = {:.3f} K'.format(T)]
                y_lim = ax1.get_ylim()
                x_lim = ax1.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.5*w
                text_y = y_lim[1] - 0.05*h
                if type(text_list) == list: 
                    n_str = len(text_list)
                    for i_fig in range(n_str):
                        ax1.text(text_x, text_y - 0.05*h*i_fig, text_list[i_fig])
                elif type(text_list) == str:
                    ax1.text(text_x, text_y, text_list)
                else:
                    raise TypeError('text_list must be a list of strings or a string')
            #ax1.legend(loc=7)
            #ax2.legend(loc=7)

        elif mode == 'G':
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 2
            
            gamma_arr = X_arr
            T_arr = log_data['T_gamma']/gamma_arr
            fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(7,5))
            ax1.errorbar(T_arr, 1e4*mu_arr, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$T$ [K]', fontsize = labelfont)
            ax1.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            
            #log_err = sigma_mu_arr/np.abs(mu_arr) - 0.5*sigma_mu_arr**2/np.abs(mu_arr)**2
            #ax2.errorbar(1/T_arr, np.log(np.abs(mu_arr)), yerr=log_err, fmt='o', capsize=2, label='$\\log(\\mu)$')
            ax2.scatter(1/T_arr, np.log(np.abs(1e4*mu_arr)), label='$\\log(\\mu)$')
            ax2.set_xlabel('$1/T$ [1/K]', fontsize = labelfont)
            ax2.set_ylabel('$\\log(\\mu)$', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ##compute log slope
            beta_log, beta_log_sigma_sq = OLS(np.reshape(1/T_arr, (-1,1)), np.log(np.abs(mu_arr)))
            #plot fit line
            #ax2.plot(1/T_arr, beta_log[0] + beta_log[1]/T_arr)
            plt.tight_layout()
            if show_text:
                y_lim = ax2.get_ylim()
                x_lim = ax2.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.5*w
                text_y = y_lim[1] - 0.05*h
                ax2.text(text_x, text_y, 'slope = {:.3f}'.format(beta_log[1]))
                #Place text
                text_list = [log_data['timestamp'], '$p = {}$'.format(log_data['p']), '$A = {}$'.format(A),\
                        'axis={}'.format(Eaxis), 'Coulomb: {}'.format(log_data['coulomb_status'].rstrip())]
                y_lim = ax1.get_ylim()
                x_lim = ax1.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.5*w
                text_y = y_lim[1] - 0.05*h
                if type(text_list) == list: 
                    n_str = len(text_list)
                    for i_fig in range(n_str):
                        ax1.text(text_x, text_y - 0.05*h*i_fig, text_list[i_fig])
                elif type(text_list) == str:
                    ax1.text(text_x, text_y, text_list)
                else:
                    raise TypeError('text_list must be a list of strings or a string')
            #ax1.legend(loc=7)
            #ax2.legend(loc=7)
            #fig.savefig(folder + '/' + 'mu_vs_T_' + folder_name  + '.png')
            
            ## Plot additional mu vs gamma plot
            fig_gamma, ax1  = plt.subplots(1, figsize=(7,5))
            ax1.errorbar(gamma_arr, 1e4*mu_arr, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$\\Gamma$', fontsize = labelfont)
            ax1.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            plt.tight_layout()
            fig_gamma.patch.set_alpha(alpha=1)
            fig_gamma.savefig(folder + '/' + 'mu_vs_gamma_'.format(mode) + folder_name  + '.png')
            
            
            ##===================================================================
            #Plot fraction of 6-coordinated particles vs T
            frac6_A = np.zeros(X_arr.shape)
            for i_f6A, f6arr in enumerate(frac6_Xlist):
                frac6_A[i_f6A] = f6arr[0]
            
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 4
            fig_f6, (ax1_f6, ax2_f6) = plt.subplots(1,2, figsize=(7,5))
            ax1_f6.scatter(T_arr, frac6_A)
            ax1_f6.set_xlabel('$T$ [K]', fontsize = labelfont)
            ax1_f6.set_ylabel('$\\alpha_6$', fontsize = labelfont)
            ax1_f6.tick_params(labelsize = tickfont)
           
            ax2_f6.scatter(frac6_A, 1e4*mu_arr)
            ax2_f6.set_xlabel('$\\alpha_6$', fontsize = labelfont)
            ax2_f6.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            fig_f6.patch.set_alpha(alpha=1)
            plt.tight_layout()
            fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.png')
            #fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.eps')
            #fig_f6.savefig(folder + '/' + folder_name +'_fraction6_vs_A.pdf')
        
            ##===================================================================
            #Plot psi6 vs A
            psi6_A = np.zeros(X_arr.shape)
            for i_f6A, psi6arr in enumerate(psi6_Xlist):
                psi6_A[i_f6A] = psi6arr[0]
            
            labelfont = 22
            tickfont = labelfont - 8
            legendfont = labelfont - 4
            fig_f6, (ax1_f6, ax2_f6) = plt.subplots(1,2, figsize=(7,5))
            ax1_f6.scatter(T_arr, psi6_A)
            ax1_f6.set_xlabel('$T$ [K]', fontsize = labelfont)
            ax1_f6.set_ylabel('$\\psi_6$', fontsize = labelfont)
            ax1_f6.tick_params(labelsize = tickfont)
           
            ax2_f6.scatter(psi6_A, 1e4*mu_arr)
            ax2_f6.set_xlabel('$\\psi_6$', fontsize = labelfont)
            ax2_f6.set_ylabel('$\\mu$ [cm$^2$/Vs]', fontsize = labelfont)
            fig_f6.patch.set_alpha(alpha=1)
            plt.tight_layout()
            fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.png')
            #fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.eps')
            #fig_f6.savefig(folder + '/' + folder_name +'_psi6_vs_A.pdf')

        

        else:
            RuntimeError('Expected mode one of: A, p, G')
        plt.tight_layout()
        fig.patch.set_alpha(alpha=1)
        fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name  + '.png')
        fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name  + '.eps')
        fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name  + '.pdf')
        plt.close('all')
        #Save mobility vs A data
        mu_X_data = {'{}_arr'.format(mode):X_arr, 'mu_arr':mu_arr, 'sigma_mu_arr': sigma_mu_arr} # mu in SI units
        with open(folder + '/mu_{}_data_'.format(mode) + folder_name + '.dat'.format(mode), 'w') as ff:
            pickle.dump(mu_X_data, ff)




