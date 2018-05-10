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
import inspect
import ripplon_scattering_module as rsm

#Use latex fonts
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)


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

def OLS(x, y):
    '''OLS: x must be a vertical two-dimensional array'''
    X = np.hstack((np.reshape(np.ones(x.shape[0]), (-1,1)), x))#.transpose()
    Xpr = X.transpose()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(Xpr, X)), Xpr), y)
    #Estimate errors
    sigma_sq = np.dot(y - np.dot(X, beta), y - np.dot(X, beta))/(len(y) - 1.)
    sigma_beta_sq = sigma_sq*np.linalg.inv(np.dot(Xpr, X))
    return beta, sigma_beta_sq # = [f_0, df/d(A^2)]

def OLS_1(x, y, sigma_y):
    '''OLS: x must be a vertical two-dimensional array'''
    X = x#.transpose()
    Xpr = X.transpose()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(Xpr, X)), Xpr), y)
    #Estimate errors
    #sigma_sq = np.dot(y - np.dot(X, beta), y - np.dot(X, beta))/(len(y) - 1.)
    #estimate sigma_sq directly from error data
    sigma_sq = np.mean(sigma_y**2)
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
clip_small = False
plot_xonly_str = ''
for i in range(len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        folder_list.append(sys.argv[i])
    if sys.argv[i] == '--xonly':
        plot_xonly = True
        plot_xonly_str = '_xonly_'
    if sys.argv[i] == '--showtext':
        show_text = True
    if sys.argv[i] == '--clipsmall':
        clip_small = True
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
        sigma_T_eff = np.zeros(NE)
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
            sigmasq_from_rho = 1/n_frames*(2*np.sum(rho_arr, axis = 0) - rho_arr[0,:])
            sigma_vcmx[i] = np.sqrt(np.abs(sigmasq_from_rho[0])) # factor of 2 for 2\sigma error bars
            sigma_vcmy[i] = np.sqrt(np.abs(sigmasq_from_rho[1]))

            #sigma_vcmx[i] = np.sqrt(np.mean((v_cm_data[:,0] - v_cmx[i])**2))/np.sqrt(n_frames)
            #sigma_vcmy[i] = np.sqrt(np.mean((v_cm_data[:,1] - v_cmy[i])**2))/np.sqrt(n_frames)
            #sigma_vcmx[i] = bootstrap_sterr(v_cm_data[:,0], B=100)
            #sigma_vcmy[i] = bootstrap_sterr(v_cm_data[:,1], B=100)
            
            v_rel = np.swapaxes(v, 0,1) - v_cm_data
            T_data = 0.5*np.mean(v_rel[:,:,0]**2 + v_rel[:,:,1]**2, axis = 0)
            T_eff[i] = np.mean(T_data)
            #Estimate standard error of T_eff
            rho_arr = np.zeros(Nt)
            for i_t,delta in enumerate(t_arr):
                rho_arr[i_t] = np.mean((T_data - T_eff[i])*np.roll((T_data - T_eff[i]), delta))
            sigmasq_from_rho = 1/n_frames*(2*np.sum(rho_arr) - rho_arr[0])
            sigma_T_eff[i] = np.sqrt(np.abs(sigmasq_from_rho))
            #sigma_T_eff[i] = np.std(T_data)/np.sqrt(len(T_data))
            # Save the results
            data = {'E_list':E_list, 'v_cmx':v_cmx, 'v_cmy':v_cmy, 'sigma_vcmx':sigma_vcmx, 'sigma_vcmy':sigma_vcmy, 'T_eff':T_eff}
            with open(sf[0] + '/tdata_{}{:2.6f}_'.format(mode, X) + '.dat', 'wb') as f:
                pickle.dump(data, f)
        ## ============================================================================
        # Estimate mobility for given X
        E_list_SI = E_list*unit_M*unit_D/unit_t**2/unit_Q/3.335e-10
        if Eaxis == 'x':
            #[v_cm] = unit_D/unit_t
            beta, beta_sigma = OLS_1(np.reshape(E_list_SI, (-1,1)), v_cmx*unit_D/unit_t, sigma_vcmx*unit_D/unit_t)
        else:
            beta, beta_sigma = OLS_1(np.reshape(E_list_SI, (-1,1)), v_cmy*unit_D/unit_t, sigma_vcmy*unit_D/unit_t)
        mu_arr[isf] = beta[0] #in SI units: m^2/Vs
        sigma_mu_arr[isf] = np.sqrt(beta_sigma[0,0])
        
        ## =======================================================================
        # Calculate theoretical mobility
        T = T_gamma/gamma
        #tau_rec = relaxation_tau(k_arr, T, Ntheta)
        #p_arr = hbar*k_arr
        #dp = p_arr[1] - p_arr[0]
        #integral = np.sum(p_arr**3/tau_rec*np.exp(-p_arr**2/(2*T)))*dp
        #integral = scipy.integrate.simps(p_arr**3/tau_rec*np.exp(-p_arr**2/(2*T)), dx=dp)
        #mu = integral/(2*T**2)*e_charge
        mu_oe = rsm.mu_one_e(T)
        mu_me = rsm.mu_many_e(T)
        v_theory_oe = mu_oe*E_list
        v_theory_me = mu_me*E_list

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
        
        fig.savefig(sf[0] + '/' + folder_name +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.pdf')
        ax1.plot(E_list, v_theory_oe, label='e-gas')
        ax1.plot(E_list, v_theory_me, label='e-liquid')
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
        fig.savefig(folder + '/' + folder_name + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        fig.savefig(sf[0] + '/' + folder_name + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        plt.close('all')

        
        ## =======================================================================
        #Plot v_drift vs E in Gaussian units
        labelfont = 18
        tickfont = labelfont - 6
        legendfont = labelfont - 4
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
        coeff_E_native_to_mVcm = unit_M*unit_D/unit_t**2*e_charge/1.60217e-19/100*1e3
        coeff_v_native_to_cms = unit_D*100/unit_t
        #E_list_mVcm = E_list*coeff_E_native_to_mVcm #in mV/cm
        E_list_mVcm = E_list_SI*10
        v_cmx_cms = v_cmx*coeff_v_native_to_cms # in cm/s
        v_cmy_cms = v_cmy*coeff_v_native_to_cms # in cm/s
        sigma_vcmx_cms = sigma_vcmx*coeff_v_native_to_cms
        sigma_vcmy_cms = sigma_vcmy*coeff_v_native_to_cms
        ax1.errorbar(E_list_mVcm, v_cmx_cms, yerr=sigma_vcmx_cms, fmt='o', capsize=2, label='$v_\\perp$')
        if not plot_xonly:
            ax1.errorbar(E_list_mVcm, v_cmy_cms, yerr=sigma_vcmy_cms, fmt='o', capsize=2, label ='$v_\\parallel$')
        ax1.plot(E_list_mVcm, E_list_mVcm*10*mu_arr[isf], color ='red', label = 'fit', lw = 2, ls = '--')
        ax1.plot(E_list_mVcm, E_list_mVcm*10*(mu_arr[isf] + sigma_mu_arr[isf]), color ='grey', label = 'fit+error', lw = 2, ls = '--')
        ax1.plot(E_list_mVcm, E_list_mVcm*10*(mu_arr[isf] - sigma_mu_arr[isf]), color ='grey', label = 'fit-error', lw = 2, ls = '--')
        ax1.set_xlabel('$E_{\\mathrm{d}}$ [mV/cm]', fontsize=labelfont)
        ax1.set_ylabel('$v_{\\mathrm{d}}$ [cm/s]', fontsize = labelfont)
        ax1.set_xlim(0, E_list_mVcm[-1]*1.1)
        #ax1.legend(loc=2, fontsize=legendfont)
        ax1.tick_params(labelsize=tickfont)

        formatter = mticker.ScalarFormatter()
        formatter.set_powerlimits((-3,2))
        ax1.yaxis.set_major_formatter(formatter)
        ax1.yaxis.offsetText.set_fontsize(tickfont)

        #ax2.scatter(E_list_mVcm, T_eff)
        ax2.errorbar(E_list_mVcm, T_eff, yerr=sigma_T_eff, fmt='o', capsize=2, label='$T$')
        ax2.set_xlabel('$E_{\\mathrm{d}}$ [mV/cm]', fontsize = labelfont)
        ax2.set_ylabel('$T$ [K]', fontsize = labelfont)
        ax2.set_xlim(0, np.max(E_list_mVcm)*1.1)
        ax2.tick_params(labelsize = tickfont)
        
        #fig.patch.set_alpha(alpha=1)
        #plt.tight_layout()
        if folder[-1] == '/':
            folder_name = folder.split('/')[-2]
        else:
            folder_name = folder.split('/')[-1]
        
        #fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.png')
        #fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.eps')
        fig.savefig(sf[0] + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.pdf')
        
        #fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.png')
        #fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.eps')
        fig.savefig(folder + '/' + folder_name + '_humanUnits' +'_{}{:2.6f}'.format(mode, X) + '_vTvsE.pdf')
        #ax1.plot(E_list_mVcm, mu_oe/(unit_M*unit_D/unit_t**2/unit_Q/3.335e-10)*unit_D/unit_t*10*E_list_mVcm, label='e-gas th.', lw=1.5)
        #ax1.plot(E_list_mVcm, mu_me/(unit_M*unit_D/unit_t**2/unit_Q/3.335e-10)*unit_D/unit_t*10*E_list_mVcm, label='e-liquid th.', lw=1.5)
        if log_data['coulomb_status'][:2] == 'on':
            ax1.plot(E_list_mVcm, v_theory_me*unit_D/unit_t*1e2, label='e-liquid th.', lw=2, color='black')
        else:
            ax1.plot(E_list_mVcm, v_theory_oe*unit_D/unit_t*1e2, label='e-gas th.', lw=2, color='black')
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
        #ax1.legend(loc=2, fontsize = legendfont)
        fig.patch.set_alpha(alpha=1)
        #plt.tight_layout()
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=1.0)
        #fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        #fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.eps')
        fig.savefig(folder + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        #fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.png')
        #fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.eps')
        fig.savefig(sf[0] + '/' + folder_name +'_humanUnits' + '_{}{:2.6f}'.format(mode, X) + '_vTvsE_wT.pdf')
        plt.close('all')

    #Scaling for log(mu/mu_0)
    mu_0 = 3.5e7
        
    if len(X_arr) <= 1:
        print('Skip mu-vs-X plot since there is just one point')
    else:
        if mode == 'A':
            sort_ind = np.argsort(X_arr)
            mu_arr = mu_arr[sort_ind]
            sigma_mu_arr = sigma_mu_arr[sort_ind]
            X_arr.sort()
            labelfont = 18
            tickfont = labelfont - 6
            legendfont = labelfont - 4
            fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(8,4))
            # mu_arr is in m^2/Vs
            ax1.errorbar(X_arr, mu_arr*1e4, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax1.set_ylabel('$\\mu_{\\perp}$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.locator_params(nbins=5, axis='y')
            ax1.yaxis.set_major_formatter(formatter)
            ax1.yaxis.offsetText.set_fontsize(tickfont)
            y_lim = ax1.get_ylim()
            ax1.set_ylim([0, y_lim[1]*1.1])
            ##compute only for points far enough from zero level
            first_small = len(X_arr)
            if clip_small:
                mu_max = np.max(mu_arr)
                small_ind = np.where(mu_arr < 0.03*mu_max)[0]
                first_small = small_ind[0]
                print(small_ind)
                print('mu_arr = {}'.format(mu_arr))
                print('mu_max = {}'.format(mu_max))
                print('0.05*mu_max = {}'.format(mu_max))
                print('Fitting {} first points'.format(first_small))
            

            log_mu = np.log(np.abs(1e4*mu_arr/mu_0))
            ax2.errorbar(X_arr, log_mu, yerr=sigma_mu_arr/mu_arr, mfc='white', color='C0', fmt='o', capsize=2)
            ax2.errorbar(X_arr[:first_small], log_mu[:first_small], yerr=sigma_mu_arr[:first_small]/mu_arr[:first_small], label='$\\log(\\mu_\\perp/\\mu_0)$', color='C0', fmt='o', capsize=2)

            #ax2.scatter(X_arr[:first_small], np.log(np.abs(1e4*mu_arr[:first_small])), label='$\\log(\\mu)$', color='C0')
            #ax2.scatter(X_arr, np.log(np.abs(1e4*mu_arr)), label='$\\log(\\mu)$', facecolors='none', edgecolors='C0')
            ax2.set_xlabel('$A$ [K]', fontsize = labelfont)
            ax2.set_ylabel('$\\log(\\mu_{\\perp}/\\mu_0)$', fontsize = labelfont)
            ax2.tick_params(labelsize=tickfont)
            #compute log slope
            beta_log, beta_log_sigma_sq = OLS(np.reshape(X_arr[:first_small], (-1,1)), log_mu[:first_small])
            #plot fit line
            ax2.plot(X_arr, beta_log[0] + X_arr*beta_log[1], label = 'fit', color='black')
            y_lim = ax2.get_ylim()
            #ax2.set_ylim([0, y_lim[1]*1.1])
            if show_text:
                y_lim = ax2.get_ylim()
                x_lim = ax2.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[0] + 0.2*w
                text_y = y_lim[1] - 0.05*h
                ax2.text(text_x, text_y, 'slope = {:.3f}$\\pm${:.3f}'.format(beta_log[1], np.sqrt(beta_log_sigma_sq[1,1])))
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
            #write labels (a) and (b)
            show_ab = True
            if show_ab:
                #ax1
                y_lim = ax1.get_ylim()
                x_lim = ax1.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[1] - 0.12*w
                text_y = y_lim[1] - 0.08*h
                ax1.text(text_x, text_y, '(a)', fontsize=labelfont)
                #ax2
                y_lim = ax2.get_ylim()
                x_lim = ax2.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[1] - 0.12*w
                text_y = y_lim[1] - 0.08*h
                ax2.text(text_x, text_y, '(b)', fontsize=labelfont)

            #ax1.legend(loc=7)
        
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
            ax1.yaxis.offsetText.set_fontsize(tickfont)
            
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
            labelfont = 18
            tickfont = labelfont - 6
            legendfont = labelfont - 4
            
            
            gamma_arr = X_arr
            T_arr = log_data['T_gamma']/gamma_arr
            
            sort_ind = np.argsort(gamma_arr)
            mu_arr = mu_arr[sort_ind]
            sigma_mu_arr = sigma_mu_arr[sort_ind]
            T_arr = T_arr[sort_ind]
            gamma_arr.sort()
            
            fig, (ax1, ax2)  = plt.subplots(1,2, figsize=(8,4))
            ax1.errorbar(T_arr, 1e4*mu_arr, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$T$ [K]', fontsize = labelfont)
            ax1.set_ylabel('$\\mu_\\perp$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            ax1.yaxis.offsetText.set_fontsize(tickfont)
            
            log_err = sigma_mu_arr/np.abs(mu_arr) - 0.5*sigma_mu_arr**2/np.abs(mu_arr)**2
            #ax2.errorbar(1/T_arr, np.log(np.abs(mu_arr)), yerr=log_err, fmt='o', capsize=2, label='$\\log(\\mu)$')
            first_small = len(X_arr)
            if clip_small:
                mu_max = np.max(mu_arr)
                small_ind = np.where(mu_arr < 0.05*mu_max)[0]
                first_small = small_ind[0]
                print(small_ind)
                print('mu_arr = {}'.format(mu_arr))
                print('mu_max = {}'.format(mu_max))
                print('0.07*mu_max = {}'.format(mu_max))
                print('Fitting {} first points'.format(first_small))

            log_mu = np.log(np.abs(1e4*mu_arr/mu_0))
            #ax2.errorbar(1/T_arr, np.log(np.abs(1e4*mu_arr)), yerr=log_err, label='$\\log(\\mu)$', mfc='white', color='C0', fmt='o', capsize=2)
            ax2.errorbar(1/T_arr[:first_small], log_mu[:first_small], yerr=log_err[:first_small],\
                    label='$\\log(\\mu)$', color='C0', fmt='o', capsize=2)


            ax2.scatter(1/T_arr, log_mu, label='$\\log(\\mu_{\\perp}/\mu_0)$', facecolor = 'none', edgecolor='C0')
            ax2.set_xlabel('$1/T$ [1/K]', fontsize = labelfont)
            ax2.set_ylabel('$\\log(\\mu_{\\perp}/\\mu_0)$', fontsize = labelfont)
            ax2.tick_params(labelsize=tickfont)
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
            show_ab = True
            if show_ab:
                #ax1
                y_lim = ax1.get_ylim()
                x_lim = ax1.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[1] - 0.12*w
                text_y = y_lim[1] - 0.08*h
                ax1.text(text_x, text_y, '(a)', fontsize=labelfont)
                #ax2
                y_lim = ax2.get_ylim()
                x_lim = ax2.get_xlim()
                h = y_lim[1] - y_lim[0]
                w = x_lim[1] - x_lim[0]
                text_x = x_lim[1] - 0.12*w
                text_y = y_lim[1] - 0.08*h
                ax2.text(text_x, text_y, '(b)', fontsize=labelfont)



            ## Plot additional mu vs gamma plot
            fig_gamma, ax1  = plt.subplots(1, figsize=(7,5))
            ax1.errorbar(gamma_arr, 1e4*mu_arr, yerr = 1e4*sigma_mu_arr, fmt='o', capsize=2, label='$\\mu$')
            ax1.set_xlabel('$\\Gamma$', fontsize = labelfont)
            ax1.set_ylabel('$\\mu_\\perp$ [cm$^2$/Vs]', fontsize = labelfont)
            ax1.tick_params(labelsize=tickfont)
            ax1.yaxis.set_major_formatter(formatter)
            plt.tight_layout()
            fig_gamma.patch.set_alpha(alpha=1)
            #fig_gamma.savefig(folder + '/' + 'mu_vs_gamma_'.format(mode) + folder_name  + '.png')
            plt.close('all')
        

        else:
            RuntimeError('Expected mode one of: A, p, G')
        
        show_text_line = ''
        if show_text:
            show_text_line = '_wText'
        if clip_small:
            show_text_line += 'Clip'

        plt.tight_layout()
        fig.patch.set_alpha(alpha=1)
        #fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name + show_text_line  + '.png')
        #fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name + show_text_line + '.eps')
        fig.savefig(folder + '/' + 'mu_vs_{}_'.format(mode) + folder_name + show_text_line + '.pdf')
        plt.close('all')
        #Save mobility vs A data
        mu_X_data = {'{}_arr'.format(mode):X_arr, 'mu_arr':mu_arr, 'sigma_mu_arr': sigma_mu_arr} # mu in SI units
        with open(folder + '/mu_{}_data_'.format(mode) + folder_name + '.dat'.format(mode), 'w') as ff:
            pickle.dump(mu_X_data, ff)




