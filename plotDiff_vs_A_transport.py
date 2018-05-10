from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import gsd
import gsd.fl
import numpy as np
import os
import sys
import datetime
import time
import pickle
from shutil import copyfile
import inspect
import md_tools27 as md_tools

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)


"""
This script plots diffusion coefficients for data obtained in transport simulations
"""

def read_log(path):
    coulomb_status = ''
    with open(path + '/log.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                timestamp = line.rstrip()
            if line[:10] == '# Periodic':
                words = line.split(' ')
                p = int(words[9])
            if line[:4] == '# a ':
                words = line.split(' ')
                repeat_x = int(words[6])
                repeat_y = int(words[9])
                Np = 2*repeat_x*repeat_y
            if line[:7] == '# Gamma':
                words = line.split(' ')
                gamma = float(words[2])
                T = float(words[5])
                dt = float(words[9])
            if line[:9] == '# Coulomb':
                words = line.split(' ')
                coulomb_status = words[-1]
            if line[:9] == '# N_therm':
                words = line.split(' ')
                snap_period = int(words[5])
    return {'timestamp': timestamp, 'p':p, 'Np': Np, 'gamma':gamma, 'T':T, 'coulomb_status':coulomb_status, 'snap_period':snap_period,\
            'dt':dt}

def OLS(x, y):
    '''OLS: x must be a vertical two-dimensional array'''
    X = np.hstack((np.reshape(np.ones(x.shape[0]), (-1,1)), x))#.transpose()
    Xpr = X.transpose()
    beta = np.dot(np.dot(np.linalg.inv(np.dot(Xpr, X)), Xpr), y)
    #Estimate errors
    sigma_sq = np.dot(y - np.dot(X, beta), y - np.dot(X, beta))/(len(y) - 1.)
    sigma_beta_sq = sigma_sq*np.linalg.inv(np.dot(Xpr, X))
    return beta, sigma_beta_sq # = [f_0, df/d(A^2)]



def diffusion_from_transport_gsd(folder_path, f_name, center_fixed = True, useframes = -1):
    """
    
    Diffusion constant D is calculated from 4Dt = <(r(t) - r(0))^2>, or 2D_x*t = <(x(t) - x(0))^2>.
    The average is calculated over all particles and over different time origins.
    Time origins go from 0 to n_frames/2, and t goes from 0 to n_frames/2. This way,
    the data are always within the trajectory.
                                                                    
    center_fixed = True: eliminate oveall motion of center of mass
    return D_x, D_y
    D_x, D_y diffusion for x- and y-coordinates;
    """
    params = read_log(folder_path)
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    with gsd.fl.GSDFile(folder_path + f_name, 'rb') as f:
        n_frames = f.nframes
        box = f.read_chunk(frame=0, name='configuration/box')
        half_frames = int(n_frames/2) - 1 #sligtly less than half to avoid out of bound i
        if useframes < 1 or useframes > half_frames:
            useframes = half_frames
        t_step = f.read_chunk(frame=0, name='configuration/step')
        n_p = f.read_chunk(frame=0, name='particles/N')
        x_sq_av = np.zeros(useframes)
        y_sq_av = np.zeros(useframes)
        for t_origin in range(n_frames - useframes - 1):
            pos_0 = f.read_chunk(frame=t_origin, name='particles/position')
            mean_pos_0 = np.mean(pos_0, axis = 0)
            pos = pos_0
            pos_raw = pos_0
            for j_frame in range(useframes):
                pos_m1 = pos
                pos_m1_raw = pos_raw
                pos_raw = f.read_chunk(frame=j_frame + t_origin, name='particles/position') - pos_0
                pos = md_tools.correct_jumps(pos_raw, pos_m1, pos_m1_raw, box[0], box[1])
                if center_fixed:
                    pos -= np.mean(pos, axis = 0) - mean_pos_0 #correct for center of mass movement
                x_sq_av[j_frame] += np.mean(pos[:,0]**2)
                y_sq_av[j_frame] += np.mean(pos[:,1]**2)
    x_sq_av /= (n_frames - useframes - 1)
    y_sq_av /= (n_frames - useframes - 1)
    print('x_sq_av = {}'.format(x_sq_av[:10]))
    # OLS estimate for beta_x[0] + beta_x[1]*t = <|x_i(t) - x_i(0)|^2>
    a = np.ones((useframes, 2)) # matrix a = ones(half_frames) | (0; dt; 2dt; 3dt; ...)
    a[:,1] = params['snap_period']*params['dt']*np.cumsum(np.ones(useframes), axis = 0) - params['dt']
    b_cutoff = int(useframes/10) #cutoff to get only linear part of x_sq_av, makes results a bit more clean
    beta_x = np.linalg.lstsq(a[b_cutoff:, :], x_sq_av[b_cutoff:], rcond=-1)
    beta_y = np.linalg.lstsq(a[b_cutoff:, :], y_sq_av[b_cutoff:], rcond=-1)
    
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.scatter(a[:,1], x_sq_av, label='$\\langle x^2\\rangle$')
    ax.scatter(a[:,1], y_sq_av, label='$\\langle y^2\\rangle$')
    ax.legend(loc=7)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\\langle r_i^2 \\rangle$')
    fig.savefig(folder_path + 'r2_diff.pdf')
    plt.close('all')
    print('beta_x = {}'.format(beta_x))
    print('beta_y = {}'.format(beta_y))
    print('t_step = {}'.format(t_step))
    D_x = beta_x[0][1]/2
    D_y = beta_y[0][1]/2
    print('D_x = {}'.format(D_x))
    print('D_y = {}'.format(D_y))
    return D_x, D_y


def print_help():
    print('This script plots diffusion vs A for data taken in transport measurements.')
    print('Only data points with E=0 are used')
    print('===========================================================')
    print('Usage: python plotDiff_vs_A_transport.py mobility_data/a32x32_* [--options]')
    print('This will process all folders that match mobility_data/a32x32_*')
    print('===========================================================')
    print('Options:')
    print('\t--cmfixed will subtract the displacement of the center of mass in diffusion calculation (default behavior)')
    print('\t--cmfree will NOT subtract the displacement of the center of mass in diffusion calculation (default behavior)')
    print('\t--showtext will print text info on the plots')
    print('\t--help or -h will print this help')
    
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

##=======================================================================
# Make a list of folders we want to process
cm_fixed = True #default that can be changed by --cmfree
cm_fixed_str = 'cm_fixed'
show_text = False
folder_list = []
for i in range(len(sys.argv)):
    if os.path.isdir(sys.argv[i]):
        folder_list.append(sys.argv[i])
    elif i != 0 and sys.argv[i] != '--cmfree' and sys.argv[i] != '--cmfixed':
        print(sys.argv[i] + ' is not a folder')
    elif sys.argv[i] == '--cmfree':
        cm_fixed = False
        cm_fixed_str = 'cm_free'
    elif sys.argv[i] == '--cmfixed':
        cm_fixed = True
        cm_fixed_str = 'cm_fixed'
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
    subfolder_lists.append(sf_list)


##=======================================================================
#calculate current and Teff for all files and plot the results:
for ifold, folder in enumerate(folder_list):
    print('==========================================================')
    print(folder)
    print('==========================================================')
    Dx_arr = np.zeros(len(subfolder_lists[ifold]))
    Dy_arr = np.zeros(len(subfolder_lists[ifold]))
    A_arr = np.zeros(len(subfolder_lists[ifold]))
    for isf, sf in enumerate(subfolder_lists[ifold]):
        sf_words = sf[0].split('/')
        print(sf_words[-1])
        A_arr[isf] = float(sf_words[-1][1:])
        if sf_words[-1][0] != 'A':
            raise ValueError("Expected subfolder name to start with `A`, in {}".format(fname))
        log_data = read_log(sf[0])
        A = float(sf_words[-1][1:])
        NE = 0
        fnames = []
        for fname in sf[1]:
            if fname[:2] == 'E_':
                fnames.append(fname)
                NE += 1
        fnames.sort()
        fname = fnames[0]
        print(fname)
        fname_words = fname.split('_')
        Dx, Dy = diffusion_from_transport_gsd(sf[0], fname, center_fixed = cm_fixed, useframes = -1)
        Dx_arr[isf] = Dx
        Dy_arr[isf] = Dy
    folder_name = folder.split('/')[-1]
    #Save mobility vs A data
    DxDy_data = {'Dx_arr':Dx_arr, 'Dy_arr':Dy_arr}
    with open(folder + '/DxDy_data_' + cm_fixed_str + '_' + folder_name + '.dat', 'w') as ff:
        pickle.dump(DxDy_data, ff)
        
    # Plot results
    labelfont = 22
    tickfont = labelfont - 4
    legendfont = labelfont - 2
    
    fig, ax1  = plt.subplots(1,1, figsize=(6,6))
    T_gamma = log_data['T']*log_data['gamma'] # same as e^2(\pi*n_s)^{1/2}
    ax1.scatter(A_arr, Dx_arr*unit_D**2/unit_t*1e4, label='$D_\\perp$', color = 'green', marker='o')
    ax1.set_xlabel('$A$ [K]', fontsize = labelfont)
    ax1.set_ylabel('$D$ [cm$^2$/s]', fontsize = labelfont)
    ax1.scatter(A_arr, Dy_arr*unit_D**2/unit_t*1e4, label='$D_\\parallel$', color = 'red', marker='s')
    ax1.tick_params(labelsize=tickfont)
    
    formatter = mticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((-3,2))
    ax1.xaxis.set_major_formatter(formatter)
    x_min_max =[np.min(A_arr), np.max(A_arr)]
    dX = x_min_max[1] - x_min_max[0]
    ax1.set_xlim([-dX*0.1, x_min_max[1] + 0.1*dX])
    ax1.set_ylim([0, 1.1*np.max(Dy_arr*unit_D**2/unit_t*1e4)])
    #Place text
    if show_text:
        text_list = [log_data['timestamp'], '$p = {}$'.format(log_data['p']),\
                'Coulomb: {}'.format(log_data['coulomb_status'].rstrip()), 'T = {:.3f} K'.format(log_data['T']), '$\\Gamma={:.1f}$'.format(log_data['gamma'])]
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
    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    ax1.legend(loc=1, fontsize=legendfont)
    #fig.savefig(folder + '/' + 'DxDy_' + cm_fixed_str + '_' + folder_name  + '.png')
    fig.savefig(folder + '/' + 'DxDy_' + cm_fixed_str + '_' + folder_name  + '.pdf')
    plt.close('all')




