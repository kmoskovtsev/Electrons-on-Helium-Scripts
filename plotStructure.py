#from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
from multiprocessing import Pool
from hoomd.data import boxdim

from matplotlib import rc
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('text', usetex=True)

"""
This script plots pair-correlation function, structure factor, Delaunay triangulation, and \psi_6 order parameter (with correlation function)
for all .gsd files in folders specified through command line.
Example:
    python plotStructure.py diff_data/a32x32_A0.5* --sf p32 -NP 10
will plot structure images for all gsd files in all folders that match diff_data/a32x32_A0.5* template, only in subfolders `p32`,
using 10 parallel processes.
Args:
--sf <fubfolder>: subfolder to process (e.g. p32)
--NP <number>: number of subprocesses to use for parallelization. Very efficient acceleration by a factor of <number>.
"""

def plot_structure(fpath):
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    print(fpath)
    fpath_words = fpath.split('/')
    folder_path = '/'.join(fpath_words[:-1]) + '/'
    plt.close('all')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,12))

    frame = 0

    labelfont = 28
    tickfont = labelfont - 4
    labelpad = -25
    g = md_tools.pair_correlation_from_gsd(fpath, n_bins = (256, 256), frames =(frame , frame +5))
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = f_gsd.read_chunk(frame=frame, name='particles/position')
        box = boxdim(*box_array[0:3])
    fig, ax1, cax1 = md_tools.plot_pair_correlation(g, box, fig=fig, ax=ax1, tickfont=tickfont)
    ax1.set_title('Pair correlation function')
    ax1.set_xlabel('$x/a$', fontsize = labelfont)
    ax1.set_ylabel('$y/a$', fontsize = labelfont, labelpad=labelpad)
    ax1.tick_params(labelsize = tickfont)

    S = np.abs(np.fft.fft2(g))
    S[0,0] = 0
    S_recentered = np.zeros(S.shape)
    #Exchange diagonal quarters
    S_recentered[:(S.shape[0]//2 - 1), :(S.shape[1]//2 - 1)] = S[(S.shape[0]//2 + 1):, (S.shape[1]//2 + 1):]
    S_recentered[(S.shape[0]//2 - 1):, (S.shape[1]//2 - 1):] = S[:(S.shape[0]//2 + 1), :(S.shape[1]//2 + 1)]
    #Exchange off-diagonal quarters
    S_recentered[(S.shape[0]//2 - 1):, :(S.shape[1]//2 - 1)] = S[:(S.shape[0]//2 + 1), (S.shape[1]//2 + 1):] #red off-diag
    S_recentered[:(S.shape[0]//2 - 1), (S.shape[1]//2 - 1):] = S[(S.shape[0]//2 + 1):, :(S.shape[1]//2 + 1)] #blue off-diag
    kx_lim = 2*np.pi/box.Lx*S.shape[0]
    ky_lim = 2*np.pi/box.Ly*S.shape[1]
    k_box = boxdim(kx_lim, ky_lim, 1)
    fig, ax2, cax2 = md_tools.plot_pair_correlation(S_recentered, k_box, fig=fig, ax=ax2, cmap='hot', tickfont=tickfont)
    #ax2.pcolor(S_recentered, cmap='hot')
    ax2.set_title('Structure factor')
    ax2.set_xlabel('$k_x$', fontsize = labelfont)
    ax2.set_ylabel('$k_y$', fontsize = labelfont, labelpad=labelpad)
    ax2.tick_params(labelsize = tickfont)

    psi = md_tools.psi6_order_from_gsd(fpath, frame=frame)

    fig, ax3, cax3 = md_tools.plot_pair_correlation(np.transpose(psi.real), box, alpha=1, cmap='cool', fig=fig, ax=ax3,\
                                                   origin_marker=False, minmax=[-1,1], tickfont=tickfont)
    fig, ax3 = md_tools.plot_delone_triangulation(fpath, frame=frame, fig=fig, ax=ax3)
    ax3.set_title('$\\psi_6$ and Delaunay triangulation')
    ax3.set_xlabel('$x/a$', fontsize = labelfont)
    ax3.set_ylabel('$y/a$', fontsize = labelfont, labelpad=labelpad)
    ax3.tick_params(labelsize = tickfont)

    cf_psi = md_tools.compute_psi6_correlation_from_gsd(fpath, Nframes=1, frame_step=1, nxny=(100,100))

    fig, ax4, cax4 = md_tools.plot_pair_correlation(np.abs(np.transpose(cf_psi)), box, fig=fig, ax=ax4,\
                                                   origin_marker=False, cmap='rainbow', minmax = [0,np.abs(cf_psi).max()],\
                                                                          tickfont=tickfont, interp='gaussian', imshow=True)
    ax4.set_title('${<}\psi_6^{*}(r)\psi_6(0){>}$')
    ax4.set_xlabel('$x/a$', fontsize = labelfont)
    ax4.set_ylabel('$y/a$', fontsize = labelfont, labelpad=labelpad)
    ax4.tick_params(labelsize = tickfont)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')

    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    
    #fig.savefig(folder_path + 'struct_' + fpath_words[-1] + '_' + fpath_words[-2] + '.png')
    #fig.savefig(folder_path + 'struct_' + fpath_words[-1] + '_' + fpath_words[-2] + '.eps')
    fig.savefig(folder_path + 'struct_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')
    plt.close('all')
    
    #labelfont = 22
    #tickfont = labelfont - 8
    ## Plot snapshot
    fig, ax = md_tools.plot_positions(pos=pos, box=box, figsize = (7, 7), gridon = False, s=8)
    ax.set_xlabel('$x/a$', fontsize= labelfont)
    ax.set_ylabel('$y/a$', fontsize= labelfont)
    ax.tick_params(labelsize=tickfont)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    #fig.savefig(folder_path + 'snapshot_' + fpath_words[-1] + '_' + fpath_words[-2] + '.png')
    #fig.savefig(folder_path + 'snapshot_' + fpath_words[-1] + '_' + fpath_words[-2] + '.eps')
    fig.savefig(folder_path + 'snapshot_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')
    
    plt.close('all')


def plot_structure_sep(fpath):
    # This funciton plots the pair correlation function and Delaunay in separate plots (for paper)
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    print(fpath)
    fpath_words = fpath.split('/')
    folder_path = '/'.join(fpath_words[:-1]) + '/'
    plt.close('all')

    ## Pair correlation function
    fig, ax1 = plt.subplots(1,1, figsize=(6,6))
    frame = 0

    labelfont = 32
    tickfont = labelfont - 4
    labelpad = -25
    g = md_tools.pair_correlation_from_gsd(fpath, n_bins = (256, 256), frames =(frame , frame +5))
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = f_gsd.read_chunk(frame=frame, name='particles/position')
        box = boxdim(*box_array[0:3])
    fig, ax1, cax1 = md_tools.plot_pair_correlation(g, box, fig=fig, ax=ax1, cmap='Blues', tickfont=tickfont)
    #ax1.set_title('Pair correlation function')
    ax1.set_xlabel('$x/a_s$', fontsize = labelfont)
    ax1.set_ylabel('$y/a_s$', fontsize = labelfont, labelpad=labelpad)
    ax1.tick_params(labelsize = tickfont)
    ax1.set_aspect('equal')

    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    fig.savefig(folder_path + 'pair_corr_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')
    
    ## Delaunay triangulation and psi6
    fig, ax3 = plt.subplots(1,1, figsize=(6,6))
    psi = md_tools.psi6_order_from_gsd(fpath, frame=frame)
    colormap = 'rainbow';
    fig, ax3, cax3 = md_tools.plot_pair_correlation(np.transpose(psi.real), box, alpha=0.7, cmap=colormap, fig=fig, ax=ax3,\
                                                   origin_marker=False, minmax=[-1,1], imshow=True, tickfont=tickfont)
    fig, ax3 = md_tools.plot_delone_triangulation(fpath, frame=frame, fig=fig, ax=ax3)
    #ax3.set_title('$\\psi_6$ and Delaunay triangulation')
    ax3.set_xlabel('$x/a_s$', fontsize = labelfont)
    ax3.set_ylabel('$y/a_s$', fontsize = labelfont, labelpad=labelpad)
    ax3.tick_params(labelsize = tickfont)
    ax3.set_aspect('equal')

    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    
    fig.savefig(folder_path + 'delaunay_' + colormap + '_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')
    plt.close('all')



def print_help():
    print('This script plots snapshots, correlation functions, and Delaunay triangulation for all .gsd files found in folders.')
    print('The .gsd files should be in the first-level subfolders in the specified folders')
    print('===========================================================')
    print('Usage: python plotStructure.py mobility_data/a32x32_* [--options]')
    print('This will process all folders that match mobility_data/a32x32_*')
    print('===========================================================')
    print('Options:')
    print('\t--NP N - will use N parallel processes in the calculations')
    print('\t--sf [subfolder] - will only process the specified subfolder in all folders')
    print('\t--help or -h will print this help')
    
    
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


for i in range(len(sys.argv)):
    if sys.argv[i] == '--help' or sys.argv[i] == '-h':
        print_help()
        exit()


if __name__ == '__main__':
    ##=======================================================================
    # Make a list of folders we want to process
    folder_list = []
    selected_subfolders = []
    Nproc = 1
    for i in range(len(sys.argv)):
        if os.path.isdir(sys.argv[i]):
            folder_list.append(sys.argv[i])
        elif sys.argv[i] == '--NP':
            try:
                Nproc = int(sys.argv[i+1])
            except:
                raise RuntimeError('Could not recognize the value of --NP. argv={}'.format(argv))
        elif sys.argv[i] == '--sf':
            try:
                selected_subfolders.append(sys.argv[i+1])
            except:
                raise RuntimeError('Could not recognize the value of --sf. argv={}'.format(argv))


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
    #calculate correlation function, structure factor, snapshot, and correlation of angular order for each gsd
    for ifold, folder in enumerate(folder_list):
        print('==========================================================')
        print(folder)
        print('==========================================================')
        # Keep only selected subfolders in the list is there is selection
        if len(selected_subfolders) > 0:
            sf_lists_to_go = []
            for isf, sf in enumerate(subfolder_lists[ifold]):
                sf_words = sf[0].split('/')
                if sf_words[-1] in selected_subfolders:
                    sf_lists_to_go.append(sf)
        else:
            sf_lists_to_go = subfolder_lists[ifold]

        for isf, sf in enumerate(sf_lists_to_go):
            sf_words = sf[0].split('/')
            print(sf_words[-1])
            fnames = []
            fpaths = []
            for fname in sf[1]:
                if fname[-4:] == '.gsd':
                    fnames.append(fname)
                    fpaths.append(sf[0] + '/' + fname)
            fnames.sort()
            fpaths.sort()
            print(fnames)
            p = Pool(Nproc, maxtasksperchild=1)
            p.map(plot_structure, fpaths)
            p.map(plot_structure_sep, fpaths)
            p.close()
            p.join()

