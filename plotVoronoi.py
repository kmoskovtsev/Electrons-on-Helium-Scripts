from __future__ import division
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
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import colors

from matplotlib import rc


"""
This script plots Voronoi tessellation with color-coded cell sizes
for all .gsd files in folders specified through command line.
Example:
    python plotVoronoi.py diff_data/a32x32_A0.5* --sf p32 -NP 10
will plot structure images for all gsd files in all folders that match diff_data/a32x32_A0.5* template, only in subfolders `p32`,
using 10 parallel processes.
Args:
--sf <fubfolder>: subfolder to process (e.g. p32)
--NP <number>: number of subprocesses to use for parallelization. Very efficient acceleration by a factor of <number>.
"""

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def VoronoiAreas(vor, box):
    a = np.ones(len(vor.point_region))*0.85
    for i, reg_ind in enumerate(vor.point_region):
        if vor.regions[reg_ind] != -1 and np.abs(vor.points[i,0]) <= 0.5*box.Lx and np.abs(vor.points[i,1]) <= 0.5*box.Ly:
            #print(vor.points[i,:])
            vertices_ind = vor.regions[reg_ind]
            vertices_r = vor.vertices[vertices_ind]
            #print(vertices_r)
            a[i] = PolyArea(vertices_r[:,0], vertices_r[:,1])
            #print(a[i])
    return a


def plot_structure(fpath):
    print(fpath)
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    fpath_words = fpath.split('/')
    folder_path = '/'.join(fpath_words[:-1]) + '/'
    plt.close('all')
    fig, ax = plt.subplots(1,1, figsize=(7,8))

    max_frame = 40


    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames_total = f_gsd.nframes
        if max_frame > n_frames_total:
            print('max_frame > n_frames_total')
            max_frames = n_frames_total
            #raise ValueError('frames beyond n_frames_total')
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])
        pos0 = f_gsd.read_chunk(frame=0, name='particles/position')
        cm0x = np.mean(pos0[:,0])
        cm0y = np.mean(pos0[:,1])
        pos_av = np.zeros(pos0.shape)
        pos_m1_cntn = pos0
        pos_m1_inbox = pos0
        for iframe in range(0, max_frame):
            pos = f_gsd.read_chunk(frame=iframe, name='particles/position')
            pos_cntn = md_tools.correct_jumps(pos, pos_m1_cntn, pos_m1_inbox, box.Lx, box.Ly)
            cmx = np.mean(pos_cntn[:,0])
            cmy = np.mean(pos_cntn[:,1])
            pos_cntn_mcm = np.zeros(pos.shape)
            pos_cntn_mcm[:,0] = pos_cntn[:,0] - (cmx - cm0x)
            pos_cntn_mcm[:,1] = pos_cntn[:,1] - (cmy - cm0y)
            pos_av += pos_cntn_mcm
            pos_m1_cntn = pos_cntn
            pos_m1_inbox = pos
        pos_av /= max_frame


    virtual_pos, virtual_ind = md_tools.create_virtual_layer(pos_av, box)

    vor = Voronoi(virtual_pos[:, 0:2])
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, line_width=1)
    a = VoronoiAreas(vor, box)
    norm = colors.Normalize(vmin=0.84, vmax=0.88)
    ax.scatter(virtual_pos[:,0], virtual_pos[:,1], c = a, norm=norm, s=150, cmap='inferno')

    ax.set_xlim([-0.5*box.Lx, 0.5*box.Lx])
    ax.set_ylim([-0.5*box.Ly, 0.5*box.Ly])

    
    labelfont = 26
    tickfont = labelfont - 4
    ax.set_xlabel('$x/a$', fontsize= labelfont)
    ax.set_ylabel('$y/a$', fontsize= labelfont)
    ax.tick_params(labelsize=tickfont)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    #fig.savefig(folder_path + 'Voronoi_' + fpath_words[-1] + '_' + fpath_words[-2] + '.png')
    fig.savefig(folder_path + 'Voronoi_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')
    plt.close('all')
    
    #Plot a snapshot
    fig, ax = plt.subplots(1,1, figsize=(7,8))
    ax.scatter(pos0[:,0], pos0[:,1], s=10)

    ax.set_xlim([-0.5*box.Lx, 0.5*box.Lx])
    ax.set_ylim([-0.5*box.Ly, 0.5*box.Ly])

    
    labelfont = 22
    tickfont = labelfont - 8
    ax.set_xlabel('$x/a$', fontsize= labelfont)
    ax.set_ylabel('$y/a$', fontsize= labelfont)
    ax.tick_params(labelsize=tickfont)
    #ax.set_aspect('equal')

    plt.tight_layout()
    fig.patch.set_alpha(alpha=1)
    fig.savefig(folder_path + 'snapshot_' + fpath_words[-1] + '_' + fpath_words[-2] + '.pdf')




def print_help():
    print('This script plots Voronoi tessellation for all .gsd files found in folders.')
    print('The .gsd files should be in the first-level subfolders in the specified folders')
    print('===========================================================')
    print('Usage: python plotVoronoi.py mobility_data/a32x32_* [--options]')
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
                if fname[-4:] == '.gsd' and fname[:6] == 'sample':
                    fnames.append(fname)
                    fpaths.append(sf[0] + '/' + fname)
            fnames.sort()
            fpaths.sort()
            print(fnames)
            p = Pool(Nproc, maxtasksperchild=1)
            p.map(plot_structure, fpaths)
            p.close()
            p.join()

