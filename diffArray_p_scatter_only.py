from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import hoomd
import hoomd.md
import gsd
#import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf as erf
import os
import sys
import datetime
import ewald_module as em
import md_tools27 as md_tools
import time
import pickle
from shutil import copyfile
import inspect, os
import random
import ripplon_scattering_module as rsm
import phonon_scattering_module as psm



def scattering_wrapper(kmin, kmax, Nk, N_W, N_theta_rpl, Ntheta, NY, T):
    k_arr = np.linspace(kmin, kmax, Nk)
    #Ripplon scattering parameters
    rsm.init(unit_M, unit_D, unit_E)
    W_resampled = np.linspace(0,1, N_W)
    theta_arr_rpl = np.linspace(0, 2*np.pi, N_theta_rpl)
    w_k, W_inv, W_cumul, vmin, vmax = rsm.scattering_parameters(T, kmin, kmax, Nk, N_W, N_theta_rpl)
    
    #Phonon scattering parameters
    fname = 'fkkpt_kmin{:.0f}_kmax{:.0f}_Nk{:d}_Ntheta{:d}.dat'.format(kmin, kmax, Nk, Ntheta)
    print(fname)
    try:
        fkkp_t, kmin, kmax = psm.read_fkkp_t_from_file('/mnt/home/moskovts/MD/fkkp_t_tables/' + fname)
    except:
        raise RuntimeError("The 'fkkp_t file for these parameters does not exist. You can create it by " +\
                           "running compute_bare_fkkp_t and saving by write_fkkp_t_to_file")
    # Calculate final tables for phonon scattering, with thermal factors
    psm.init(unit_M, unit_D, unit_E)
    bare_fkkp = psm.compute_bare_fkkp(fkkp_t, k_arr)
    Y_inv = psm.compute_Yinv(fkkp_t, NY)
    fkkp = psm.dress_fkkp(bare_fkkp, k_arr, T)
    wk_ph = psm.compute_total_wk(fkkp, k_arr)
    F_inv = hbar/m_e*psm.compute_cumulative_Fkkp_inv(fkkp, k_arr, tol=0.1)
    return w_k, W_inv, wk_ph, F_inv, Y_inv, vmin, vmax

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

#Data collection parameters
snap_period = 1000

#Ewald table dimentions
width = 2000 # number of mesh points in x direction (real space)
height = 2000 # number of mesh points in y direction

#Integrator parameters
kmin = 1
kmax = 150
Nk = 200
N_theta_rpl = 1000
N_W = 400
Ntheta = 20
NY = 10

N_therm = 1.5e6 # n_steps to reach energy balance, roughly reciprocal phonon scattering rate
    
    
    
    
curr_fname = inspect.getfile(inspect.currentframe())
curr_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

start_time = time.time()


## ==========================================
# Parse args
num_var = 11
A, p, a, repeat_x, repeat_y, dt, regime, subfolder, gamma, data_steps, coulomb = (-1,)*num_var

print(sys.argv)

if len(sys.argv) == num_var*2 + 1:
    for i in xrange(1, num_var*2 + 1, 1):
        if sys.argv[i] == '-A':
            A = float(sys.argv[i+1])
        if sys.argv[i] == '-p':
            p = int(sys.argv[i+1])
        if sys.argv[i] == '-a':
            a = float(sys.argv[i+1])
        if sys.argv[i] == '--rx':
            repeat_x = int(sys.argv[i+1])
        if sys.argv[i] == '--ry':
            repeat_y = int(sys.argv[i+1])
        if sys.argv[i] == '--dt':
            dt = float(sys.argv[i+1])
        if sys.argv[i] == '--reg':
            regime = sys.argv[i+1]
            if regime not in set(['crystal', 'thermal']):
                raise ValueError('regime must be one of: crystal, thermal')
        if sys.argv[i] == '--sf':
            subfolder = sys.argv[i+1]
        if sys.argv[i] == '--gamma':
            gamma = float(sys.argv[i+1])
        if sys.argv[i] == '--dst':
            #how many steps for data collection
            data_steps = int(sys.argv[i+1])
        if sys.argv[i] == '--coulomb':
            coulomb = sys.argv[i+1]
else:
    raise RuntimeError("Not enough arguments (must be e.g. -A 2.4 -p 3 ...")    
if A < 0 or p < 0 or a < 0 or repeat_x < 0 or repeat_y < 0 or dt < 0 or regime < 0 or subfolder < 0 or gamma < 0 or data_steps < 0 or coulomb < 0:
    raise RuntimeError("Not enough valid arguments")
    
if coulomb.lower()[0] == 'y':
    try:
        hoomd.context.initialize('--mode=gpu')
    except:
        time.sleep(10)
        hoomd.context.initialize('--mode=gpu')
else:
    hoomd.context.initialize('--mode=cpu')
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


a1_unit = np.array([np.sqrt(3)*a, 0, 0])
a2_unit = np.array([0, a, 0]) # to accomodate hexagonal lattice
a3 = np.array([0, 0, 1])


# Create a unit cell with one electron:
uc = hoomd.lattice.unitcell(N = 2,
                            a1 = a1_unit,
                            a2 = a2_unit,
                            a3 = a3,
                            dimensions = 2,
                            position = [[0,0,0], [a*np.sqrt(3)*0.5, 0.5*a, 0]],
                            type_name = ['A', 'A'],
                            mass = [1.0, 1.0],
                            charge = [e_charge, e_charge],
                            diameter = [0.01*a, 0.01*a],
                            moment_inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            orientation = [[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
system = hoomd.init.create_lattice(uc, [repeat_x, repeat_y])

a1 = a1_unit*repeat_x
a2 = a2_unit*repeat_y


# ==============================================
# ADD EWALD

eta = 0.5*a*max(2*repeat_x, repeat_y) # in units of distance
table_dir_path = '/home/moskovts/MD/table_force/'

table_f_name = 'r{:d}x{:d}_wh{:d}x{:d}a{:.2f}.dat'.format(repeat_x, repeat_y, width, height, a)

coulomb_status = 'Coulomb off'
if coulomb.lower()[0] == 'y':
    if not os.path.isfile(table_dir_path + table_f_name):
        print('Calculating new Ewald force table, saving to: {}'.format(table_f_name))
        #create mesh covering quarter-unit-cell:
        mesh_x, mesh_y = em.mesh_quarter_uc(a1, a2, width, height)

        #short- and long-range potential energy:
        V_s = em.V_short(mesh_x, mesh_y, e_charge, eta)
        V_l = em.V_long(mesh_x, mesh_y, e_charge, eta)

        #short- and long-range force
        F_s = em.F_short(mesh_x, mesh_y, e_charge, eta)
        F_l = em.F_long(mesh_x, mesh_y, e_charge, eta)

        #Write potential and force to file
        table_f_name = em.export_to_file(table_dir_path, mesh_x, mesh_y, V_s + V_l, F_s + F_l, filename=table_f_name)
        print(table_f_name)

    table = hoomd.md.pair.table2D(width, height, 0.5*a1[0], 0.5*a2[1])
    table.set_from_file(table_dir_path + table_f_name)
    print('Add Ewald force to HOOMD, using file: {}'.format(table_f_name))
    coulomb_status = 'Coulomb on'


## =======================================================================
# Add integrator

all = hoomd.group.all()
hoomd.md.integrate.mode_standard(dt=dt)

n_s = 2/a1_unit[0]/a2_unit[1]
T_gamma = e_charge**2*np.sqrt(np.pi*n_s)
T = T_gamma/gamma

w_k, W_inv, wk_ph, F_inv, Y_inv, vmin, vmax = scattering_wrapper(kmin, kmax, Nk, N_W, N_theta_rpl, Ntheta, NY, T)
wk_ph *= 20 # Increase the energy relaxation rate artificially. Factor 20 corresponds to roughly 10^6 1/s rate.
k_arr = np.linspace(kmin, kmax, Nk)

scatter = hoomd.md.integrate.custom_scatter2D(group=all, Nk=Nk, NW=N_W, NY=NY, seed=987)
scatter.set_tables(w_k, W_inv,  wk_ph, F_inv, Y_inv, vmin, vmax)

## =======================================================================
# Add Periodic

periodic = hoomd.md.external.periodic_cos()
periodic.force_coeff.set('A', A=A, i=0, p=p, phi=np.pi)
periodic.disable()
## =======================================================================
# Prepare crystal state if required:
if regime == 'crystal':
    snapshot = system.take_snapshot(all=True)
    vel = snapshot.particles.velocity[:]*0
    angles = np.random.random(vel.shape[0])*2*np.pi
    W_rand = np.random.random(vel.shape[0])
    if coulomb.lower()[0] == 'y':
        v_abs = np.sqrt(-4*T*np.log(1 - W_rand))
    else:
        v_abs = np.sqrt(-2*T*np.log(1 - W_rand))
    vel[:,0] = v_abs*np.cos(angles)
    vel[:,1] = v_abs*np.sin(angles)
    vel = vel - np.mean(vel, axis = 0)
    snapshot.particles.velocity[:] = vel
    system.restore_snapshot(snapshot)
    hoomd.run(10000, quiet=True)
    crystal_state = system.take_snapshot(all=True)
## =======================================================================
# Prepare random (liquid) state if required:
if regime == 'thermal':
    snapshot = system.take_snapshot(all=True)
    vel = snapshot.particles.velocity[:]*0
    angles = np.random.random(vel.shape[0])*2*np.pi
    W_rand = np.random.random(vel.shape[0])
    T_disorder = 0.8
    if coulomb.lower()[0] == 'y':
        v_abs = np.sqrt(-4*T_disorder*np.log(1 - W_rand))
    else:
        v_abs = np.sqrt(-2*T_disorder*np.log(1 - W_rand))
    vel[:,0] = v_abs*np.cos(angles)
    vel[:,1] = v_abs*np.sin(angles)
    vel = vel - np.mean(vel, axis = 0)
    snapshot.particles.velocity[:] = vel
    system.restore_snapshot(snapshot)
    hoomd.run(10000, quiet=True)
    thermal_state=system.take_snapshot(all=True)


## =======================================================================
# Create folders
general_path = '/mnt/home/moskovts/MD/diff_data_new/'

timestamp = datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d-%H%M%S")

#create working directory
if not os.path.isdir(general_path):
    os.mkdir(general_path)

general_path = general_path + subfolder + '/'

if not os.path.isdir(general_path):
    os.mkdir(general_path)

folder_path = general_path + 'p{:02d}'.format(p) + '/'
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
print(folder_path)


#copy this script and table force into the results folder:
copyfile(curr_path + '/' + curr_fname, folder_path + curr_fname)
if (not os.path.isfile(general_path + table_f_name)) and coulomb.lower()[0] == 'y':
    copyfile(table_dir_path + table_f_name, general_path + table_f_name)

# make a log file
if not os.path.isfile(folder_path + 'log.txt'):
    fl = open(folder_path + 'log.txt', 'w')
    fl.write(timestamp + '\n')
    fl.write('# N_therm= {} ; snap_period= {} ; data_steps= {} \n'.format(N_therm, snap_period, data_steps))
    fl.write('# Periodic is on, A = {} , p= {} \n'.format(A, p))
    fl.write('# a = {} , repeat_x= {} , repeat_y= {} \n'.format(a, repeat_x, repeat_y))
    fl.write('# Ewald: width= {} , height= {} , eta= {} \n'.format(width, height, eta))
    fl.write('# regime: {} \n'.format(regime))
    fl.write('# ' + coulomb_status + '\n')
    fl.write('# Gamma= {} ; T= {} ;  dt= {}\n'.format(gamma, T, dt))
    fl.write('# T_gamma = {}'.format(T_gamma))
    fl.close()

#gsd filenames:
data_fname = 'G_{:08.4f}_.gsd'.format(gamma)
sample_fname = 'sample_G_{:08.4f}_.gsd'.format(gamma)

##=======================================================================
# Calculate trajectories
if A > 0:
    periodic.force_coeff.set('A', A=A, i=0, p=p, phi=np.pi)
    periodic.enable()
#Calculate trajectories for given E
if regime == 'thermal':
    system.restore_snapshot(thermal_state)
elif regime == 'crystal':
    system.restore_snapshot(crystal_state)

#Bring to steady state
print('Thermalize at T = {:.4f} K...)'.format(T))
hoomd.run(N_therm)

#Take snapshots
gsd_dump = hoomd.dump.gsd(filename=folder_path + data_fname, period=snap_period, group=hoomd.group.all(), phase=0,\
                            static=['attribute', 'topology'])
hoomd.run(data_steps)
gsd_dump.disable()

#Record sample trajectories with higher resolution
gsd_dump = hoomd.dump.gsd(filename=folder_path + sample_fname, period=25, group=hoomd.group.all(), phase=0,\
                            static=['attribute', 'topology'])
hoomd.run(10000)
gsd_dump.disable()

## ===================================================================
end_time = time.time()
print("Elapsed time: {} s".format(end_time - start_time))
