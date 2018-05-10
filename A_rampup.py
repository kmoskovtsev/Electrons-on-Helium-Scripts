from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import hoomd
import hoomd.md
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

curr_fname = inspect.getfile(inspect.currentframe())
curr_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

start_time = time.time()
hoomd.context.initialize('--mode=gpu');

## ==========================================
# Parse args
num_var = 11
Amin, Amax, AN, p, a, repeat_x, repeat_y, dt, regime, subfolder, gamma = (-1,)*num_var

if len(sys.argv) == num_var*2 + 1:
    for i in xrange(1, num_var*2 + 1, 1):
        if sys.argv[i] == '--Amin':
            Amin = float(sys.argv[i+1])
        if sys.argv[i] == '--Amax':
            Amax = float(sys.argv[i+1])
        if sys.argv[i] == '--AN':
            AN = int(sys.argv[i+1])
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
            if regime not in set(['crystal', 'random']):
                raise ValueError('regime must be one of: crystal, random')
        if sys.argv[i] == '--sf':
            subfolder = sys.argv[i+1]
        if sys.argv[i] == '--gamma':
            gamma = float(sys.argv[i+1])
else:
    raise RuntimeError("Not enough arguments (must be e.g. -A 2.4 -p 3")    
if Amin<0 or Amax<0 or AN<0 or p<0 or a<0 or repeat_x<0 or repeat_y<0 or dt<0 or regime<0 or subfolder<0:
    raise RuntimeError("Not enough valid arguments")
    
unit_M = 9.10938356e-31 # kg, electron mass
unit_D = 1e-6 # m, micron
unit_E = 1.38064852e-23 # m^2*kg/s^2
unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # = 2.568638150515e-10 s
print("unit_t = {} s".format(unit_t))
epsilon_0 = 8.854187817e-12 # F/m = C^2/(J*m), vacuum permittivity


#Charge through Gaussian units:
unit_Q = np.sqrt(unit_E*1e7*unit_D*1e2) # Coulombs
unit_Qe = unit_Q/4.8032068e-10 # e, unit charge in units of elementary charge e
#print("unit_Q = {:.10e} statC = {:.10e} e".format(unit_Q, unit_Qe))
e_charge = 1/unit_Qe # electron charge in units of unit_Q
#print("Elementary charge = {:.10e} unit_Q".format(e_charge))

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



width = 2000 # number of mesh points in x direction (real space)
height = 2000 # number of mesh points in y direction
eta = 0.5*a*max(2*repeat_x, repeat_y) # in units of distance
table_dir_path = '/home/moskovts/MD/table_force/'

table_f_name = 'r{:d}x{:d}_wh{:d}x{:d}a{:.2f}.dat'.format(repeat_x, repeat_y, width, height, a)


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


## ==============================================
# Add integrator and periodic

all = hoomd.group.all();
hoomd.md.integrate.mode_standard(dt=dt);
langevin = hoomd.md.integrate.langevin(group=all, kT= 1., seed=987);

periodic = hoomd.md.external.periodic_cos()
periodic.force_coeff.set('A', A=0, i=0, p=p, phi=np.pi)
periodic.disable()

crystal_state = system.take_snapshot(all=True)
## =======================================================================
# Prepare random (liquid) state if required:
if regime == 'random':
    n_s = 2/a1_unit[0]/a2_unit[1]
    gamma_to_T = e_charge**2*np.sqrt(np.pi*n_s)
    langevin.set_params(kT=gamma_to_T/10)
    hoomd.run(1000, quiet=True)
    random_state = system.take_snapshot(all=True)
    random_state.particles.velocity[:] = random_state.particles.velocity[:]*0

## =======================================================================

snap_period = 500
#N_snaps = 200
#N_therm = int(20*np.pi*a**1.5/dt) # n_steps to thermalize ~ 40 periods of oscillation
N_therm = int(200*np.pi*a**1.5/dt) # n_steps to thermalize ~ 40 periods of oscillation



hoomd.md.integrate.mode_standard(dt=dt);
diff_path = '/mnt/home/moskovts/MD/diffusion_test_data/'
log_file = 'hpcc_diff_log.txt'

time.sleep(60*random.random()) #add random wait to avoid identical timestamps
timestamp = datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d-%H%M%S")

#create working directory
if not os.path.isdir(diff_path):
    os.mkdir(diff_path)

diff_path = diff_path + subfolder + '/'

if not os.path.isdir(diff_path):
    os.mkdir(diff_path)

folder_path = diff_path + timestamp + '/'
#create individual directory avoiding duplicate timestamps
n_dup = 1
dup_flag = True
timestamp_try = timestamp
folder_path = diff_path + timestamp_try + '/'
while dup_flag:
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        dup_flag = False
    else:
        timestamp_try = timestamp + '_{}'.format(n_dup)
        n_dup += 1
        folder_path = diff_path + timestamp_try + '/'

print(folder_path)

#copy this script and table force into the results folder:
copyfile(curr_path + '/' + curr_fname, folder_path + curr_fname)
if not os.path.isfile(diff_path + table_f_name):
    copyfile(table_dir_path + table_f_name, diff_path + table_f_name)

# make a list of filenames for all gammas
fl = open(folder_path + 'list.txt', 'w')
fl.write('# N_therm = {}; snap_period = {};\n'.format(N_therm, snap_period))
fl.write('# Periodic is on, A ={}, p={}\n'.format(Amax, p))
fl.write('# Amin ={}, A_max={}, p={}\n'.format(Amin, Amax, p))
fl.write('# a ={}, repeat_x={}, repeat_y={}\n'.format(a, repeat_x, repeat_y))
fl.write('# Ewald: width ={}, height={}, eta={}\n'.format(width, height, eta))
fl.write('# regime: {}\n'.format(regime))
fl.write('# file    Gamma    T    dt\n')

A_list = np.linspace(Amin, Amax, AN)

f_list = []
n_s = 2/a1_unit[0]/a2_unit[1]
gamma_to_T = e_charge**2*np.sqrt(np.pi*n_s)    
T = gamma_to_T/gamma
for i in range(len(A_list)):
    f_list.append('{:05d}.gsd'.format(i))
    fl.write(f_list[-1] + '\t' + '{:.8f}\t'.format(gamma) + '{:.8f}\t'.format(T) + '{:.8f}\t'.format(dt) +\
            '{:.8f}'.format(A_list[i]) + '\n')
fl.close()


if not os.path.isfile(diff_path + log_file):
    with open(diff_path + log_file, 'w') as fl:
        fl.write('#timestamp\ta\trepeat_x/repeat_y\tp\tA\tdt\tewald_width/ewald_height\teta\tregime\n')
with open(diff_path + log_file, 'a') as fl:
    fl.write(('{}\t'*9 + '\n').format(timestamp, a, '{}x{}'.format(repeat_x, repeat_y),\
            p, Amax, dt, '{}x{}'.format(width, height), eta, regime))

    
if regime == 'random':
    system.restore_snapshot(random_state)
elif regime == 'crystal':
    system.restore_snapshot(crystal_state)
periodic.enable()
langevin.set_params(kT = T)
##=======================================================================
# Calculate trajectories
for i, A in enumerate(A_list):
    periodic.force_coeff.set('A', A=A, i=0, p=p, phi=np.pi)
    gsd_dump = hoomd.dump.gsd(filename=folder_path + f_list[i], period=snap_period, group=hoomd.group.all(), phase=0)
    #Thermalize
    print('Thermalizing at T = {:.2} K, A = {:.2}... ({:d} of {:d})'.format(T, A, i, AN))
    hoomd.run(N_therm)
    gsd_dump.disable()
periodic.disable()


## ===================================================================
# Plot the results
D_x_list, D_y_list, gamma_list, T_list = md_tools.diffusion_from_gsd(folder_path, useframes = 100)

data = {'D_x_list': D_x_list, 'D_y_list': D_y_list, 'gamma_list': gamma_list}
f = open(folder_path + 'Dxy_gamma_' + timestamp + '.dat', 'wb')
pickle.dump(data, f)
f.close()
text_list = [timestamp, '$p = {}$'.format(p), '$A = {}$'.format(A)]

fig, ax1, ax2 = md_tools.plot_DxDy(D_x_list, D_y_list, gamma_list, timestamp, text_list,\
                                          folder = diff_path)
fig.savefig(diff_path + timestamp + '/' + timestamp + '_diff.png')

# Plot on one canvas
fig2, axx = plt.subplots(1,1, figsize = (8,6))
axx.scatter(gamma_list, D_x_list, label='$D_x$')
axx.scatter(gamma_list, D_y_list, label='$D_y$')
axx.legend()
fig2.savefig(diff_path + timestamp + '/' + timestamp + 'diff_single.png')
## ===================================================================
end_time = time.time()
print("Elapsed time: {} s".format(end_time - start_time))
