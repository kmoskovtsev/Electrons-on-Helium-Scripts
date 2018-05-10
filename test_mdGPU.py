
from __future__ import division
import hoomd.md
import hoomd
import hoomd.md
#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.special import erf as erf
import os
import datetime
import ewald_module as em


hoomd.context.initialize('--mode=gpu');

base_dir = '/mnt/home/moskovts/MD/'
traj_file = 'gpu_test003.dsg'
snap_period = 10
N_snaps = 300

unit_M = 9.10938356e-31 # kg, electron mass
unit_D = 1e-6 # m, micron
unit_E = 1.38064852e-23 # m^2*kg/s^2
unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # = 2.568638150515e-10 s
print("unit_t = {} s".format(unit_t))
epsilon_0 = 8.854187817e-12 # F/m = C^2/(J*m), vacuum permittivity

# Charge through SI (the end result is the same as in CGS):
#unit_Q = np.sqrt(4*np.pi*epsilon_0*unit_D*unit_E) # Coulombs
#unit_Qe = unit_Q/1.60217662e-19 # e, unit charge in units of elementary charge e
#print("unit_Q = {:.10e} C = {:.10e} e".format(unit_Q, unit_Qe))
#e_charge = 1/unit_Qe # electron charge in units of unit_Q
#print("Elementary charge = {:.10e} unit_Q".format(e_charge))

#Charge through Gaussian units:
unit_Q = np.sqrt(unit_E*1e7*unit_D*1e2) # Coulombs
unit_Qe = unit_Q/4.8032068e-10 # e, unit charge in units of elementary charge e
print("unit_Q = {:.10e} statC = {:.10e} e".format(unit_Q, unit_Qe))
e_charge = 1/unit_Qe # electron charge in units of unit_Q
print("Elementary charge = {:.10e} unit_Q".format(e_charge))


## Create lattice ================================================================

a1_unit = np.array([0.2*np.sqrt(3)*0.5, 0, 0])
a2_unit = np.array([0, 0.2, 0]) # to accomodate hexagonal lattice
#a2_unit = np.array([0, 0.2, 0])
a3 = np.array([0, 0, 1])

repeat_x = 32
repeat_y = 32

# Create a unit cell with one electron:
uc = hoomd.lattice.unitcell(N = 1,
                            a1 = a1_unit,
                            a2 = a2_unit,
                            a3 = a3,
                            dimensions = 2,
                            position = [[0,0,0]],
                            type_name = ['A'],
                            mass = [1.0],
                            charge = [e_charge],
                            diameter = [0.08],
                            moment_inertia = [[0.0, 0.0, 0.0]],
                            orientation = [[1.0, 0, 0, 0]]);
system = hoomd.init.create_lattice(uc, [repeat_x, repeat_y])

a1 = a1_unit*repeat_x
a2 = a2_unit*repeat_y

## Add Ewald ==============================================================================

width = 1000 # number of mesh points in x direction (real space)
height = 1000 # number of mesh points in y direction
eta = 0.8 # in units of distance
dir_path = base_dir + 'table_force/'

#create mesh covering quarter-unit-cell:
mesh_x, mesh_y = em.mesh_quarter_uc(a1, a2, width, height)

#short- and long-range potential energy:
V_s = em.V_short(mesh_x, mesh_y, e_charge, eta)
V_l = em.V_long(mesh_x, mesh_y, e_charge, eta)

#short- and long-range force
F_s = em.F_short(mesh_x, mesh_y, e_charge, eta)
F_l = em.F_long(mesh_x, mesh_y, e_charge, eta)

#Write potential and force to file
f_name = em.export_to_file(dir_path, mesh_x, mesh_y, V_s + V_l, F_s + F_l)
print(f_name)
table = hoomd.md.pair.table2D(width, height, 0.5*a1[0], 0.5*a2[1])
table.set_from_file(dir_path + f_name)

print('Ewald added to HOOMD')


#table.disable()


## Initiate integrator
dt = 0.001

all = hoomd.group.all();
hoomd.md.integrate.mode_standard(dt=dt);
langevin = hoomd.md.integrate.langevin(group=all, kT=1.2, seed=987);

#gsd_dump = hoomd.dump.gsd(filename=base_dir + traj_file, period=snap_period, group=hoomd.group.all(), phase=0)
hoomd.run((N_snaps + 1)*snap_period)

#gsd_dump.disable()
