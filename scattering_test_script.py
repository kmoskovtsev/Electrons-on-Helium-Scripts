from __future__ import division
import hoomd
import hoomd.md
import gsd.fl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.special import erf as erf
import os
import datetime
import ewald_module as em
import md_tools27
import ripplon_scattering_module as rsm


hoomd.context.initialize('--mode=gpu');

print('================ UNITS ==================')
unit_M = 9.10938356e-31 # kg, electron mass
unit_D = 1e-6 # m, micron
unit_E = 1.38064852e-23 # m^2*kg/s^2
print('unit_M = {:e} kg'.format(unit_M))
print('unit_D = {:e} m'.format(unit_D))
print('unit_E = {:e} J'.format(unit_E))
unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # = 2.568638150515e-10 s
print("unit_t = {} s".format(unit_t))
epsilon_0 = 8.854187817e-12 # F/m = C^2/(J*m), vacuum permittivity
hbar = 1.0545726e-27/(unit_E*1e7)/unit_t

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


a = 1
a1_unit = np.array([np.sqrt(3)*a, 0, 0])
a2_unit = np.array([0, a, 0]) # to accomodate hexagonal lattice
a3 = np.array([0, 0, 1])

repeat_x = 100
repeat_y = 100

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
                            diameter = [0.05*a, 0.05*a],
                            moment_inertia = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            orientation = [[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
system = hoomd.init.create_lattice(uc, [repeat_x, repeat_y])

a1 = a1_unit*repeat_x
a2 = a2_unit*repeat_y

dt = 0.01
all = hoomd.group.all()
hoomd.md.integrate.mode_standard(dt=dt)

T = 0.2
k_min = 1
k_max = 500
N_theta = 1000
N_k = 1000
N_W = 400
rsm.init(unit_M, unit_D, unit_E)
W_resampled = np.linspace(0,1, N_W)
theta_arr = np.linspace(0, 2*np.pi, N_theta)
w_k, theta_resampled, W_cumul, vmin, vmax = rsm.scattering_parameters(T, k_min, k_max, N_k, N_W, N_theta)

scatter = hoomd.md.integrate.custom_scatter2D(group=all, Nk=N_k, NW=N_W, seed=987)

scatter.set_tables(w_k, theta_resampled, vmin, vmax)

k = 200
snapshot = system.take_snapshot(all=True)
snapshot.particles.velocity[:] = snapshot.particles.velocity[:]*0
snapshot.particles.velocity[:,0] = snapshot.particles.velocity[:,0] + hbar*k
system.restore_snapshot(snapshot)

#gsd_dump = hoomd.dump.gsd(filename='test_scatter_gpu_k{}.gsd'.format(k), period=1, group=hoomd.group.all(), phase=0, overwrite=True,\
#                          static=['attribute', 'topology'])
n_steps = 1000
hoomd.run(n_steps)
#gsd_dump.disable()


