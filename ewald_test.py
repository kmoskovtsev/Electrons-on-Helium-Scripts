import ewald_module as em
import numpy as np

def ai_from_mesh(mesh_x, mesh_y):
    """
    ai_from_mesh(mesh_x, mesh_y)
        Determine unit cell vectors a1 and a2 from the mesh grid.
    """
    height, width = mesh_x.shape
    lx = 2*(mesh_x[0, 0] + mesh_x[0, width -1 ])
    ly = 2*(mesh_y[0,0] + mesh_y[height - 1, 0])
    print(lx)
    print(ly)
    a1 = np.array([lx, 0])
    a2 = np.array([0, ly])
    return a1, a2, lx, ly

def test_ewald(eta, i,j):
    print('=====================================================')
    print('eta = {}'.format(eta))
    #short- and long-range force
    F_coul = F_coulomb(mesh_x, mesh_y, e_charge, 10,10)
    F_coul_abs = np.sqrt(F_coul[:,:,0]**2 + F_coul[:,:,1]**2)

    F_s = em.F_short(mesh_x, mesh_y, e_charge, eta)
    F_l = em.F_long(mesh_x, mesh_y, e_charge, eta)

    V_s = em.V_short(mesh_x, mesh_y, e_charge, eta)
    V_l = em.V_long(mesh_x, mesh_y, e_charge, eta)


    F_t = F_s + F_l
    #F_t = em.F_s_ri([0,0], mesh_x, mesh_y, eta)*e_charge**2
    xx = mesh_x[j, i]
    yy = mesh_y[j, i]
    rr = np.sqrt(xx**2 + yy**2)
    F_abs = np.sqrt(F_t[:,:,0]**2 + F_t[:,:,1]**2)
    F_l_abs = np.sqrt(F_l[:,:,0]**2 + F_l[:,:,1]**2)
    print('\nF_abs = {}\n'.format(F_abs[j, i]))
    print('F_t_x = {}'.format(F_t[j, i, 0]))
    print('F_t_y = {}'.format(F_t[j, i, 1]))
    print('F_l_abs = {}'.format(F_l_abs[j, i]))
    print('F_l_x = {}'.format(F_l[j, i, 0]))
    print('F_l_y = {}'.format(F_l[j, i, 1]))
    #print("x = {}, y = {}, r = {}".format(xx, yy, rr))
    print("simple coulomb sum: {}".format(F_coul_abs[j, i]))
    print('F_coul_x = {}'.format(F_coul[j, i, 0]))
    print('F_coul_y = {}'.format(F_coul[j, i, 1]))
    
    print('V_l = {}'.format(V_l[j,i]))
    print('V_s = {}'.format(V_s[j,i]))
    print('V_l + V_s = {}'.format(V_l[j,i] + V_s[j,i]))

    
def r_from_ri(ri, mesh_x, mesh_y):
    #distance from ri to r (both 2D vectors)
    return np.sqrt((mesh_x - ri[0])**2 + (mesh_y - ri[1])**2)
    
    
def F_coulomb_ri(ri, mesh_x, mesh_y):
    x_ri = mesh_x - ri[0]
    y_ri = mesh_y - ri[1]
    r_ri = r_from_ri(ri, mesh_x, mesh_y)
    term1 = 1/r_ri**3
    return np.dstack((term1*x_ri, term1*y_ri))
    
def F_coulomb(mesh_x, mesh_y, charge, nx, ny):
    """
        Total hard-core Coulomb force force from Ewald procedure
        return: 3D array res, with F_x = res[:, :, 0], F_y = res[:, :, 1].
        F_x and F_y evaluated on the mesh.
    """
    # Retrieve geometry:
    (a1, a2, lx, ly) = ai_from_mesh(mesh_x, mesh_y)
    
    
    print("{} x {} layers of images used for Coulomb force".format(nx, ny))
    
    # init F:
    F = np.dstack((0*mesh_x, 0*mesh_x))
    
    # sum over all significant images:
    for i in range(-(nx - 1), nx + 1):
        for j in range(-(ny - 1), ny + 1):
            F += F_coulomb_ri(i*a1 + j*a2, mesh_x, mesh_y)
    F *= charge**2
    return F

unit_M = 9.10938356e-31 # kg, electron mass
unit_D = 1e-6 # m, micron
unit_E = 1.38064852e-23 # m^2*kg/s^2
unit_t = np.sqrt(unit_M*unit_D**2/unit_E) # = 2.568638150515e-10 s
print("unit_t = {} s".format(unit_t))
epsilon_0 = 8.854187817e-12 # F/m = C^2/(J*m), vacuum permittivity

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

repeat_x = 10
repeat_y = 20

a1 = a1_unit*repeat_x
a2 = a2_unit*repeat_y

width = 50 # number of mesh points in x direction (real space)
height = 50 # number of mesh points in y direction
#create mesh covering quarter-unit-cell:
mesh_x, mesh_y = em.mesh_quarter_uc(a1, a2, width, height)





    
i, j = (3, 1)
test_ewald(2 ,i,j)
test_ewald(5, i, j)
test_ewald(10, i, j)
