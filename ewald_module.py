from scipy.special import erfc as erfc
import numpy as np
import datetime
import os

def __init__():
    pass

def mesh_quarter_uc(a1, a2, width, height):
    """
    mesh_quarter_uc(a1, a2, width, height)
        Create meshgrid covering a quarter-unit-cell from (0,0) to (0.5*a1, 0.5*a2).
        Unit cell must be rectangular.
        width and height determine the number of points in x and y directions.
        The mesh does not include the boundaries of the covered area; the outermost layer of
        points is half-step distance from the boundary.
    """
    # Check the input:
    if np.dot(a1, a2) != 0:
        raise ValueError("a1 and a2 should be perpendicular!")
    if width < 1 or np.floor(width) != width \
    or height < 1 or np.floor(height) != height:
        raise ValueError("width and height must be positive integers")
        
    # grid steps in x and y directions:
    h1 = 0.5*a1[0]/float(width)
    h2 = 0.5*a2[1]/float(height)

    half_a1_range = np.linspace(0.5*h1, 0.5*a1[0] - 0.5*h1, width)
    half_a2_range = np.linspace(0.5*h2, 0.5*a2[1] - 0.5*h2, height)

    mesh_x, mesh_y = np.meshgrid(half_a1_range, half_a2_range)
    return mesh_x, mesh_y

def ai_from_mesh(mesh_x, mesh_y):
    """
    ai_from_mesh(mesh_x, mesh_y)
        Determine unit cell vectors a1 and a2 from the mesh grid.
    """
    height, width = mesh_x.shape
    lx = 2*(mesh_x[0, 0] + mesh_x[0, width -1 ])
    ly = 2*(mesh_y[0,0] + mesh_y[height - 1, 0])
    a1 = np.array([lx, 0])
    a2 = np.array([0, ly])
    return a1, a2, lx, ly

def r_from_ri(ri, mesh_x, mesh_y):
    #distance from ri to r (both 2D vectors)
    return np.sqrt((mesh_x - ri[0])**2 + (mesh_y - ri[1])**2)

def phi_s(r, eta):
    #short-range potential
    return erfc(r/2./eta)/r

def V_short(mesh_x, mesh_y, charge, eta):
    """ short_V(mesh_x, mesh_y, charge, width, height, eta)
            calculate short range potential energy on the mesh.
            - mesh_x, mesh_y should cover the interior of the quarter-unit-cell from (0,0) to (0.5*a1, 0.5*a2)
            The mesh should be prepared with mesh_quarter_uc(...) function.
            - charge: electron charge.
            - eta: separation length in Ewald summation
            
            This function determines how many images we need to include to get relative accuracy < 1e-10.
            Images are included in layers surrounding the unit cell.
    """
    # Retrieve geometry:
    (a1, a2, lx, ly) = ai_from_mesh(mesh_x, mesh_y)
    det_a = a1[0]*a2[1] - a1[1]*a2[0] # determinant of a-matrix
    # Determine which periodic images give significant contribution
    # These formula ensure exp( - r/eta) < 1e-10 for contributions from the most
    # distant significant images to any point in unit cell.
    # nx, ny - number of images in each direction (e.g. ny above and ny below)
    nx, ny = (1, 1) # arbitrary starting point
    accuracy_x = 1
    accuracy_y = 1
    while accuracy_x > 1e-25 and nx < 1000:
        nx += 1
        accuracy_x = erfc(0.5*(nx - 0.5)*lx/eta)
    while accuracy_y > 1e-25 and ny < 1000:
        ny += 1
        accuracy_y = erfc(0.5*(ny - 0.5)*ly/eta)
    
    accuracy = max(accuracy_x, accuracy_y)
    print("*** Short-Range Potential ***")
    print("{} x {} layers of images used for short-range potential".format(nx, ny))
    print("{:.3e} relative accuracy is achieved in short-range potential".format(accuracy))
    #init V
    V = mesh_x*0;
    
    # sum over all significant images:
    for i in range(-(nx - 1), nx + 1):
        for j in range(-(ny - 1), ny + 1):
            r_ij = r_from_ri(i*a1 + j*a2, mesh_x, mesh_y) # distance from image (i,j) to each grid point
            V += phi_s(r_ij, eta)
    V -= 4*np.sqrt(np.pi)*eta/det_a
    V *= charge**2
    return V
    

def V_long(mesh_x, mesh_y, charge, eta):
    """
    Long-range part of the potential in Ewald summation evaluated on mesh points from (0,0) to (0.5*a1, 0.5*a2).
    n1 = number of k-points to sum in direction 1
    n2 = number of k-points to sum in direction 2
    Particles' charge.
    """
    # Retrieve geometry:
    (a1, a2, lx, ly) = ai_from_mesh(mesh_x, mesh_y)
    
    
    deta = a1[0]*a2[1] - a1[1]*a2[0] # determinant of a-matrix
    if deta == 0:
        raise ValueError("determinant of (a1, a2) matrix must be nonzero")
    #reciprocal lattice vectors:
    b1 = 2*np.pi/deta*np.array([a2[1], -a2[0]])
    b2 = 2*np.pi/deta*np.array([-a1[1], a1[0]])
    
    # Determine n1 and n2, numbers of k-points to sum in each direction:
    n1, n2 = 2, 2 # arbitrary starting point
    accuracy_x = 1
    accuracy_y = 1
    while accuracy_x > 1e-25 and n1 < 1000:
        n1 += 1
        accuracy_x = erfc(eta*n1*np.sqrt(b1[0]**2 + b1[1]**2))
    while accuracy_y > 1e-25 and n2 < 1000:
        n2 += 1
        accuracy_y = erfc(eta*n2*np.sqrt(b2[0]**2 + b2[1]**2))
    
    accuracy = max(accuracy_x, accuracy_y)
    print("*** Long-Range Potential ***")
    print("n1 = {} and  n2 = {} k-points used in long-range potential".format(n1, n2))
    print("{:.3e} relative accuracy is achieved in long-range potential".format(accuracy))
    
    V = 0*mesh_x
    # sum over k-vectors
    # could sum over fewer values and use symmetry, but not for now
    for i1 in range(-(n1 - 1), n1):
        for i2 in range(-(n2 - 1), n2):
            # for all k != 0
            if not (i1 == 0 and i2 == 0):
                k = i1*b1 + i2*b2
                k_abs = np.sqrt(k[0]**2 + k[1]**2)
                V += np.cos(k[0]*mesh_x + k[1]*mesh_y)*erfc(k_abs*eta)/k_abs
    V *= 2*np.pi*charge**2/deta
    return V
    

def F_long(mesh_x, mesh_y, charge, eta):
    """
    F_long(mesh_x, mesh_y, charge, eta)
    
    Long-range part of the force F created by a particle located at the origin,
    and by all its periodic images.
    
    eta - Ewald threshold (short/long range);
    a1, a2 - lattice vectors;
    mesh_x, mesh_y - mesh grid.
    
    return: 3D array res, with F_x = res[:, :, 0], F_y = res[:, :, 1].
    F_x and F_y evaluated on the mesh.
    """
    # Retrieve geometry:
    (a1, a2, lx, ly) = ai_from_mesh(mesh_x, mesh_y)
    
    det_a = a1[0]*a2[1] - a1[1]*a2[0] # determinant of a-matrix
    
    b1 = 2*np.pi/det_a*np.array([a2[1], -a2[0]])
    b2 = 2*np.pi/det_a*np.array([-a1[1], a1[0]])
    F = np.dstack((0*mesh_x, 0*mesh_x))

    # Determine n1 and n2, numbers of k-points to sum in each direction:
    n1, n2 = 2, 2 # arbitrary starting point
    accuracy_x = 1
    accuracy_y = 1
    while accuracy_x > 1e-35 and n1 < 1000:
        n1 += 1
        accuracy_x = erfc(eta*n1*np.sqrt(b1[0]**2 + b1[1]**2))
    while accuracy_y > 1e-35 and n2 < 1000:
        n2 += 1
        accuracy_y = erfc(eta*n2*np.sqrt(b2[0]**2 + b2[1]**2))
    
    accuracy = max(accuracy_x, accuracy_y)
    print("*** Long-Range Force ***")
    print("n1 = {} and  n2 = {} k-points used in long-range force".format(n1, n2))
    print("{:.3e} relative accuracy is achieved in long-range force".format(accuracy))
    
    
    for i1 in range(-(n1 - 1), n1):
        for i2 in range(-(n2 - 1), n2):
            if not (i1 == 0 and i2 == 0):
                k = i1*b1 + i2*b2
                k_abs = np.sqrt(k[0]**2 + k[1]**2)
                term1 = np.sin(k[0]*mesh_x + k[1]*mesh_y)*erfc(k_abs*eta)/k_abs
                F[:,:,0] += k[0]*term1
                F[:,:,1] += k[1]*term1
    F *= 2*np.pi/det_a*charge**2
    return F        
    
def F_s_ri(ri, mesh_x, mesh_y, eta):
    """
    F_s_ri(ri, mesh_x, mesh_y, eta)
    
        Short-range part of force F created by a particle located at ri evaluated at the mesh.
        This part should be evaluated separately for every image.
        
        The result must be multiplied by (charge)^2 to get force.

        eta - Ewald threshold (short/long range).

        return: 3D array res, with F_x = res[:, :, 0], F_y = res[:, :, 1].
        F_x and F_y evaluated on the mesh.
    """
    x_ri = mesh_x - ri[0]
    y_ri = mesh_y - ri[1]
    r_ri = r_from_ri(ri, mesh_x, mesh_y)
    term1 = 1/(np.sqrt(np.pi)*eta)*np.exp(-r_ri**2/4./eta**2)/r_ri**2
    term2 = erfc(0.5*r_ri/eta)/r_ri**3
    return np.dstack(((term1 + term2)*x_ri, (term1 + term2)*y_ri))

def F_short(mesh_x, mesh_y, charge, eta):
    """
    F_short(mesh_x, mesh_y, charge, eta)
        
        Total short-range force from Ewald procedure
        
        eta - Ewald threshold (short/long range).

        return: 3D array res, with F_x = res[:, :, 0], F_y = res[:, :, 1].
        F_x and F_y evaluated on the mesh.
    """
    # Retrieve geometry:
    (a1, a2, lx, ly) = ai_from_mesh(mesh_x, mesh_y)
    
    # Determine which periodic images give significant contribution
    # These formula ensure exp( - r/eta) < 1e-10 for contributions from the most
    # distant significant images to any point in unit cell.
    # nx, ny - number of images in each direction (e.g. ny above and ny below)
    nx, ny = (1, 1) # arbitrary starting point
    accuracy_x = 1
    accuracy_y = 1
    while accuracy_x > 1e-25 and nx < 1000:
        nx += 1
        accuracy_x = max(erfc(0.5*(nx - 0.5)*lx/eta), np.exp(- 0.25*(nx - 0.5)**2*lx**2/eta**2))
    while accuracy_y > 1e-25 and ny < 1000:
        ny += 1
        accuracy_y = max(erfc(0.5*(ny - 0.5)*ly/eta), np.exp(- 0.25*(ny - 0.5)**2*ly**2/eta**2))
    
    accuracy = max(accuracy_x, accuracy_y)
    print("*** Short-Range Force ***")
    print("{} x {} layers of images used for short-range force".format(nx, ny))
    print("{:.3e} relative accuracy is achieved in short-range force".format(accuracy))
    
    # init F:
    F = np.dstack((0*mesh_x, 0*mesh_x))
    
    # sum over all significant images:
    for i in range(-(nx - 1), nx + 1):
        for j in range(-(ny - 1), ny + 1):
            F += F_s_ri(i*a1 + j*a2, mesh_x, mesh_y, eta)
    F *= charge**2
    return F


def export_to_file(dir_path, mesh_x, mesh_y, V_tot, F_tot, filename = None):
    """
    export_to_file(dir_path, mesh_x, mesh_y, V_tot, F_tot)
        Export total potential and force to a file
        Format:
        The file should have width*height (uncommented) lines and 5 columns:
        rxrx   ryry   VV   FxFx   FyFy 
        The order in which the points appear is fixed. It should start with point  (i=0,j=0),
        listing the first row of mesh points starting with  j=0  (ry=h2/2) and
        i  going from  0  to (width - 1),
        then  proceeding to the row with  j=1 , etc.
        filename is either provided through filename or generated based on current time.
    """
    height, width = mesh_x.shape
    
    if dir_path[0] != '/':
        raise ValueError("dir_path must be absolute, starting with '/'")
    # Add '/' at the end if absent:
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'
        
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
    if filename == None:
        timestamp = datetime.datetime.strftime(datetime.datetime.now(), format="%Y%m%d-%H%M%S")
        f_name = timestamp + '_table.dat'
    else:
        f_name = filename
    f = open(dir_path + f_name, 'w')

    f.write('# rx ry V Fx Fy\n')

    for j in range(height):
        for i in range(width):
            line = '{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(mesh_x[j,i], mesh_y[j,i], V_tot[j,i],\
                                                                 F_tot[j, i, 0], F_tot[j, i, 1])
            f.write(line)
    f.close()
    print('{} lines written to {}'.format((i + 1)*(j + 1), dir_path + f_name))
    return f_name


    
def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    #ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])