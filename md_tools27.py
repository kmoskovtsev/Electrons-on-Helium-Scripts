from __future__ import division
from matplotlib.animation import FuncAnimation
import gsd.fl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from hoomd.data import boxdim
from scipy.spatial import Delaunay

def __init__():
    """
    This package is supposed to operate on data calculated in a rectangular box.
    The package assumes this and doesn't check.
    """
    pass

def correct_jumps(pos, pos_m1_cntn, pos_m1_inbox, lx, ly):
    """    
    Correct jumps over the periodic boundary. Jumps by lx or ly make diffusion calculation
    nonsense.
    pos - positions in current frame
    pos_m1_cntc - "continuous" position in the previous frame: corrected for jumps from previous step
    pos_m1_inbox - "raw" position in the previous frame: inside simulation box
    lx, ly - unit cell length and width
    
    return: corrected positions in current frame
    """
    
    dr = pos - pos_m1_inbox
    # x-direction jumps:
    ind = np.where(dr[:,0] > lx/2)
    dr[ind, 0] = dr[ind, 0] - lx
    ind = np.where(dr[:,0] < - lx/2)
    dr[ind, 0] = dr[ind, 0] + lx
    
    #y-jumps:
    ind = np.where(dr[:,1] > ly/2)
    dr[ind, 1] = dr[ind, 1] - ly
    ind = np.where(dr[:,1] < - ly/2)
    dr[ind, 1] = dr[ind, 1] + ly

    return pos_m1_cntn + dr
    


def get_file_list(folder_path):
    """ 
        Read filenames and corresponding Gamma and T_values in a folder
    """
    f_list, gamma_list, T_list, dt_list = [], [], [], []
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    try:
        fl = open(folder_path + 'list.txt', 'r')
    except:
        print('Could not find list.txt in {}'.format(folder_path))
        raise FileNotFoundError
    for line in fl.readlines():
        if line[0] != '#':
            words = line.split('\t')
            f_list.append(words[0])
            gamma_list.append(float(words[1]))
            T_list.append(float(words[2]))
            dt_list.append(float(words[3]))
    fl.close()
    return f_list, gamma_list, T_list, dt_list

    
    
def diffusion_from_gsd(folder_path, center_fixed = True, useframes = -1):
    """
    Calculate diffusion coefficients vs Gamma from gsd files located in folder_path.
    The folder must have list.txt that has the following structure:
    
    # file    Gamma    T    dt
    00000.gsd	145.00000000	1.09746597	0.00100000
    00001.gsd	144.28571429	1.10289897	0.00100000
    
    Lines starting with # are ignored. The columns are tab-separated
    Each gsd file should contain a long enough trajectory in thermalized state.
    
    Diffusion constant D is calculated from 4Dt = <(r(t) - r(0))^2>.
    The average is calculated over all particles and over different time origins.
    Time origins go from 0 to n_frames/2, and t goes from 0 to n_frames/2. This way,
    the data are always within the trajectory.
    
    center_fixed = True: eliminate oveall motion of center of mass
    
    return D_x, D_y, gamma_list, T_list
    D_x, D_y diffusion for x- and y-coordinates;
    all arrays are ordered for right correspondence, e.g. D_x[i] <-> gamma_list[i]
    """
    f_list, gamma_list, T_list, dt_list = get_file_list(folder_path)
    
    D_x_list = np.zeros(len(f_list))
    D_y_list = np.zeros(len(f_list))
    
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'

    for i, f_name in enumerate(f_list):
        print(f_name)
        with gsd.fl.GSDFile(folder_path + f_name, 'rb') as f:
            n_frames = f.nframes
            box = f.read_chunk(frame=0, name='configuration/box')
            half_frames = int(n_frames/2) - 1 #sligtly less than half to avoid out of bound i
            if useframes < 1 or useframes > half_frames:
                useframes = half_frames
            t_step0 = f.read_chunk(frame=0, name='configuration/step')
            t_step1 = f.read_chunk(frame=1, name='configuration/step')
            snap_period = t_step1[0] - t_step0[0]
            n_p = f.read_chunk(frame=0, name='particles/N')
            if i == 0: #create square-average displacement once
                x_sq_av = np.zeros((useframes, len(f_list)))
                y_sq_av = np.zeros((useframes, len(f_list)))
            for t_origin in range(n_frames - useframes - 1):
                pos_0 = f.read_chunk(frame=t_origin, name='particles/position')
                mean_pos_0 = np.mean(pos_0, axis = 0)
                pos = pos_0
                pos_raw = pos_0
                for j_frame in range(useframes):
                    pos_m1 = pos
                    pos_m1_raw = pos_raw
                    pos_raw = f.read_chunk(frame=j_frame + t_origin, name='particles/position') - pos_0
                    pos = correct_jumps(pos_raw, pos_m1, pos_m1_raw, box[0], box[1])
                    if center_fixed:
                        pos -= np.mean(pos, axis = 0) - mean_pos_0 #correct for center of mass movement
                    
                    x_sq_av[j_frame, i] += np.mean(pos[:,0]**2)
                    y_sq_av[j_frame, i] += np.mean(pos[:,1]**2)
            x_sq_av[:, i] /= (n_frames - useframes - 1)
            y_sq_av[:, i] /= (n_frames - useframes - 1)
            # OLS estimate for beta_x[0] + beta_x[1]*t = <|x_i(t) - x_i(0)|^2>
            a = np.ones((useframes, 2)) # matrix a = ones(half_frames) | (0; dt; 2dt; 3dt; ...)
            a[:,1] = snap_period*dt_list[i]*np.cumsum(np.ones(useframes), axis = 0) - dt_list[i]
            b_cutoff = int(useframes/10) #cutoff to get only linear part of x_sq_av, makes results a bit more clean
            beta_x = np.linalg.lstsq(a[b_cutoff:, :], x_sq_av[b_cutoff:,i], rcond=-1)
            beta_y = np.linalg.lstsq(a[b_cutoff:, :], y_sq_av[b_cutoff:,i], rcond=-1)
            D_x_list[i] = beta_x[0][1]/4
            D_y_list[i] = beta_y[0][1]/4
    return D_x_list, D_y_list, gamma_list, T_list

    
    
def gsd_trajectory(fpath, axis, periodic = False, center_fixed = True):
    """
    Return a trajectory data for all particles in gsd file located at fpath;
    axis = 0 for x coordinates, 1 for y coordinates;
    periodic = True to contain particles within box, False - allow particles to diffuse out of box;
    
    """
    if axis != 0 and axis != 1:
        raise ValueError('axis must be 0 or 1')
    
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = np.zeros((n_frames, n_p[0]))
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        mean_pos_0 = np.mean(pos_frame, axis = 0)
        for j_frame in range(n_frames):
            pos_m1 = pos_frame
            pos_frame = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            if not periodic:
                pos_frame = correct_jumps(pos_frame, pos_m1, box[0], box[1])
            if center_fixed:
                pos_frame -= np.mean(pos_frame, axis = 0) - mean_pos_0
            pos[j_frame, :] = pos_frame[:, axis]
    return pos

def find_neighbors(pos, box, rcut = 1.4):
    """ Find neighbors for each particle. The neighbors are determined to be within rcut distance from the
    particle.
    \param pos - N x 3 array of positions
    \param box - hoomd box object, simulation box
    \param rcut - cutoff radius for neighbors
    Return neighbor_list and neighbor_num.
    neighbor_list - N x 30 array of int, each row i containing indices of neighbors of i-th particle,
                    including the particle itself.
                    The remaining part of the row is filled with -1 (e.g. for 6 neighbors, remaining 23 sites are -1).
    neighbor_num - int array of size N containing numbers of neighbors for each particle. 
    """
    neighbor_list = np.zeros((pos.shape[0], 30), dtype=int) - 1
    neighbor_num = np.zeros(pos.shape[0], dtype = int)
    dist_list = np.zeros((pos.shape[0], 30), dtype=float) - 1

    for i, r in enumerate(pos):
        pos_ref = reshape_to_box(pos, r, box)
        box_ind = np.where((pos_ref[:,0] > -rcut)*(pos_ref[:,0] < rcut)*(pos_ref[:,1] > -rcut)*(pos_ref[:,1] < rcut))[0]
        box_pos = pos_ref[box_ind, :]
        dist = np.sqrt(np.sum(box_pos**2, axis = 1))
        box_neighbors = np.where(dist < rcut)[0]
        box_dist = dist[box_neighbors]
        neighbor_ind = box_ind[box_neighbors]
        neighbor_num[i] = len(neighbor_ind) - 1
        for j, ind in enumerate(neighbor_ind):
            neighbor_list[i, j] = ind
            dist_list[i, j] = box_dist[j]
    return neighbor_list, neighbor_num
    
def create_virtual_layer(pos, box):
    """
    Create a virtual layer around the simulation box of ~2 interelectron distances of particles that are periodic replicas from the
    simulation box.
    """
    N = pos.shape[0]
    a = 2*np.sqrt(box.Lx*box.Ly/N) # approximately 1-3 interelectron distances
    # Add full periodic replicas around the box
    shift_arr = [-1, 0, 1]
    full_virtual_pos = np.zeros((N*9, 3))
    counter = 0
    for i in shift_arr:
        for j in shift_arr:
            #i,j = 0,0 when counter=4
            pos_shifted = np.zeros(pos.shape)
            pos_shifted[:,0] = pos[:,0] + i*box.Lx
            pos_shifted[:,1] = pos[:,1] + j*box.Ly
            full_virtual_pos[counter*N:(counter + 1)*N, :] = pos_shifted
            counter += 1
    # Crop the extended virtual box to the original box plus a layer of width `a`
    box_ind = np.where((np.abs(full_virtual_pos[:,0]) < 0.5*box.Lx + a)*(np.abs(full_virtual_pos[:,1]) < 0.5*box.Ly + a))[0]
    crop_virtual_pos = full_virtual_pos[box_ind, :]
    
    return crop_virtual_pos, box_ind
                

    
def find_neighbors_delone(pos, box):
    """ Find neighbors for each particle with the Delone triangulation methos.
    \param pos - N x 3 array of positions
    \param box - hoomd box object, simulation box
    \param rcut - cutoff radius for neighbors
    Return neighbor_list and neighbor_num.
    neighbor_list - N x 30 array of int, each row i containing indices of neighbors of i-th particle,
                    including the particle itself.
                    The remaining part of the row is filled with -1 (e.g. for 6 neighbors, remaining 23 sites are -1).
    neighbor_num - int array of size N containing numbers of neighbors for each particle. 
    """
    N = pos.shape[0]
    neighbor_list = np.zeros((pos.shape[0], 30), dtype=int) - 1
    neighbor_num = np.zeros(pos.shape[0], dtype = int)
    dist_list = np.zeros((pos.shape[0], 30), dtype=float) - 1
    virtual_pos, virtual_ind = create_virtual_layer(pos, box)
    tri = Delaunay(virtual_pos[:, 0:2])
    (indices, indptr) = tri.vertex_neighbor_vertices
    for vi in range(virtual_pos.shape[0]):
        i = virtual_ind[vi] #index of a particle in the big virtual box with full repeated cells
        i_real = i%N
        # for real particles in the real box
        if i >= N*4 and i < N*5:
            # indices of neighbors in the vitrual box
            vi_neighb = indptr[indices[vi]:indices[vi+1]]
            # recover real neighbor indices in the original box
            i_neighb = virtual_ind[vi_neighb]%N
            n_neighb = len(i_neighb)
            neighbor_list[i_real, :n_neighb] = i_neighb
            neighbor_num[i_real] = n_neighb
    return neighbor_list, neighbor_num


    
def animate_gsd(fpath, savefile = None, periodic = False, center_fixed = True, interval = 100, figsize = (12, 12), rcut = 1.4,\
               neighb = False):
    """
    Create animation from a gsd file, where fpath is the path+filename to the gsd file.
    lx, ly - dimensions of the rectangular simulation box;
    savefile - filename for saving the animation. Not trying to save if the value is None;
    periodic - show with periodic boundary conditions if True, let particles migrate out of box if False;
    interval - time interval between frames in microseconds;
    
    return fig, ax
    """
    def init():
        scat.set_offsets(pos[0, :, :])
        
        time_text.set_text('')
        return scat, time_text

    def update(frame_number):
        scat.set_offsets(pos[frame_number, :, :])
        time_text.set_text('frame = {} of {}'.format(frame_number, n_frames))
        if neighb:
            five_ind = np.where(neighbor_num[frame_number, :] == 5)[0]
            seven_ind = np.where(neighbor_num[frame_number, :] == 7)[0]
            five_scat.set_offsets(pos[frame_number, five_ind, :])
            seven_scat.set_offsets(pos[frame_number, seven_ind, :])
        else:
            five_scat.set_offsets(empty_pos)
            seven_scat.set_offsets(empty_pos)
        return scat, time_text, five_scat, seven_scat
    
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = np.zeros((n_frames, n_p[0], 2))
        neighbor_num = np.zeros((n_frames, int(n_p)), dtype = int)
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        pos_frame_raw = pos_frame
        mean_pos_0 = np.mean(pos_frame, axis = 0)
        for j_frame in range(n_frames):
            pos_m1 = pos_frame
            pos_m1_raw = pos_frame_raw
            pos_frame_raw = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            if not periodic:
                pos_frame = correct_jumps(pos_frame_raw, pos_m1, pos_m1_raw, box[0], box[1])
            else:
                pos_frame = pos_frame_raw
            if center_fixed:
                pos_frame -= np.mean(pos_frame, axis = 0) - mean_pos_0
            pos[j_frame, :, :] = pos_frame[:, 0:2]
            if neighb:
                boxdim_box = boxdim(box[0], box[1], box[2])
                neighbor_list, neighbor_num[j_frame, :] = find_neighbors_delone(pos[j_frame, :,:], boxdim_box, rcut = rcut)
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-box[0], box[0]), ylim=(-box[1], box[1]))

    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1],
                      s = 3,
                      facecolors='blue')
    empty_pos = np.zeros(0)
    if neighb:
        five_ind_0 = np.where(neighbor_num[0, :] == 5)[0]
        seven_ind_0 = np.where(neighbor_num[0, :] == 7)[0]
        seven_scat = ax.scatter(pos[0, seven_ind_0, 0], pos[0, seven_ind_0, 1],
                          s = 5,
                          facecolors='green')
        five_scat = ax.scatter(pos[0, five_ind_0, 0], pos[0, five_ind_0, 1],
                          s = 5,
                          facecolors='red')
    else:
        seven_scat = ax.scatter(empty_pos, empty_pos)
        five_scat = ax.scatter(empty_pos, empty_pos)
    time_text = ax.text(0.02, 1.05, '', transform=ax.transAxes)

    animation = FuncAnimation(fig, update, interval=interval, frames=n_frames, blit=True)
    
    if not savefile == None:
        try:
            animation.save(savefile, fps=30)
        except Exception as ee: print(ee)
    return fig, ax, animation
    
    
def plot_DxDy(Dx, Dy, gamma_list, timestamp, text_list = [], text_pos = 'c', folder = ''):
    """Create two-panel plot for Dx, Dy vs Gamma
    text_list = list of strings that will be displayed on the plot
    text_pos = 'l', 'c', or 'r' for left, center, and right alignment of the text
    timestamp = string that will serve as a base for the filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 6))
    ax1.scatter(gamma_list, Dx)
    ax1.set_xlabel('$\Gamma$')
    ax1.set_ylabel('$D_x$')

    ax2.scatter(gamma_list, Dy)
    ax2.set_xlabel('$\Gamma$')
    ax2.set_ylabel('$D_y$')
    fig.tight_layout()
    fig.patch.set_alpha(1)
    
    #determine text position
    y_lim = ax1.get_ylim()
    x_lim = ax1.get_xlim()
    h = y_lim[1] - y_lim[0]
    w = x_lim[1] - x_lim[0]
    text_y = y_lim[1] - 0.1*h

    text_x = {
        'l': x_lim[0] + 0.1*w,
        'c': x_lim[0] + 0.4*w,
        'r': x_lim[0] + 0.7*w
    }.get(text_pos, x_lim[0] + 0.4*w)
    
    #print text
    if type(text_list) == list: 
        n_str = len(text_list)
        for i in range(n_str):
            ax1.text(text_x, text_y - 0.05*h*i, text_list[i])
    elif type(text_list) == str:
        ax1.text(text_x, text_y, text_list)
    else:
        raise TypeError('text_list must be a list of strings or a string')
    if folder != '':
        if folder[-1] != '/':
            folder = folder + '/'
    fig.savefig(folder + timestamp + '_diff.png')
    return fig, ax1, ax2
    
    
def plot_positions(system=None, pos=None, box=None, figsize = (7, 7), gridon = True, ax=None, fig=None, s=5):
    """ Show positions of all particles in the system,
    where system is the hoomd system, produced by hoomd.init.
    Show grid lines if gridon == True
    """
    if system != None:
        snapshot = system.take_snapshot(all=True)
        box = snapshot.box
        pos = snapshot.particles.position
    if ax==None or fig==None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.6*box.Lx, 0.6*box.Lx), ylim=(-0.6*box.Ly, 0.6*box.Ly))
    
    scat = ax.scatter(pos[:, 0], pos[:, 1],
                      s = s,
                      facecolors='blue')
    if gridon:
        ax.grid()
    return fig, ax
    
    
def reshape_to_box(pos, ref, box):
    """
    Center the simulation box arund ref, and return particles to within the simulation box.
    Return new positions.
    """
    pos_ref = pos - ref
    mask = np.fix(pos_ref[:, 0]/(0.5*box.Lx)) #-1 or 1 if to the left or right from the centered box 
    pos_ref[:, 0] -= mask*box.Lx
    
    mask = np.fix(pos_ref[:, 1]/(0.5*box.Ly))
    pos_ref[:, 1] -= mask*box.Ly
    return pos_ref

def is_odd(num):
    return num & 0x1

def pair_correlation(pos, box, n_bins = (100, 100)):
    """ Calculate pair correlation function for a given snapshot.
        \param positions: N x 3 array, particle positions
        \param box: simulation box object with fields Lx and Ly
        \param n_bins: tuple of size 2, numbers of bins in x and y direction for correlation function, both even
        Return: array of shape n_bins, pair correlation function normalized to (N - 1). Zero of coordinate is in the middle
        of the pixel grid (between pixels indexed (n_bins[i]/2 - 1) and (n_bins[i]/2)).
    """
    try:
        if is_odd(n_bins[0]) or is_odd(n_bins[1]):
            raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
    except:
        raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
        
    bins_x = np.linspace(-box.Lx/2, box.Lx/2, n_bins[0] + 1)
    bins_y = np.linspace(-box.Ly/2, box.Ly/2, n_bins[1] + 1)
    g = np.zeros(n_bins)
    for r in pos:
        pos_ref = reshape_to_box(pos, r, box)
        ind_x = np.digitize(pos_ref[:,0], bins_x) - 1 
        ind_y = np.digitize(pos_ref[:,1], bins_y) - 1
        if np.max(ind_x) >= n_bins[0]:
            ind_x[np.where(ind_x >= n_bins[0])] = n_bins[0] - 1 
        if np.max(ind_y) >= n_bins[1]:
            ind_y[np.where(ind_y >= n_bins[1])] = n_bins[1] - 1
            
        g[ind_x[:], ind_y[:]] += 1
    g[int(n_bins[0]/2), int(n_bins[1]/2)] = 0
    #Normalize so that the sum over all pixels is N - 1
    g /= pos.shape[0]
    #Normalize to the particle density so that integral of g is (N - 1)/n_s
    n_s = pos.shape[0]/box.Lx/box.Ly
    g *= n_bins[0]*n_bins[1]/box.Lx/box.Ly
    g /= n_s 
    return g

def pair_correlation_from_gsd(filename, n_bins = (100, 100), frames =(0, -1)):
    """ Calculate pair correlation function.
        Averaged over frames in the range from frames[0] to frames[1]. Negative frames are counted from the end of time.
        \param filename: name of the gsd file
        \param n_bins: tuple of size 2, numbers of bins in x and y direction for correlation function, both even
        Return: array of shape n_bins, pair correlation function normalized to (N - 1). Zero of coordinate is in the middle
        of the pixel grid (between pixels indexed (n_bins[i]/2 - 1) and (n_bins[i]/2)).
    """
    try:
        if is_odd(n_bins[0]) or is_odd(n_bins[1]):
            raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
    except:
        raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
        
    g = np.zeros(n_bins)
    with gsd.fl.GSDFile(filename, 'rb') as f_gsd:
        n_frames_total = f_gsd.nframes
        if frames[0] > n_frames_total or frames[1] > n_frames_total:
            raise ValueError('frames beyond n_frames_total')
        #translate negative indices into positive domain:
        abs_frames = (frames[0] -(frames[0]//n_frames_total)*n_frames_total, \
                     frames[1] -(frames[1]//n_frames_total)*n_frames_total)
        if abs_frames[0] > abs_frames[1]:
            raise ValueError('frames[0] must be to the left from frames[1]')
        all_frames = np.arange(0, n_frames_total, 1)
        selected_frames = all_frames[abs_frames[0]:abs_frames[1] + 1]
        n_frames = abs_frames[1] - abs_frames[0] + 1
        
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])
        pos = np.zeros((n_frames, n_p[0], 2))
        for j_frame in range(n_frames):
            pos_frame = f_gsd.read_chunk(frame=selected_frames[j_frame], name='particles/position')
            g += pair_correlation(pos_frame, box, n_bins = n_bins)
    g /= n_frames
    return g
    
    
def plot_pair_correlation(g, box, figsize = (8,8), cmap = "plasma", interp = 'none', tickfont=18,\
                          nbins=50, alpha=1, fig=None, ax=None, origin_marker=True, minmax=None, imshow=False):
    if ax==None or fig==None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.5*box.Lx, 0.5*box.Lx), ylim=(-0.5*box.Ly, 0.5*box.Ly))
    x = np.linspace(-box.Lx/2, box.Lx/2, g.shape[0])
    y = np.linspace(-box.Ly/2, box.Ly/2, g.shape[1])
    X, Y = np.meshgrid(x, y)
    if minmax == None:
        levels = MaxNLocator(nbins=nbins).tick_values(g.min(), g.max())
        ticks = [g.min(), g.max()]
    else:
        levels = MaxNLocator(nbins=nbins).tick_values(minmax[0], minmax[1])
        ticks = minmax
    cmapo = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmapo.N, clip=True)#
    
    #pc = ax.pcolor(X, Y, np.transpose(g), cmap = cmap,\
    #alpha=alpha, norm=norm, edgecolor='none', antialiaseds=False)
    if imshow:
        pc = ax.imshow(np.flipud(np.transpose(g)), extent=[X.min(), X.max(), Y.min(), Y.max()], cmap=cmap,\
                       alpha=alpha, interpolation=interp, norm=norm)
    else:
        pc = ax.pcolor(X, Y, np.transpose(g), cmap = cmapo,\
        alpha=alpha, norm=norm, edgecolor=None, antialiaseds=False, linewidth=0)
        pc.set_rasterized(True)
    if origin_marker:
        ax.scatter(0,0, c='r', marker = '+')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(pc, cax=cax, ax=ax, ticks=ticks)#, fraction=0.056, pad=0.04)
    cbar.ax.tick_params(labelsize=tickfont)
    cbar.ax.set_yticklabels(['{:.2f}'.format(ticks[0]), '{:.2f}'.format(ticks[1])])
    
    return fig, ax, cax
    
    
def psi_order(pos, box, nx=100, ny=100, rcut=1.4):
    """ 
    \param pos - N x 3 array of positions
    \param box - hoomd box object, simulation box
    \param rcut - cutoff radius for neighbors
    Return neighbor_list and neighbor_num.
    neighbor_list - N x 30 array of int, each row i containing indices of neighbors of i-th particle,
                    including the particle itself.
                    The remaining part of the row is filled with -1 (e.g. for 6 neighbors, remaining 23 sites are -1).
    neighbor_num - int array of size N containing numbers of neighbors for each particle. 
    """
    neighbor_list = np.zeros((pos.shape[0], 30), dtype=int) - 1
    neighbor_num = np.zeros(pos.shape[0], dtype = int)
    dist_list = np.zeros((pos.shape[0], 30), dtype=float) - 1
    psi = np.zeros((ny, nx), dtype=complex)
    hx = box.Lx/nx
    hy = box.Ly/ny
    X = np.linspace(-0.5*box.Lx, 0.5*box.Lx - hx, nx)
    Y = np.linspace(-0.5*box.Ly, 0.5*box.Ly - hy, ny)
    r = np.zeros(3)
    
    for i in range(nx):
        for j in range(ny):
            r[0] = X[i]
            r[1] = Y[j]
            pos_ref = reshape_to_box(pos, r, box)
            r_ind = np.where((pos_ref[:,0] > -rcut)*(pos_ref[:,0] < rcut)*(pos_ref[:,1] > -rcut)*(pos_ref[:,1] < rcut))[0]
            r_pos = pos_ref[r_ind, :]
            dist = np.sqrt(np.sum(r_pos**2, axis = 1))
            r_nearest_ind = np.argmin(dist)
            nearest_ind = r_ind[r_nearest_ind]
            nr = pos_ref[nearest_ind, :] # coordinate of nearest particle in r-centered box
            box_ind = np.where((pos_ref[:,0] > -rcut + nr[0])*(pos_ref[:,0] < rcut + nr[0])*\
                               (pos_ref[:,1] > -rcut + nr[1])*(pos_ref[:,1] < rcut + nr[1]))[0]
            box_pos = pos_ref[box_ind, :]
            dist = np.sqrt(np.sum((box_pos - nr)**2, axis = 1))
            box_neighbors = np.where(dist < rcut)[0]
            nb_rv = box_pos[box_neighbors, :] - nr # radius-vectors to nearest neighbors                   
            nb_dist = np.sqrt(np.sum(nb_rv**2, axis = 1))
            nb_rv = nb_rv[np.where(nb_dist != 0)]
            nb_dist = nb_dist[np.where(nb_dist != 0)]
            cos_theta = nb_rv[:,0]/nb_dist
            theta = np.arccos(cos_theta) # sign of the angle still uncertain
            neg_y_ind = np.where(nb_rv[:,1] < 0)[0]
            theta[neg_y_ind] = 2*np.pi - theta[neg_y_ind] #resolve uncertainty from arccos
            psi[j,i] = np.mean(np.exp(1j*6*theta))
    return psi
    
    
def psi_order_delone(pos, box, nx=100, ny=100, rcut=1.4):
    """ 
    \param pos - N x 3 array of positions
    \param box - hoomd box object, simulation box
    \param rcut - cutoff radius for neighbors
    Return neighbor_list and neighbor_num.
    neighbor_list - N x 30 array of int, each row i containing indices of neighbors of i-th particle,
                    including the particle itself.
                    The remaining part of the row is filled with -1 (e.g. for 6 neighbors, remaining 23 sites are -1).
    neighbor_num - int array of size N containing numbers of neighbors for each particle. 
    """
    #neighbor_list = np.zeros((pos.shape[0], 30), dtype=int) - 1
    #neighbor_num = np.zeros(pos.shape[0], dtype = int)
    dist_list = np.zeros((pos.shape[0], 30), dtype=float) - 1
    psi = np.zeros((ny, nx), dtype=complex)
    hx = box.Lx/nx
    hy = box.Ly/ny
    X = np.linspace(-0.5*box.Lx, 0.5*box.Lx - hx, nx)
    Y = np.linspace(-0.5*box.Ly, 0.5*box.Ly - hy, ny)
    r = np.zeros(3)
    n_list, n_num = find_neighbors_delone(pos, box)
    
    for i in range(nx):
        for j in range(ny):
            r[0] = X[i]
            r[1] = Y[j]
            pos_ref = reshape_to_box(pos, r, box)
            r_ind = np.where((pos_ref[:,0] > -rcut)*(pos_ref[:,0] < rcut)*(pos_ref[:,1] > -rcut)*(pos_ref[:,1] < rcut))[0]
            r_pos = pos_ref[r_ind, :]
            dist = np.sqrt(np.sum(r_pos**2, axis = 1))
            r_nearest_ind = np.argmin(dist)
            nearest_ind = r_ind[r_nearest_ind]
            nr = pos_ref[nearest_ind, :] # coordinate of nearest particle in r-centered box
            #box_ind = np.where((pos_ref[:,0] > -rcut + nr[0])*(pos_ref[:,0] < rcut + nr[0])*\
            #                   (pos_ref[:,1] > -rcut + nr[1])*(pos_ref[:,1] < rcut + nr[1]))[0]
            #box_pos = pos_ref[box_ind, :]
            #dist = np.sqrt(np.sum((box_pos - nr)**2, axis = 1))
            #box_neighbors = np.where(dist < rcut)[0]
            neighbors_ind = n_list[nearest_ind, :n_num[nearest_ind]]
            nb_rv = pos_ref[neighbors_ind, :] - nr # radius-vectors to nearest neighbors                   
            nb_dist = np.sqrt(np.sum(nb_rv**2, axis = 1))
            nb_rv = nb_rv[np.where(nb_dist != 0)]
            nb_dist = nb_dist[np.where(nb_dist != 0)]
            cos_theta = nb_rv[:,0]/nb_dist
            theta = np.arccos(cos_theta) # sign of the angle still uncertain
            neg_y_ind = np.where(nb_rv[:,1] < 0)[0]
            theta[neg_y_ind] = 2*np.pi - theta[neg_y_ind] #resolve uncertainty from arccos
            psi[j,i] = np.mean(np.exp(1j*6*theta))
    return psi

def psi6_order_from_gsd(fpath, frame=0):
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames_total = f_gsd.nframes
        if frame > n_frames_total:
            raise ValueError('frames beyond n_frames_total')
        #translate negative indices into positive domain:
        abs_frame = frame -(frame//n_frames_total)*n_frames_total
        pos = f_gsd.read_chunk(frame=abs_frame, name='particles/position')
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])
    psi = psi_order_delone(pos, box, nx=100, ny=100, rcut=1.3)
    return psi
    
def compute_psi6_correlation_from_gsd(fpath, Nframes=1, frame_step=1, nxny=(100,100)):
    nx, ny = nxny
    cf_psi = np.zeros((ny, nx), dtype=complex)
    for frame in range(0, Nframes, frame_step):    
        with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
            n_frames_total = f_gsd.nframes
            if frame > n_frames_total:
                print('frame {} beyond n_frames_total = {}'.format(frame, n_frames_total))
            else:
                #translate negative indices into positive domain:
                abs_frame = frame -(frame//n_frames_total)*n_frames_total
                pos = f_gsd.read_chunk(frame=abs_frame, name='particles/position')
                box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
                box = boxdim(*box_array[0:3])
                psi = psi_order(pos, box, nx=nx, ny=ny, rcut=1.3)
                cf_psi += correlation_function(psi)
    cf_psi /= min(Nframes, n_frames_total)
    return cf_psi
    
def plot_delone_triangulation(fpath, frame=0, fig=None, ax=None):
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames_total = f_gsd.nframes
        if frame > n_frames_total:
            raise ValueError('frames beyond n_frames_total')
        #translate negative indices into positive domain:
        abs_frame = frame -(frame//n_frames_total)*n_frames_total
        pos = f_gsd.read_chunk(frame=abs_frame, name='particles/position')
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])

    if ax==None or fig==None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.6*box.Lx, 0.6*box.Lx), ylim=(-0.6*box.Ly, 0.6*box.Ly))
    xlim = (-0.5*box.Lx, 0.5*box.Lx)
    ylim = (-0.5*box.Ly, 0.5*box.Ly)

    virtual_pos, virtual_ind = create_virtual_layer(pos, box)
    tri = Delaunay(virtual_pos[:, 0:2])
    ax.triplot(virtual_pos[:,0], virtual_pos[:,1], tri.simplices.copy(), color='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax
    
    
def correlation_function(f):
    cf = 0*f
    nynx = cf.shape
    #print('================================================================')
    for j in range(-nynx[0]//2, nynx[0]//2):
        for i in range(-nynx[1]//2, nynx[1]//2):
            #print('i={}, j={}, coor_i={}, coor_j={}'.format(i,j, nynx[1]//2 + i, nynx[0]//2 + j))
            f1 = np.roll(f, -j, axis=0)
            f2 = np.roll(f1, -i, axis=1)
            cf[nynx[0]//2 + j,nynx[1]//2 + i] = np.mean(f*np.conj(f2))
            #print('cf={}'.format(cf[nynx[0]//2 + j,nynx[1]//2 + i]))
    return cf
    
def Eee_from_gsd(fpath, table_path, width, height, step = 1):
    """
    Return interaction energy along a trajectory in gsd file located at fpath;
    table - interaction pair force object
    """
    import hoomd
    import hoomd.md
    hoomd.context.initialize('--mode=cpu')
    #hoomd = imp.reload(hoomd)
    system = hoomd.init.read_gsd(fpath, frame = 0)
    dt = 0.001
    all = hoomd.group.all();
    hoomd.md.integrate.mode_standard(dt=dt);
    langevin = hoomd.md.integrate.langevin(group=all, kT=0.1, seed=987);
    snapshot = system.take_snapshot(all=True)
    print(fpath)
    Eee = []
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        table = hoomd.md.pair.table2D(width, height, 0.5*box[0], 0.5*box[1])
        table.set_from_file(table_path)
        pos = np.zeros((n_frames, n_p[0]))
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        for j_frame in range(0, n_frames, step):
            pos_frame = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            
            snapshot.particles.position[:] = pos_frame[:]
            system.restore_snapshot(snapshot)
            hoomd.run(1, quiet=True)
            Eee.append(table.get_energy(hoomd.group.all()))
    return np.array(Eee)

def Eee_Ep_from_gsd(fpath, table_path, width, height, p=None, A=None, phi=None, step = 1):
    """
    Return interaction energy and external periodic potential energy along a trajectory in gsd file located at fpath;
    table - interaction pair force object
    """
    import hoomd
    import hoomd.md
    hoomd.context.initialize('--mode=cpu')
    #hoomd = imp.reload(hoomd)
    system = hoomd.init.read_gsd(fpath, frame = 0)
    if phi==None:
        phi=np.pi
    if p!=None and A!=None:
        periodic = hoomd.md.external.periodic_cos()
        periodic.force_coeff.set('A', A=A, i=0, p=p, phi=phi)
    dt = 0.001
    all = hoomd.group.all();
    hoomd.md.integrate.mode_standard(dt=dt);
    langevin = hoomd.md.integrate.langevin(group=all, kT=0.01, seed=987);
    snapshot = system.take_snapshot(all=True)
    print(fpath)
    Eee = []
    Ep = []
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        table = hoomd.md.pair.table2D(width, height, 0.5*box[0], 0.5*box[1])
        table.set_from_file(table_path)
        pos = np.zeros((n_frames, n_p[0]))
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        for j_frame in range(0, n_frames, step):
            pos_frame = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            snapshot.particles.position[:] = pos_frame[:]
            system.restore_snapshot(snapshot)
            hoomd.run(1, quiet=True)
            Eee.append(table.get_energy(hoomd.group.all()))
            Ep.append(periodic.get_energy(hoomd.group.all()))
    return np.array(Eee), np.array(Ep)
    
    
    
def sweep_density(pos, box, window, pts, axis=0):
    """ Density of particles along a chosen axis, computed using sweeping window average.
    \param pos - Nx3 array of particle coordinates
    \param box - hoomd box object with fields Lx and Ly
    \param window - averaging window width
    \param pts - number of points to compute density at
    \param axis - axis (0 or 1) for computing the density along
    """
    if axis == 0:
        L = box.Lx
        h = box.Ly
    elif axis == 1:
        L = box.Ly
        h = box.Lx
    else:
        raise ValueError('axis must be 0 or 1')
    X = np.linspace(-L/2, L/2, pts)
    Y = np.zeros(pts)
    for i, x in enumerate(X):
        Y[i] = len(np.where((pos[:,axis] > x - window/2)*((pos[:,axis] < x + window/2)))[0])
    Y /= (window*h)
    return X, Y
    
def rotate_positions(pos, box, alpha):
    R = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0],\
                 [0, 0, 1]])
    rot_pos = np.transpose(np.dot(R, np.transpose(pos)))
    rot_pos = reshape_to_box(rot_pos, np.zeros(3), box)
    return rot_pos
