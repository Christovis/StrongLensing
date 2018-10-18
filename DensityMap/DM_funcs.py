import os, sys
import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors.kde import KernelDensity
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

def histedges_equalN(x, nbin):
    npt = len(x)
    bin_edges = np.interp(np.linspace(0, npt, nbin+1), np.arange(npt), np.sort(x))
    bin_edges[0] = 0
    bin_edges[-1] *= 1.1
    return bin_edges


def cluster_subhalos(id_in, vrms_in, x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        if b == 0:
            id_out = id_in[binds]
            vrms_out = vrms_in[binds]
            x_out = x_in[binds]
            y_out = y_in[binds]
            z_out = z_in[binds]
            split_size_1d[b] = int(len(binds[0]))
        else:
            id_out = np.hstack((id_out, id_in[binds]))
            vrms_out = np.hstack((vrms_out, vrms_in[binds]))
            x_out = np.hstack((x_out, x_in[binds]))
            y_out = np.hstack((y_out, y_in[binds]))
            z_out = np.hstack((z_out, z_in[binds]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    return id_out, vrms_out, x_out, y_out, z_out, split_size_1d, split_disp_1d


def cluster_particles(mass_in, x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        if b == 0:
            mass_out = mass_in[binds]
            x_out = x_in[binds]
            y_out = y_in[binds]
            z_out = z_in[binds]
            split_size_1d[b] = int(len(binds[0]))
        else:
            mass_out = np.hstack((mass_out, mass_in[binds]))
            x_out = np.hstack((x_out, x_in[binds]))
            y_out = np.hstack((y_out, y_in[binds]))
            z_out = np.hstack((z_out, z_in[binds]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    return mass_out, x_out, y_out, z_out, split_size_1d, split_disp_1d


def adaptively_smoothed_maps(pos, h, mass, Lbox, centre, ncells, smooth_fac):
    """
    Gaussien smoothing kernel for 'neighbour_no' nearest neighbours
    Input:
        h: distance to furthest particle for each particle
    """
    #X = pos - centre
    print('adapt. smooth; pos range %f - %f' % (np.min(pos[:, 0]), np.max(pos[:, 0])))
    hbins = int(np.log2(h.max()/h.min()))+2
    hbin_edges = 0.8*h.min()*2**np.arange(hbins)
    hbin_mids = np.sqrt(hbin_edges[1:]*hbin_edges[:-1])
    hmask = np.digitize(h, hbin_edges)-1  # returns bin for each h
    sigmaS = np.zeros((len(hbin_mids), ncells, ncells))
    if comm_rank == 0:
        print('::: Star adaptively smoothed for-loop %d, %d' % (len(hbin_mids), hbins))
    for i in np.arange(len(hbin_mids)):
        if comm_rank == 0:
            print('foor loop round %d' % i)
        maskpos = pos[hmask==i]  #X
        maskm = mass[hmask==i]
        print('masks', maskpos, maskm)
        maskSigma, xedges, yedges = np.histogram2d(maskpos[:, 0], maskpos[:, 1],
                                                   bins=[ncells, ncells],
                                                   weights=maskm)
        pixelsmooth = smooth_fac*hbin_mids[i]/(xedges[1]-xedges[0])
        sigmaS[i] = gaussian_filter(maskSigma, pixelsmooth, truncate=3)
    return np.sum(sigmaS, axis=0), xedges, yedges


def projected_surface_density_smooth_old(pos, mass, centre, fov, ncells,
                                     smooth_fac, neighbour_no, ptype):
    """
    Input:
        pos: particle positions
        mass: particle masses
        centre: centre of sub-&halo
        fov: field-of-view
        ncells: number of grid cells
    """
    pos = pos - centre
    
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*fov,
                           np.abs(pos[:, 1]) < 0.5*fov)
    pos = pos[_indx, :]
    mass = mass[_indx]
    
    if (ptype == 'dm' and len(mass) <= 32):
        return np.zeros((ncells, ncells))
    elif (ptype == 'gas' and len(mass) <= 16):
        return np.zeros((ncells, ncells))
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return np.zeros((ncells, ncells))
    else:
        # Find 'smoothing lengths'
        if comm_rank == 0:
            print('\t number of particles in FOV: %d \n' % len(mass))
            startins = time.time()
        kdt = KDTree(pos, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(pos, k=neighbour_no, return_distance=True)
        if comm_rank == 0:
            print(':: KDTree took %f' % (time.time() - startins))
            startins = time.time()
        h = np.max(dist, axis=1)  # furthest particle for each particle
        centre = np.array([0, 0, 0])
        mass_in_cells, xedges, yedges = adaptively_smoothed_maps(pos, h,
                                                                 mass,
                                                                 fov,
                                                                 centre,
                                                                 ncells,
                                                                 smooth_fac)
        if comm_rank == 0:
            print(':: adapt. smooth took %f' % (time.time() - startins))
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
        return sigma


def projected_surface_density_smooth(pos, centre, fov, ncells):
    """
    Input:
        pos: particle positions
        mass: particle masses
        centre: centre of sub-&halo
        fov: field-of-view
        ncells: number of grid cells
    """
    pos = pos - centre
    
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.6*fov,
                           np.abs(pos[:, 1]) < 0.6*fov)
    pos = pos[_indx, :]
    n = 1024*1024
    h = (4*np.std(pos[:, :2])**5/(3*n))**(1/5)
    #TODO: plot this falty situation
    kde_skl = KernelDensity(bandwidth=h,
                            kernel='gaussian',
                            algorithm='ball_tree')
    
    xx, yy = np.mgrid[min(pos[:, 0]):max(pos[:, 0]):complex(ncells), 
                      min(pos[:, 1]):max(pos[:, 1]):complex(ncells)]

    xy_sample = np.vstack([xx.ravel(), yy.ravel()]).T

    kde_skl.fit(pos[:, :2])
    sigma = np.exp(kde_skl.score_samples(xy_sample))
    sigma = sigma.reshape(xx.shape)
    return sigma, h


def projected_surface_density(pos, mass, centre, Lbox, ncells):
    """
    """
    # centre particles around subhalo
    pos = pos - centre
    # 2D histogram map
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.6*Lbox,
                           np.abs(pos[:, 1]) < 0.6*Lbox)
    pos = pos[_indx, :]
    mass = mass[_indx]
    if len(mass) == 0:
        return np.zeros((ncells, ncells))
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return np.zeros((ncells, ncells))
    else:
        mass_in_cells, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                                       bins=[ncells, ncells],
                                                       weights=mass)
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
        return sigma

