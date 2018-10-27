import os, sys, time
import numpy as np
import pandas as pd
import h5py
import sklearn
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KDTree
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from mypmesh import pm as mesh


def select_particles(_pos, _centre, _width, _regiontype='box'):
    _pos = _pos - _centre
    if _regiontype == 'box':
        _indx = np.logical_and(np.abs(_pos[:, 0]) < 0.5*_width,
                               np.abs(_pos[:, 1]) < 0.5*_width,
                               np.abs(_pos[:, 2]) < 0.5*_width)
        _indx = np.where(_indx)[0]
    elif _regiontype == 'sphere':
        _dist = np.sqrt(_pos[:, 0]**2 +
                        _pos[:, 1]**2 +
                        _pos[:, 2]**2)
        _indx = np.where(_dist <= 0.5*_width)[0]
    return _indx



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


def projected__density_gauss_adaptive(pos, mass, centre, fov, ncells,
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
    print('%d particles in FOV' % len(pos))

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


def projected_density_pmesh(pos,mass,centre,fov,bins,window='tsc',projection_axis=2):
    pm = mesh.ParticleMesh(Nmesh = [bins,bins], BoxSize=fov)
    pos = pos-centre + np.ones(3)*0.5*fov
    axes = [0,1,2]
    axes.remove(projection_axis)
    mass2D =  pm.paint(pos[:,axes],mass=mass,resampler=window)
    dx = float(fov)/bins
    Sigma = mass2D / (dx*dx)
    xedges = np.linspace(-0.5*fov,0.5*fov,bins+1)
    yedges = np.linspace(-0.5*fov,0.5*fov,bins+1)
    xs = 0.5*(xedges[1:]+xedges[:-1])
    ys = 0.5*(yedges[1:]+yedges[:-1])
    return Sigma, xs, ys


def projected_density_pmesh_adaptive(pos, mass, centre, fov, ncells, hmax=10,
                                     window='tsc', projection_axis=0,
                                     smooth_fac=1.0, neighbour_no=32,
                                     particle_type=1):
    dx = float(fov)/ncells
    pos += np.ones(3)*0.5*fov
    X = np.copy(pos)
    pm = mesh.ParticleMesh(Nmesh=[ncells,ncells], BoxSize=fov)
    # Find 'smoothing lengths'
    if len(X) > neighbour_no:
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(X, k=neighbour_no, return_distance=True)
        h = smooth_fac*np.max(dist,axis=1)/dx
    else:
        h = np.ones(len(X))*hmax
    h[h>hmax]=hmax
    h[h<1] = 1

    axes = [0,1,2]
    axes.remove(projection_axis)
    mass2D =  pm.paint(pos[:, axes],hsml=smooth_fac*h,mass=mass,resampler=window)
    Sigma = mass2D / (dx*dx)
    #xedges = np.linspace(-0.5*fov, 0.5*fov, ncells+1)
    #yedges = np.linspace(-0.5*fov, 0.5*fov, ncells+1)
    #xs = 0.5*(xedges[1:]+xedges[:-1])
    #ys = 0.5*(yedges[1:]+yedges[:-1])
    return Sigma.value #, xs, ys


def projected_density_gauss(pos, centre, fov, ncells):
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
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*Lbox,
                           np.abs(pos[:, 1]) < 0.5*Lbox)
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

