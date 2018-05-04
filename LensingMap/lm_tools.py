#!/usr/bin/env python

from __future__ import division
import re
import os.path
import numpy as np
from pylab import *
import scipy.stats as stats
from scipy import ndimage 
import scipy.interpolate as interpolate
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15
import matplotlib.pyplot as plt
# Requires: Python (2.7.13), NumPy (>= 1.8.2), SciPy (>= 0.13.3)
import sklearn
from sklearn.neighbors import KDTree
from scipy.ndimage.filters import gaussian_filter
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
#from cython.view cimport array as cvarray
######################################################## 


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def source_selection(src_id, src_z, halo_id):
    """
    Find redshift of sources which are likely to be multiple imaged
    Input:
        src_id[np.array(int)] - LightCone-IDs of sources
        src_z[np.array(float)] - redshift of sources
        halo_id[int] - ID of subhalo acting as lens
    Output:
        zs[int] - redshift of source
    """
    src_indx = np.where(src_id == halo_id)
    indx = np.argmax(src_z[src_indx[0]])
    zs = src_z[src_indx[0][indx]]
    return zs


def sigma_crit(zLens, zSource):
    Ds = Planck15.angular_diameter_distance(zSource)
    Dl = Planck15.angular_diameter_distance(zLens)
    Dls = Planck15.angular_diameter_distance_z1z2(zLens, zSource)
    sig_crit = (const.c**2/(4*np.pi*const.G))*Ds/(Dl*Dls)
    return sig_crit


def adaptively_smoothed_maps(pos, h, mass=None, fov=2, bins=512,
		centre=np.array([0, 0, 0]), smooth_fac=0.5):
	"""
	Gaussien smoothing kernel for 'neighbour_no' nearest neighbours
	
	Input:
		pos: particle positions
		h: furthest particle of 'neighbour_no' of particles from each particle
		fov: field of view
		centre: lense centre
	"""
	Lbox = fov
	Ncells = bins
	X = pos - centre
    
	if mass is None:
		m = np.ones(len(X))
	else:
		m = mass

	hbins = int(np.log2(h.max()/h.min()))+2
	hbin_edges = 0.8*h.min()*2**np.arange(hbins)
	hbin_mids = np.sqrt(hbin_edges[1:]*hbin_edges[:-1])
	hmask = np.digitize(h, hbin_edges)-1  # returns bin for each h
	sigmaS = np.zeros((len(hbin_mids), Ncells, Ncells))
	for i in np.arange(len(hbin_mids)):
		maskpos = X[hmask==i]
		maskm = m[hmask==i]
		maskSigma, xedges, yedges = np.histogram2d(maskpos[:, 0], maskpos[:, 1],
												   bins=[Ncells, Ncells],
												   range=[[-0.5*Lbox,0.5*Lbox],
														  [-0.5*Lbox,0.5*Lbox]],
												   weights=maskm)
		pixelsmooth = smooth_fac*hbin_mids[i]/(xedges[1]-xedges[0])
		sigmaS[i] = gaussian_filter(maskSigma,pixelsmooth,truncate=3)
	return np.sum(sigmaS, axis=0), xedges, yedges


def projected_surface_density(pos, mass, centre, fov=2, bins=512, smooth=True,
                              smooth_fac=None, neighbour_no=None):
    """
    Fit ellipsoid to 3D distribution of points and return eigenvectors
    and eigenvalues of the result.
    The same as 'projected_surface_density_adaptive', but assumes that
    density map is not created and particle data loaded, and you can
    choose between smoothed and closest particle density map.
    Should be able to combine all three project_surface_density func's

    Args:
        pos: particle position (physical)
        mass: particle mass
        centre: lense centre [x, y, z]
        fov:
        bins:
        smooth:
        smooth_fac:
        neighbour_no:

    Returns:
        Sigma: surface density [Msun/Mpc^2]
        x: 
        y:
    """
    Lbox = fov
    Ncells = bins

    ################ Shift particle coordinates to centre ################ 
    pos = pos - centre
    
    if smooth is True:
        if smooth_fac is None:
            print "Need to supply a value for smooth_fac when smooth=True"
            return
        if neighbour_no is None:
            print "Need to supply a value for neighbour_no when smooth=True"
            return
        ######################## DM 2D smoothed map #######################
        # Find all particles within 1.4xfov
        # l.o.s. is along x-axes not z-axes !!!!!!!!!!!!!!!!!!!!!1
        X = pos[np.logical_and(np.abs(pos[:,0]) < 0.7*Lbox,
                               np.abs(pos[:,1]) < 0.7*Lbox)]
        M = mass[np.logical_and(np.abs(pos[:,0]) < 0.7*Lbox,
                                np.abs(pos[:,1]) < 0.7*Lbox)]
        print('check included particles:', len(X))
        # Find 'smoothing lengths'
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(X, k=neighbour_no, return_distance=True)
        h = np.max(dist, axis=1)  # furthest particle for each particle
        mass_in_cells, xedges, yedges = adaptively_smoothed_maps(X, h, mass=M,
                                                                 fov=Lbox,
                                                                 bins=Ncells,
                                                                 smooth_fac=smooth_fac)
    else:
        ####################### DM 2D histogram map #######################
        mass_in_cells, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                                       bins=[Ncells, Ncells],
                                                       range=[[-0.5*Lbox, 0.5*Lbox],
                                                              [-0.5*Lbox, 0.5*Lbox]],
                                                       weights=mass)
    ###################### Projected surface density ######################
    dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]
    Sigma = mass_in_cells/(dx*dy)
    xs = 0.5*(xedges[1:]+xedges[:-1])
    ys = 0.5*(yedges[1:]+yedges[:-1])
    return Sigma, xs, ys


def alpha_calc(pos, kappa, x):
    dx, dy = x[1,0,0]-x[0,0,0], x[0,1,1]-x[0,0,1]
    alpha = np.sum(kappa[:, :, None]*(pos-x)/np.sum((pos-x)**2,axis=-1)[:, :, None], 
            axis=(0,1))*dx*dy/np.pi
    return alpha


def alpha_map(kappa, x, xi0, halo, version, identifier, save=True,
              overwrite=False, verbose=False):
    mappath = 'data/strong_lensing/halo_'+str(halo).zfill(2)+'/'+version+'/'
    mapname = identifier+'_alpha'
    if os.path.isfile(mappath+mapname+'.npz') and overwrite==False:
        data = np.load(mappath+mapname+'.npz')
        alpha = data['alpha']
        xtest = data['xtest']
        xi0_load = data['xi0']  #[Mpc]
        alpha *= xi0_load / xi0.to('Mpc').value
        xtest *= xi0_load / xi0.to('Mpc').value
    else:
        print('Map does not exist. Calculating map...')
        dx, dy = x[1,0,0]-x[0,0,0], x[0,1,1]-x[0,0,1]
        xtest = x + np.array([dx*0.5,dy*0.5])
        alpha = np.zeros(xtest.shape)
        for i in np.arange(xtest.shape[0]):
            if verbose:
                print('Alpha map, line ' + str(i))
            for j in np.arange(xtest.shape[1]):
                alpha[i,j] = alpha_calc(xtest[i, j], kappa, x)
        if save:
            ensure_dir(mappath)
            np.savez(mappath+mapname, alpha=alpha,
                     xtest=xtest, xi0=xi0.to('Mpc').value)
    return alpha, xtest


def alpha_map_fourier(kappa, x, padFac):
    xlen, ylen = kappa.shape
    xpad, ypad = xlen*padFac, ylen*padFac
    Lx = x[-1,0,0]-x[0,0,0]
    Ly = x[0,-1,1]-x[0,0,1]
    # round to power of 2 to speed up FFT
    xpad = np.int(2**(np.ceil(np.log2(xpad))))
    ypad = np.int(2**(np.ceil(np.log2(ypad))))
    kappa_ft = np.fft.fft2(kappa,s=[xpad,ypad])
    Lxpad, Lypad = Lx * xpad/xlen, Ly * ypad/ylen
    # make a k-space grid
    kxgrid, kygrid = np.meshgrid(np.fft.fftfreq(xpad),np.fft.fftfreq(ypad),indexing='ij')
    kxgrid *= 2*np.pi*xpad/Lxpad
    kygrid *= 2*np.pi*ypad/Lypad
    alphaX_kfac = 2j * kxgrid / (kxgrid**2 + kygrid**2)
    alphaY_kfac = 2j * kygrid / (kxgrid**2 + kygrid**2)
    # [0,0] component mucked up by dividing by k^2
    alphaX_kfac[0,0], alphaY_kfac[0,0] = 0,0
    alphaX_ft = alphaX_kfac * kappa_ft
    alphaY_ft = alphaY_kfac * kappa_ft
    alphaX = np.fft.ifft2(alphaX_ft)[:xlen,:ylen]
    alphaY = np.fft.ifft2(alphaY_ft)[:xlen,:ylen]
    #gamma1 = np.real(gamma1)
    #gamma2 = np.real(gamma2)
    alpha = np.zeros(x.shape)
    alpha[:,:,0], alpha[:,:,1] = alphaX, alphaY
    return -alpha # worry aboutminus sign? Seems to make it work :-)


def alpha_from_kappa(kappa, xphys, yphys, xi0, Nrays, Lrays):
    """
    Calculate deflection angle

    Args:
        kappa: 2D kappa map
        xphys & yphys: edges of surface density grid
        xi0: defines number of rays going through Lrays
        Nrays: Number of, to calculate alpha from kappa
        Lrays: Lengt of, to calculate alpha from kappa

    Returns:
        detA: 
    """
    # Dimensionless lens plane coordinates
    xs, ys = xphys/xi0, yphys/xi0
    # Array of 2D dimensionless coordinates
    xsgrid, ysgrid = np.meshgrid(xs, ys, indexing='ij')
    x = np.zeros((len(xs), len(ys), 2))
    x[:,:,0] = xsgrid
    x[:,:,1] = ysgrid

    alpha = alpha_map_fourier(kappa, x, padFac=4.0)
    xtest = x

    alphaxinterp = interpolate.RectBivariateSpline(xtest[:, 0, 0], xtest[0, :, 1],
                                                   alpha[:, :, 0])
    alphayinterp = interpolate.RectBivariateSpline(xtest[:, 0, 0], xtest[0, :, 1],
                                                   alpha[:, :, 1])

    xrays = np.linspace(-0.5*(Lrays/xi0), 0.5*(Lrays/xi0), Nrays)
    yrays = np.linspace(-0.5*(Lrays/xi0), 0.5*(Lrays/xi0), Nrays)
    xraysgrid, yraysgrid = np.meshgrid(xrays, yrays, indexing='ij')
    dxray = xrays[1]-xrays[0]  # distance between rays
    dyray = yrays[1]-yrays[0]

    alphax = alphaxinterp(xrays,yrays)
    alphay = alphayinterp(xrays,yrays)

    #Jacobian Matrix, page32
    A11 = 1 - np.gradient(alphax,dxray,axis=0)
    A12 = - np.gradient(alphax,dyray,axis=1)
    A21 = - np.gradient(alphay,dxray,axis=0)
    A22 = 1 - np.gradient(alphay,dyray,axis=1)

    detA = A11*A22 - A12*A21
    mag = 1/detA  #magnification, page 101

    ka = 1 - 0.5*(A11 + A22)  #kappa?
    ga1 = 0.5*(A22 - A11)     #gamma1?
    ga2 = -0.5*(A12 + A21)    #gamma2?
    ga = (ga1**2 + ga2**2)**0.5

    lambda_t = 1 - ka - ga  # tangential eigenvalue, page 115
    lambda_r = 1 - ka + ga  # radial eigenvalue, page 115

    # alphax, alphay, xrays, yrays are all dimesnionless
    return alphax, alphay, detA, xrays, yrays, lambda_t, lambda_r


def area(vs):
    """
    Use Green's theorem to compute the area enclosed by the given contour.
    """
    a = 0
    x0, y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dy = y1-y0
        a += x0*dy
        x0 = x1
        y0 = y1
    return a


def cal_lensing_signals():
    """
    Calculate Lensing Signals
    Input:
    kap: convergence map
    bzz: box size, arcsec
    ncc: number of pixels per side
    """
    # deflection maps
    alpha1, alpha2 = cf.call_cal_alphas(kap, bzz, ncc)
    # shear maps
    npad = 5
    al11, al12, al21, al22 = cf.call_lanczos_derivative(alpha1, alpha2, bzz, ncc)
    shear1 = 0.5*(al11-al22)
    shear2 = 0.5*(al21+al12)
    shear1[:npad, :]=0.0; shear1[-npad:,:]=0.0
    shear1[:, :npad]=0.0; shear1[:,-npad:]=0.0;
    shear2[:npad, :]=0.0; shear2[-npad:,:]=0.0
    shear2[:, :npad]=0.0; shear2[:,-npad:]=0.0;
    # magnification maps
    mu = 1.0/((1.0-kap)**2.0 - shear1*shear1 - shear2*shear2)
    # lensing potential
    phi = cf.call_cal_phi(kap, bzz, ncc)
    return alpha1, alpha2, mu, phi
