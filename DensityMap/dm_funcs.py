# Python implementation using the multiprocessing module
#
from __future__ import division
import time
import collections, resource
import re, os.path, logging
import numpy as np
import h5py
from pylab import *
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# Requires: Python (2.7.13), NumPy (>= 1.8.2), SciPy (>= 0.13.3)
from sklearn.neighbors import KDTree
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib')
import read_hdf5, readsnap
import multiprocessing as mp
# surpress warnings from alpha_map_fourier
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())

################################################################################


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


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


def projected_surface_density_smooth(ll, pos, mass, centre, fov, bins, smooth,
                                     smooth_fac, neighbour_no):
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
    Lbox = fov  #[Mpc]
    Ncells = bins

    ################ Shift particle coordinates to centre ################
    pos = pos - centre
    
    if smooth_fac is None:
        print("Need to supply a value for smooth_fac when smooth=True")
        return
    if neighbour_no is None:
        print("Need to supply a value for neighbour_no when smooth=True")
        return
    ######################## DM 2D smoothed map #######################
    # Find all particles within 1.4xfov
    # actually l.o.s. is along x-axes not z-axes !
    X = pos[np.logical_and(np.abs(pos[:,0]) < 0.7*Lbox,
                           np.abs(pos[:,1]) < 0.7*Lbox)]
    M = mass[np.logical_and(np.abs(pos[:,0]) < 0.7*Lbox,
                            np.abs(pos[:,1]) < 0.7*Lbox)]
    # Find 'smoothing lengths'
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    dist, ids = kdt.query(X, k=neighbour_no, return_distance=True)
    h = np.max(dist, axis=1)  # furthest particle for each particle
    mass_in_cells, xedges, yedges = adaptively_smoothed_maps(X, h, mass=M,
                                                             fov=Lbox,
                                                             bins=Ncells,
                                                             smooth_fac=smooth_fac)
    ###################### Projected surface density ######################
    dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
    Sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
    xs = 0.5*(xedges[1:]+xedges[:-1])  #[Mpc]
    ys = 0.5*(yedges[1:]+yedges[:-1])  #[Mpc]
    return Sigma, xs, ys


#@njit("Tuple((float64[:, ::1], float64[::1], float64[::1]))"
#      "(int64, float64[:, ::1], float64[::1], float64[::1], float64, int64,"
#      "boolean, float64, int64)", fastmath=True)
def projected_surface_density(pos, mass, centre, fov, bins=512, smooth=True,
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
    Lbox = fov  #[Mpc]
    Ncells = bins

    ################ Shift particle coordinates to centre ################
    pos = pos - centre
    
    ####################### DM 2D histogram map #######################
    lenindx = np.where((pos[:, 0] < 0.5*Lbox) & (-0.5*Lbox < pos[:, 0]) &
                       (pos[:, 1] < 0.5*Lbox) & (-0.5*Lbox < pos[:, 1]))
    mass_in_cells, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                                   bins=[Ncells, Ncells],
                                                   range=[[-0.5*Lbox, 0.5*Lbox],
                                                          [-0.5*Lbox, 0.5*Lbox]],
                                                   weights=mass)
    ###################### Projected surface density ######################
    dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
    Sigma = mass_in_cells/(dx*dy)      #[Msun/Mpc^2]
    xs = 0.5*(xedges[1:]+xedges[:-1])  #[Mpc]
    ys = 0.5*(yedges[1:]+yedges[:-1])  #[Mpc]
    return Sigma, xs, ys


def area(vs):
    """
    Use Green's theorem to compute the area enclosed by the given contour.
    """
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dy = y1 - y0
        a += x0*dy
        x0 = x1
        y0 = y1
    return a


def devide_halos(halonum, cpunum, distribution):
    """
    Input:
        halonum: number of halos acting as lense
        cpunum: number of cpu's
    Output:
        lenses_per_cpu: lens ID's for each cpu
    """
    if distribution == 'equal':
        lensnum_per_cpu = np.ones(cpunum)*int(halonum/cpunum)
        lensnum_per_cpu = [int(x) for x in lensnum_per_cpu]
        missing_lenses = halonum - np.sum(lensnum_per_cpu)
        for x in range(missing_lenses):
            lensnum_per_cpu[x] += 1
        lensnum_per_cpu = np.cumsum(lensnum_per_cpu)
        lenses_per_cpu = []
        i = 0
        for x in range(cpunum):
            lenses_per_cpu.append(np.arange(i, lensnum_per_cpu[x]))
            i = lensnum_per_cpu[x]
    elif distribution == 'lin':
        linfac = np.linspace(0.5, 1.5, cpunum)[::-1]
        lensnum_per_cpu = np.ones(cpunum)*int(halonum/cpunum)
        lensnum_per_cpu = [int(x*linfac[ii]) for ii, x in enumerate(lensnum_per_cpu)]
        missing_lenses = halonum - np.sum(lensnum_per_cpu)
        sbtrk_per_cpu = int(missing_lenses/cpunum)
        lensnum_per_cpu = [int(x+sbtrk_per_cpu) for x in lensnum_per_cpu]
        missing_lenses = halonum - np.sum(lensnum_per_cpu)
        lensnum_per_cpu[0] = lensnum_per_cpu[0] + missing_lenses
        lensnum_per_cpu = np.cumsum(lensnum_per_cpu)
        lenses_per_cpu = []
        i = 0
        for x in range(cpunum):
            lenses_per_cpu.append(np.arange(i, lensnum_per_cpu[x]))
            i = lensnum_per_cpu[x]
    return lenses_per_cpu


def generate_lens_map(snapshot, sh_id, sh_vrms, sh_pos, cpunum, redshift, scale, Ncells,
                      HQ_dir, sim, sim_phy, sim_name, hfname, cosmo):
    """
    Input:
        ll: halo array indexing
        LC: Light-cone dictionary
        Halo_HF_ID: Halo-Finder Halo ID
        Halo_ID: ID of Halo
        Halo_z: redshift of Halo
        Rvir: virial radius in [Mpc]
        previous_snapnum: 
        snapnum
    Output:
    """
    
    snapshot.read(["Coordinates", "Masses", "GFM_StellarFormationTime"], parttype=[0, 1, 4])
    star_pos = snapshot.data['Coordinates']['stars']
    gas_pos = snapshot.data['Coordinates']['gas']*scale
    dm_pos = snapshot.data['Coordinates']['dm']*scale
    star_mass = snapshot.data['Masses']['stars']
    gas_mass = snapshot.data['Masses']['gas']
    dm_mass = snapshot.data['Masses']['dm']
    star_age = snapshot.data['GFM_StellarFormationTime']['stars']
    star_pos = star_pos[star_age >= 0]*scale  #[Mpc]
    star_mass = star_mass[star_age >= 0]
    del star_age, snapshot
    
    sigma_tot=[]; subhalo_id=[]
    # Run through lenses
    for ll in range(len(sh_id)):
        # Define field-of-view
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(sh_vrms[ll]/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        fov_Mpc = 4*fov_rad*sh_dist  # multiplied by for because of Oguri&Marshall

        DM_sigma, xs, ys = projected_surface_density(dm_pos,   #[Mpc]
                                                     dm_mass,
                                                     sh_pos[ll],
                                                     fov=fov_Mpc,  #[Mpc]
                                                     bins=Ncells,
                                                     smooth=False,
                                                     smooth_fac=0.5,
                                                     neighbour_no=32)
        Gas_sigma, xs, ys = projected_surface_density(gas_pos, #*a/h,
                                                      gas_mass,
                                                      sh_pos[ll], #*a/h,
                                                      fov=fov_Mpc,
                                                      bins=Ncells,
                                                      smooth=False,
                                                      smooth_fac=0.5,
                                                      neighbour_no=32)
        Star_sigma, xs, ys = projected_surface_density(star_pos, #*a/h,
                                                       star_mass,
                                                       sh_pos[ll], #*a/h,
                                                       fov=fov_Mpc,
                                                       bins=Ncells,
                                                       smooth=False,
                                                       smooth_fac=0.5,
                                                       neighbour_no=8)
        #file.write(str(mp.current_process().name) + 'created surfce densities \n')
        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        sigma_tot.append(DM_sigma + Gas_sigma + Star_sigma)
        subhalo_id.append(sh_id[ll])

        if ll % 1000 == 0:
            hf = h5py.File('./'+sim_phy[sim]+sim_name[sim]+'/DM_'+sim_name[sim]+'_' +str(ll)+'.h5', 'w')
            hf.create_dataset('density_map', data=np.asarray(sigma_tot))
            hf.create_dataset('subhalo_id', data=np.asarray(sh_id))
            hf.close()
            sigma_tot=[]; subhalo_id=[]
    hf = h5py.File('./'+sim_phy[sim]+sim_name[sim]+'/DM_'+sim_name[sim]+'_' +str(ll)+'.h5', 'w')
    hf.create_dataset('density_map', data=np.asarray(sigma_tot))
    hf.create_dataset('subhalo_id', data=np.asarray(sh_id))
    hf.close()
