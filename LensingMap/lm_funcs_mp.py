# Python implementation using the multiprocessing module
#
from __future__ import division
import time
import collections, resource
import re, os.path, logging
import numpy as np
#import numba as nb
#from numba import njit
from pylab import *
#import scipy.stats as stats
#import scipy.interpolate as interpolate
#from scipy import ndimage 
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
# Requires: Python (2.7.13), NumPy (>= 1.8.2), SciPy (>= 0.13.3)
import sklearn
from sklearn.neighbors import KDTree
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/')
import readsnap
import multiprocessing as mp
# surpress warnings from alpha_map_fourier
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())

###############################################################################


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def source_selection(src_id, src_z, src_pos, halo_id):
    """
    Find redshift of sources which are likely to be multiple imaged
    Input:
        src_id[np.array(int)] - LightCone-IDs of sources
        src_z[np.array(float)] - redshift of sources
        halo_id[int] - ID of subhalo acting as lens
    Output:
        zs[int] - redshift of source
    """
    src_indx = np.where(src_id == halo_id)[0]
    print('the lens has so many sources: ', len(src_indx))
    dist = np.sqrt(src_pos[src_indx, 1]**2 + src_pos[src_indx, 2]**2)
    src_min = np.argsort(dist)
    #indx = np.argmin(dist)
    #indx = np.argmax(src_z[src_indx])
    start = 0
    end = 2
    if len(src_indx) > end:
        return src_z[src_indx[src_min[start:end]]], src_min[start:end], src_pos[src_indx[src_min[start:end]]]  # indx
    else:
        return src_z[src_indx[src_min[:]]], src_min[:], src_pos[src_indx[src_min[:]]]


def sigma_crit(zLens, zSource, cosmo):
    Ds = cosmo.angular_diameter_distance(zSource)
    Dl = cosmo.angular_diameter_distance(zLens)
    Dls = cosmo.angular_diameter_distance_z1z2(zLens, zSource)
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
def projected_surface_density(ll, pos, mass, centre, fov, bins=512, smooth=True,
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


def cal_lensing_signals(kap, bzz, ncc):
    dsx_arc = bzz/ncc
    # deflection maps
    alpha1, alpha2 = cf.call_cal_alphas(kap, bzz, ncc)
    
    # shear maps
    npad = 5
    #al11, al12, al21, al22 = cf.call_lanczos_derivative(alpha1, alpha2, bzz, ncc)
    al11 = 1 - np.gradient(alpha1, dsx_arc, axis=0)
    al12 = - np.gradient(alpha1, dsx_arc, axis=1)
    al21 = - np.gradient(alpha2, dsx_arc, axis=0)
    al22 = 1 - np.gradient(alpha2, dsx_arc, axis=1)
    detA = al11*al22 - al12*al21
    
    kappa0 = 1 - 0.5*(al11 + al22)
    shear1 = 0.5*(al11 - al22)
    shear2 = 0.5*(al21 + al12)
    shear0 = (shear1**2 + shear2**2)**0.5
    
    # magnification maps
    mu = 1.0/((1.0-kap)**2.0-shear1*shear1-shear2*shear2)
    
    lambda_t = 1 - kappa0 - shear0  # tangential eigenvalue, page 115
    
    # lensing potential
    phi = cf.call_cal_phi(kap, bzz, ncc)

    return alpha1, alpha2, mu, phi, detA, lambda_t


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


def einstein_radii(xs, ys, detA, lambda_t, zl, cosmo, ax, method):
    curve_crit= ax.contour(xs, ys, detA,
                           levels=(0,), colors='r',
                           linewidths=1.5, zorder=200)
    Ncrit = len(curve_crit.allsegs[0])
    curve_crit = curve_crit.allsegs[0]
    curve_crit_tan= ax.contour(xs, ys,
                               lambda_t, levels=(0,), colors='r',
                               linewidths=1.5, zorder=200)
    Ncrit_tan = len(curve_crit_tan.allsegs[0])
    if Ncrit_tan > 0:
        len_tan_crit = np.zeros(Ncrit_tan)
        for i in range(Ncrit_tan):
            len_tan_crit[i] = len(curve_crit_tan.allsegs[0][i])
        curve_crit_tan = curve_crit_tan.allsegs[0][len_tan_crit.argmax()]
        if method == 'eqv':
            Rein = np.sqrt(np.abs(area(curve_crit_tan))/np.pi)  #[arcsec]
        if method == 'med':
            dist = np.sqrt(curve_crit_tan[:, 0]**2 + curve_crit_tan[:, 1]**2)
            Rein = np.median(dist)  #[arcsec]
    else:
        curve_crit_tan= np.array([])
        Rein = 0
    return Ncrit, curve_crit, curve_crit_tan, Rein


def timedelay_magnification(mu_map, phi_map, dsx_arc, Ncells, lp1, lp2,
                            alpha1, alpha2, SrcPosSky, zs, zl, cosmo):
    """
    Calculate Photon-travel-time and Magnification of strong lensed
    supernovae

    Input:

    Output:
        len(mu): number of multiple images of supernova
        delta_t: Time it takes for photon to cover distance source-observer
        mu: luminosity magnification of source
    """
    # Mapping light rays from image plane to source plan
    [sp1, sp2] = [lp1 - alpha1, lp2 - alpha2]  #yi1,yi2[arcsec]

    # Source position [arcsec]
    x = SrcPosSky[0]*u.Mpc
    y = SrcPosSky[1]*u.Mpc
    z = SrcPosSky[2]*u.Mpc
    beta1 = ((y/x)*u.rad).to_value('arcsec')
    beta2 = ((z/x)*u.rad).to_value('arcsec')
    theta1, theta2 = cf.call_mapping_triangles([beta1, beta2], 
                                               lp1, lp2, sp1, sp2)
    # calculate magnifications of lensed Supernovae
    mu = cf.call_inverse_cic_single(mu_map, 0.0, 0.0, theta1, theta2, dsx_arc)
    # calculate time delays of lensed Supernovae in Days
    prts = cf.call_inverse_cic_single(phi_map, 0.0, 0.0, theta1, theta2, dsx_arc)
    Kc = ((1.0+zl)/const.c.to('Mpc/s') * \
          (cosmo.angular_diameter_distance(zl) * \
           cosmo.angular_diameter_distance(zs) / \
           (cosmo.angular_diameter_distance(zs) - \
            cosmo.angular_diameter_distance(zl)))).to('sday').value
    delta_t = Kc*(0.5*((theta1 - beta1)**2.0 + (theta2 - beta2)**2.0) - prts)/cf.apr**2
    beta = [beta1, beta2]
    theta = [theta1, theta2]
    return len(mu), delta_t, mu, theta, beta


def plant_Tree():
    """ Create Tree to store data hierarchical """
    return collections.defaultdict(plant_Tree)


def lenslistinit():
    global l_HFID, l_haloID, l_snapnum, l_deltat, l_mu, l_haloposbox, l_zs, l_zl, l_lensplane, l_detA, l_srctheta, l_srcbeta, l_srcID, l_tancritcurves, l_einsteinradius
    l_HFID=[]; l_haloID=[]; l_snapnum=[]; l_deltat=[]; l_mu=[]; l_haloposbox=[]; l_zs=[]; l_zl=[]; l_lensplane=[]; l_detA=[]; l_srctheta=[]; l_srcbeta=[]; l_srcID=[]; l_tancritcurves=[]; l_einsteinradius=[]
    return l_HFID, l_haloID, l_snapnum, l_deltat, l_mu, l_haloposbox, l_zs, l_zl, l_lensplane, l_detA, l_srctheta, l_srcbeta, l_srcID, l_tancritcurves, l_einsteinradius

def srclistinit():
    global s_srcID, s_deltat, s_mu, s_zs, s_alpha, s_detA, s_theta, s_beta, s_tancritcurves, s_einsteinradius
    s_srcID=[]; s_deltat=[]; s_mu=[]; s_zs=[]; s_alpha=[]; s_detA=[]; s_theta=[]; s_beta=[]; s_tancritcurves=[]; s_einsteinradius=[]
    return s_srcID, s_deltat, s_mu, s_zs, s_alpha, s_detA, s_theta, s_beta, s_tancritcurves, s_einsteinradius


def generate_lens_map(lenses, cpunum, LC, Halo_HF_ID, Halo_ID, Halo_z, FOV,
                      snapnum, snapfile, h, scale, Ncells, HQ_dir, sim, sim_phy,
                      sim_name, hfname, HaloPosBox, cosmo, results_per_cpu):
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
    print('Process %s started' % mp.current_process().name)
    first_lens = lenses[0]
    previous_snapnum = snapnum[first_lens]
    memtrack = 0

    #file = open(sim_name[sim]+'_test_'+str(mp.current_process().name)+'.txt','w') 

    lenslistinit()
    # Run through lenses
    for ll in range(first_lens, lenses[-1]):
        print('Lens Nr.:', ll)
        zs, Src_ID, SrcPosSky = source_selection(LC['Src_ID'], LC['Src_z'],
                                                 LC['SrcPosSky'], Halo_ID[ll])
        zl = Halo_z[ll]
        #Lbox = Rvir[ll]*1.5*u.Mpc
        #FOV = Rvir[ll]*1.5  #Lbox.to_value('Mpc')
        
        # converting box size and pixels size from ang. diam. dist. to arcsec
        FOV_arc = (FOV[ll]/cf.Da(zl, cosmo)*u.rad).to_value('arcsec')  #[arcsec] box size
        dsx_arc = FOV_arc/Ncells                  #[arcsec] pixel size
        # initialize the coordinates of grids (light rays on lens plan)
        lp1, lp2 = cf.make_r_coor(FOV_arc, Ncells)  #[arcsec]

        # Only load new particle data if lens is at another snapshot
        if (previous_snapnum != snapnum[ll]) or (ll == first_lens):
            print('Start loading particles')
            start = time.time() 
            snap = snapfile % (snapnum[ll], snapnum[ll])
            # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
            if hfname == 'Subfind':
                DM_pos = readsnap.read_block(snap, 'POS ', parttype=1)*h*scale  #[Mpc]
                DM_mass = readsnap.read_block(snap, 'MASS', parttype=1)*1e10/h
                Gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*h*scale  #[Mpc]
                Gas_mass = readsnap.read_block(snap, 'MASS', parttype=0)*1e10/h
                Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
                Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)
                Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)
                Star_pos = Star_pos[Star_age >= 0]*h*scale  #[Mpc]
                Star_mass = Star_mass[Star_age >= 0]*1e10/h
                del Star_age
            elif hfname == 'Rockstar':
                DM_pos = readsnap.read_block(snap, 'POS ', parttype=1)*scale  #[Mpc]
                DM_mass = readsnap.read_block(snap, 'MASS', parttype=1)*1e10/h
                Gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*scale  #[Mpc]
                Gas_mass = readsnap.read_block(snap, 'MASS', parttype=0)*1e10/h
                Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
                Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)
                Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)
                Star_pos = Star_pos[Star_age >= 0]*scale  #[Mpc]
                Star_mass = Star_mass[Star_age >= 0]*1e10/h
                del Star_age
            #file.write(str(mp.current_process().name) + 'read particles \n')
            print('Loaded particles ::::: ', time.time() - start)
        previous_snapnum = snapnum[ll]
        
        start = time.time() 
        DM_sigma, xs, ys = projected_surface_density(ll,
                                                     DM_pos,   #[Mpc]
                                                     DM_mass,
                                                     HaloPosBox[ll],
                                                     fov=FOV[ll],  #[Mpc]
                                                     bins=Ncells,
                                                     smooth=False,
                                                     smooth_fac=0.5,
                                                     neighbour_no=32)
        Gas_sigma, xs, ys = projected_surface_density(ll, Gas_pos, #*a/h,
                                                      Gas_mass,
                                                      HaloPosBox[ll], #*a/h,
                                                      fov=FOV[ll],
                                                      bins=Ncells,
                                                      smooth=False,
                                                      smooth_fac=0.5,
                                                      neighbour_no=32)
        Star_sigma, xs, ys = projected_surface_density(ll, Star_pos, #*a/h,
                                                       Star_mass,
                                                       HaloPosBox[ll], #*a/h,
                                                       fov=FOV[ll],
                                                       bins=Ncells,
                                                       smooth=False,
                                                       smooth_fac=0.5,
                                                       neighbour_no=8)
        print('Constructed all surface densities in ::::: ', time.time() - start)
        #file.write(str(mp.current_process().name) + 'created surfce densities \n')
        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        tot_sigma = DM_sigma + Gas_sigma + Star_sigma
        del DM_sigma, Gas_sigma, Star_sigma

        srclistinit()
        # Run through Sources
        check_for_sources = 0
        for ss in range(len(Src_ID)):
            # Calculate critical surface density
            sigma_cr = sigma_crit(zl, zs[ss], cosmo).to_value('Msun Mpc-2')
            kappa = tot_sigma/sigma_cr
            fig = plt.figure()
            ax = fig.add_subplot(111)
            kappa = gaussian_filter(kappa, sigma=3)
            # Calculate Deflection Maps
            alpha1, alpha2, mu_map, phi, detA, lambda_t = cal_lensing_signals(kappa,
                                                                              FOV_arc,
                                                                              Ncells) 
            del kappa
            #file.write(str(ll) + '; ' + str(ss) + '; ' + \
            #           str(mp.current_process().name) + ' lensing signals \n')
            # Calculate Einstein Radii
            Ncrit, curve_crit, curve_crit_tan, Rein = einstein_radii(lp1, lp2,
                                                                     detA,
                                                                     lambda_t,
                                                                     zl, cosmo,
                                                                     ax, 'med')
            #file.write(str(mp.current_process().name)+' Rein calc \n')
            # Calculate Time-Delay and Magnification
            n_imgs, delta_t, mu, theta, beta = timedelay_magnification(mu_map, phi,
                                                                       dsx_arc,
                                                                       Ncells, lp1, lp2,
                                                                       alpha1, alpha2,
                                                                       SrcPosSky[ss],
                                                                       zs[ss],
                                                                       zl, cosmo)
            #file.write(str(mp.current_process().name)+' time delay \n')
            if n_imgs > 0:
                #file.write(str(mp.current_process().name)+' adding multi source --------------- \n')
                
                # Tree Branch 2
                s_srcID.append(Src_ID[ss])
                s_zs.append(zs[ss])
                s_beta.append(beta)
                #s_lensplane.append([lp1, lp2])
                s_detA.append(detA)
                s_tancritcurves.append(curve_crit_tan)
                s_einsteinradius.append(Rein)  #[arcsec]
                # Tree Branch 3
                s_theta.append(theta)
                s_deltat.append(delta_t)
                s_mu.append(mu)
                check_for_sources = 1
        if check_for_sources == 1:
            # Tree Branch 1
            l_HFID.append(int(Halo_HF_ID[ll]))
            l_haloID.append(int(Halo_ID[ll]))
            l_snapnum.append(int(snapnum[ll]))
            l_zl.append(Halo_z[ll])
            l_haloposbox.append(HaloPosBox[ll])
            # Tree Branch 2
            l_srcID.append(s_srcID)
            l_zs.append(s_zs)
            l_srcbeta.append(s_beta)
            #l_lensplane.append(s_lensplane)
            l_detA.append(s_detA)
            l_tancritcurves.append(s_tancritcurves)
            l_einsteinradius.append(s_einsteinradius)
            # Tree Branch 3
            l_srctheta.append(s_theta)
            l_deltat.append(s_deltat)
            l_mu.append(s_mu)
            memuseout = (sys.getsizeof(l_HFID) + sys.getsizeof(l_haloID) + \
                     sys.getsizeof(l_snapnum) + sys.getsizeof(l_zl) + \
                     sys.getsizeof(l_srcID) + sys.getsizeof(l_zs) + \
                     sys.getsizeof(l_srcbeta) + sys.getsizeof(l_detA) + \
                     sys.getsizeof(l_tancritcurves) + sys.getsizeof(l_einsteinradius) + \
                     sys.getsizeof(l_srctheta) + sys.getsizeof(l_deltat) + \
                     sys.getsizeof(l_mu) + sys.getsizeof(l_haloposbox))/1024**3  #[GB]
        memusetot = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3  #[GB]
        #print('::::::::::::: Tot. Memory Size [GB]: ', memusetot)
        if memusetot > 5:
            ########## Save to File ########
            print('save file because it is too big')
            tree = plant_Tree()

            # Tree Branches of Node 1 : Lenses
            tree['Halo_ID'] = l_haloID
            tree['HF_ID'] = l_HFID
            tree['snapnum'] = l_snapnum
            tree['zl'] = l_zl
            tree['HaloPosBox'] = l_haloposbox
            for sid in range(len(l_haloID)):
                # Tree Branches of Node 2 : Sources
                tree['Sources']['Src_ID'][sid] = l_srcID[sid]
                tree['Sources']['zs'][sid] = l_zs[sid]
                tree['Sources']['beta'][sid] = l_srcbeta[sid]
                #tree['Sources']['LP'][sid] = l_lensplane[sid]
                tree['Sources']['detA'][sid] = l_detA[sid]
                tree['Sources']['TCC'][sid] = l_tancritcurves[sid]
                tree['Sources']['Rein'][sid] = l_einsteinradius[sid]
                for imgs in range(len(l_srcID[sid])):
                    # Tree Branches of Node 3 : Multiple Images
                    tree['Sources']['theta'][sid][imgs] = l_srctheta[sid][imgs]
                    tree['Sources']['delta_t'][sid][imgs] = l_deltat[sid][imgs]
                    tree['Sources']['mu'][sid][imgs] = l_mu[sid][imgs]

            lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+'/'+hfname+'/'+sim_name[sim]+'/'
            ensure_dir(lm_dir)
            filename = lm_dir+'LM_'+mp.current_process().name+'_' + \
                       str(memtrack)+'.pickle'
            filed = open(filename, 'wb')
            pickle.dump(tree, filed)
            filed.close()
            plt.close(fig)
            memtrack += 1
            lenslistinit()
            srclistinit()

    ########## Save to File ########
    tree = plant_Tree()

    # Tree Branches of Node 1 : Lenses
    tree['Halo_ID'] = l_haloID
    tree['HF_ID'] = l_HFID
    tree['snapnum'] = l_snapnum
    tree['zl'] = l_zl
    tree['HaloPosBox'] = l_haloposbox
    for sid in range(len(l_haloID)):
        # Tree Branches of Node 2 : Sources
        tree['Sources']['Src_ID'][sid] = l_srcID[sid]
        tree['Sources']['zs'][sid] = l_zs[sid]
        tree['Sources']['beta'][sid] = l_srcbeta[sid]
        #tree['Sources']['LP'][sid] = l_lensplane[sid]
        tree['Sources']['detA'][sid] = l_detA[sid]
        tree['Sources']['TCC'][sid] = l_tancritcurves[sid]
        tree['Sources']['Rein'][sid] = l_einsteinradius[sid]
        for imgs in range(len(l_srcID[sid])):
            # Tree Branches of Node 3 : Multiple Images
            tree['Sources']['theta'][sid][imgs] = l_srctheta[sid][imgs]
            tree['Sources']['delta_t'][sid][imgs] = l_deltat[sid][imgs]
            tree['Sources']['mu'][sid][imgs] = l_mu[sid][imgs]
    #file.close()
    lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+'/'+hfname+'/'+sim_name[sim]+'/'
    ensure_dir(lm_dir)
    filename = lm_dir+'LM_1_'+mp.current_process().name+'_' + \
               str(memtrack)+'.pickle'
    filed = open(filename, 'wb')
    pickle.dump(tree, filed)
    filed.close()
    plt.close(fig)

#                ### PLOT ###
#                f, ax = plt.subplots()
#                kappa_img = ax.imshow(np.log10(kappa).T,
#                                      #extent=[Rein, Rein, Rein, Rein],
#                                      extent=[lp1.min(), lp1.max(), lp2.min(), lp2.max()],
#                                              #vmin=np.log10(0.18),
#                                              #vmax=np.log10(5),
#                                              cmap='jet_r',
#                                              origin='lower')
#                for  curve in curve_crit:
#                    ax.plot(curve.T[0], curve.T[1], color='red', zorder=300)
#                
#                if len(curve_crit_tan)>0:
#                    plt.plot(curve_crit_tan.T[0], curve_crit_tan.T[1],
#                             color='black', lw=2.5, zorder=200)
#                    circle1 = plt.Circle((0, 0), Rein,
#                                         color='k', ls='--', fill=False)
#                    ax.add_artist(circle1)
#                        
#                cbar = f.colorbar(kappa_img)
#                cbar.set_label(r'$log(\kappa)$')
#                plt.xlabel(r'$x \quad [arcsec/h]$')
#                plt.ylabel(r'$y \quad [arcsec/h]$')
#                f.savefig('LensMapTest_'+str(ss)+'_'+str(ll)+'.png', bbox_inches='tight')
