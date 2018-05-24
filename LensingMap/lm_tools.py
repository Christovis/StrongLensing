from __future__ import division
import re
import os.path
import logging
import numpy as np
from pylab import *
import scipy.stats as stats
import scipy.interpolate as interpolate
from scipy import ndimage 
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
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
    dist = np.sqrt(src_pos[src_indx, 1]**2 + src_pos[src_indx, 2]**2)
    src_min = np.argsort(dist)
    #indx = np.argmin(dist)
    #indx = np.argmax(src_z[src_indx])
    start = 0
    end = 8
    return src_z[src_indx[src_min[start:end]]], src_min[start:end], src_pos[src_indx[src_min[start:end]]]  # indx


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
        alpha *= xi0_load/xi0.to('Mpc').value
        xtest *= xi0_load/xi0.to('Mpc').value
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
    alpha = [alphax, alphay]
    rays = [xrays, yrays]
    return alpha, detA, rays, lambda_t, lambda_r


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


def cal_lensing_signals(kap, bzz, ncc):
    """
    Calculate Lensing Signals
    Input:
        kap: convergence map
        bzz: box size, arcsec
        ncc: number of pixels per side
    
    Output:
        alpha1, alpha2: Deflection Map
        mu: Magnification Map
        phi: Lensing Potential
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


def devide_halos(halonum, cpunum):
    """
    Input:
        halonum: number of halos acting as lense
        cpunum: number of cpu's
    Output:
        lenses_per_cpu: lens ID's for each cpu
    """
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
    logging.info('Lenses per CPU:', lensnum_per_cpu)
    return lenses_per_cpu


def einstein_radii(kappa, xs, ys, zl, xi0, Nrays, Lrays, cosmo, ax):
    """
    Calculate Einstein Radius, tan- and critical curve,
    map of deflection angles etc.

    Input:
        kappa: convergence map
        xs, ys: Grid cell positions
        zl: Redshift of Lens
        xi0: convert length scale
        Nrays: number of light rays
        Lrays: length of light rays
        cosmo: Cosmology

    Output:
        Rein_eqv: Einstein Radius [arcsec]
        curve_crit_tan:
        curve_crit:
        detA:
        alpha:
        Ncrit:
    """
    # Calculate deflection angle
    alpha, detA, rays, lambda_t, lambda_r = alpha_from_kappa(
            kappa, xs, ys, xi0, Nrays, Lrays.to_value('Mpc'))
    xraysgrid, yraysgrid = np.meshgrid(rays[0], rays[1], indexing='ij')

    curve_crit= ax.contour(xraysgrid*xi0, yraysgrid*xi0, detA,
                                 levels=(0,), colors='r',
                                 linewidths=1.5, zorder=200)
    Ncrit = len(curve_crit.allsegs[0])
    curve_crit = curve_crit.allsegs[0]
    curve_crit_tan= ax.contour(xraysgrid*xi0, yraysgrid*xi0,
                               lambda_t, levels=(0,), colors='r',
                               linewidths=1.5, zorder=200)
    Ncrit_tan = len(curve_crit_tan.allsegs[0])
    if Ncrit_tan > 0:
       len_tan_crit = np.zeros(Ncrit_tan)
       for i in range(Ncrit_tan):
          len_tan_crit[i] = len(curve_crit_tan.allsegs[0][i])
       curve_crit_tan = curve_crit_tan.allsegs[0][len_tan_crit.argmax()]
       Rein_eqv = ((np.sqrt(np.abs(area(curve_crit_tan))/np.pi) * \
                               u.Mpc/cosmo.angular_diameter_distance(zl)) * \
                               u.rad).to_value('arcsec')
    else:
       curve_crit_tan= np.array([])
       Rein_eqv = 0
    return rays, alpha, detA, Ncrit, curve_crit, curve_crit_tan, Rein_eqv


def timedelay_magnification(FOV, Ncells, kappa, SrcPosSky, zs, zl, cosmo):
    """
    Calculate Photon-travel-time and Magnification of strong lensed
    supernovae

    Input:
        FOV: Field-of-View [Mpc]
        Ncells: number of cells for grid on FOV
        kappa: convergence map
        SrcPosSky: Position of Source relative to Lens [Mpc]
        zs: Redshift of Source
        zl: Redshift of Lens
        cosmo: Cosmology

    Output:
        len(mu): number of multiple images of supernova
        delta_t: Time it takes for photon to cover distance source-observer
        mu: luminosity magnification of source
    """

    #converting box size and pixels size from co-moving distance to arcsec
    FOV_arc = FOV/cf.Dc(zl, cosmo)*cf.apr     #[arcsec] box size
    dsx_arc = FOV_arc/Ncells                  #[arcsec] pixel size
    # initialize the coordinates of grids (light rays on lens plan)
    lp1, lp2 = cf.make_r_coor(FOV_arc, Ncells)
    ds = FOV_arc/Ncells  # grid-cell edge length
    # Calculate the maps of deflection angles, magnifications,
    # and lensing potential
    kappa = gaussian_filter(kappa, sigma=3)
    ai1, ai2, mua, phia = cal_lensing_signals(kappa, FOV_arc, Ncells)
    # Mapping light rays from image plane to source plan
    sp1=lp1-ai1; sp2=lp2-ai2  #[arcsec]

    # Source position [arcsec]
    x = SrcPosSky[0]*u.Mpc
    y = SrcPosSky[1]*u.Mpc
    z = SrcPosSky[2]*u.Mpc
    beta1 = ((y/x)*u.rad).to_value('arcsec')
    beta2 = ((z/x)*u.rad).to_value('arcsec')
    theta1, theta2 = cf.call_mapping_triangles([beta1, beta2], 
                                               lp1, lp2, sp1, sp2)
    # calculate magnifications of lensed Supernovae
    mu = cf.call_inverse_cic_single(mua, 0.0, 0.0, theta1, theta2, dsx_arc)
    # calculate time delays of lensed Supernovae in Days
    prts = cf.call_inverse_cic_single(phia,0.0,0.0,theta1,theta2,dsx_arc)
    Kc = (1.0+zl)/cf.vc * \
         (cf.Da(zl, cosmo)*cf.Da(zs, cosmo)/cf.Da2(zl, zs, cosmo)) * \
         ((cf.Mpctometer*100/cosmo.H0)/(1e3*cf.Daytosec*cf.apr*cf.apr))
    delta_t = Kc*(0.5*((theta1 - beta1)**2.0 + (theta2 - beta2)**2.0) - prts)
    logging.info('There are %d images', len(mu))
    return len(mu), delta_t, mu

def generate_lens_map(lenses, LC, Halo_Rockstar_ID, Halo_ID, Halo_z, Rvir,
            snapnum, snapfile, h, scale, Ncells, Nrays, Lrays,
            HQ_dir, sim, sim_phy, sim_name, HaloPosBox, xi0, HaloVel, cosmo):
    """
    Input:
        ll: halo array indexing
        LC: Light-cone dictionary
        Halo_ID: ID of Halo
        Halo_z: redshift of Halo
        Rvir: virial radius in [Mpc]
        previous_snapnum: 
        snapnum
    Output:
    """
    logging.info('Process %s started' % mp.current_process().name)
    
    first_lens = lenses[0]
    previous_snapnum = snapnum[first_lens]
    # Run through lenses
    for ll in range(first_lens, lenses[-1]):
        zs, Src_ID, SrcPosSky = source_selection(LC['Src_ID'], LC['Src_z'],
                                                 LC['SrcPosSky'], Halo_ID[ll])
        zl = Halo_z[ll]
        Lbox = Rvir[ll]*0.3*u.Mpc
        FOV = Lbox.to_value('Mpc')

        # Only load new particle data if lens is at another snapshot
        if (previous_snapnum != snapnum[ll]) or (ll == first_lens):
            snap = snapfile % (snapnum[ll], snapnum[ll])
            # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
            DM_pos = readsnap.read_block(snap, 'POS ', parttype=1)*scale
            DM_mass = readsnap.read_block(snap, 'MASS', parttype=1)*1e10/h
            Gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*scale
            Gas_mass = readsnap.read_block(snap, 'MASS', parttype=0)*1e10/h
            Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale
            Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
            Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)
            Star_pos = Star_pos[Star_age >= 0]
            Star_mass = Star_mass[Star_age >= 0]*1e10/h
            del Star_age
            BH_pos = readsnap.read_block(snap, 'POS ', parttype=5)*scale
            BH_mass = readsnap.read_block(snap, 'MASS', parttype=5)*1e10/h
        previous_snapnum = snapnum[ll]
        
        DM_sigma, xs, ys = projected_surface_density(DM_pos,
                                                     DM_mass,
                                                     HaloPosBox[ll],
                                                     fov=FOV,
                                                     bins=Ncells,
                                                     smooth=False,
                                                     smooth_fac=0.5,
                                                     neighbour_no=32)
        Gas_sigma, xs, ys = projected_surface_density(Gas_pos, #*a/h,
                                                      Gas_mass,
                                                      HaloPosBox[ll], #*a/h,
                                                      fov=FOV,
                                                      bins=Ncells,
                                                      smooth=False,
                                                      smooth_fac=0.5,
                                                      neighbour_no=32)
        Star_sigma, xs, ys = projected_surface_density(Star_pos, #*a/h,
                                                       Star_mass,
                                                       HaloPosBox[ll], #*a/h,
                                                       fov=FOV,
                                                       bins=Ncells,
                                                       smooth=False,
                                                       smooth_fac=0.5,
                                                       neighbour_no=8)
        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        tot_sigma = DM_sigma + Gas_sigma + Star_sigma

        # Run through Sources
        for ss in range(len(Src_ID)):
            print(mp.current_process().name, Halo_ID[ll], ss)
            # Calculate critical surface density
            sigma_cr = sigma_crit(zl, zs[ss], cosmo).to_value('Msun Mpc-2')
            kappa = tot_sigma/sigma_cr
            fig = plt.figure()
            ax = fig.add_subplot(111)
            rays, alpha, detA, Ncrit, curve_crit, curve_crit_tan, Rein = einstein_radii(kappa, xs, ys, zl, xi0, Nrays, Lrays, cosmo, ax)
            n_imgs, delta_t, mu = timedelay_magnification(FOV, Ncells, kappa,
                                                          SrcPosSky[ss], zs[ss],
                                                          zl, cosmo)
            
            ########## Save to File ########
            # xs, ys in Mpc in lens plane, kappa measured on that grid
            # xrays, yrays, alphax, alphay in dimensionless coordinates
            if len(mu) > 1:
                lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
                ensure_dir(lm_dir)
                filename = lm_dir+'LM_L'+str(Halo_ID[ll])+'_S'+str(Src_ID[ss])+'.h5'
                    
                LensPlane = [xs, ys]

                hf = h5py.File(filename, 'w')
                hf.create_dataset('Halo_Rockstar_ID', data=Halo_Rockstar_ID[ll])
                hf.create_dataset('Halo_ID', data=Halo_ID[ll])
                hf.create_dataset('snapnum', data=snapnum[ll])
                hf.create_dataset('delta_t', data=delta_t)
                hf.create_dataset('mu', data=mu)
                hf.create_dataset('HaloPosBox', data=HaloPosBox[ll])
                hf.create_dataset('HaloVel', data=HaloVel[ll])
                hf.create_dataset('zs', data=zs[ss])
                hf.create_dataset('zl', data=zl)
                hf.create_dataset('Grid', data=LensPlane)
                hf.create_dataset('RaysPos', data=rays)
                hf.create_dataset('DM_sigma', data=DM_sigma)
                hf.create_dataset('Gas_sigma', data=Gas_sigma)
                hf.create_dataset('Star_sigma', data=Star_sigma)
                hf.create_dataset('kappa', data=kappa)
                hf.create_dataset('alpha', data=alpha)
                hf.create_dataset('detA', data=detA)
                hf.create_dataset('Ncrit', data=Ncrit)
                try:
                    hf.create_dataset('crit_curves', data=curve_crit)
                except:
                    cc = hf.create_group('crit_curve')
                    for k, v in enumerate(curve_crit):
                        cc.create_dataset(str(k), data=v)
                hf.create_dataset('tangential_critical_curves',
                                  data=curve_crit_tan)
                hf.create_dataset('eqv_einstein_radius', data=Rein)
                hf.close()
            plt.close(fig)
