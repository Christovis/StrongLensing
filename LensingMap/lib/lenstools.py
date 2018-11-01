# Python implementation using the multiprocessing module
#
from __future__ import division
import collections, resource
import os, sys, glob
import scipy
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pandas as pd
import h5py, pickle, pandas
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/lib/')
import cfuncs as cf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())
sys.settrace


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
    print('This lens has %d sources' % len(src_indx))
    dist = np.sqrt(src_pos[src_indx, 1]**2 + src_pos[src_indx, 2]**2)
    src_min = np.argsort(dist)
    #indx = np.argmin(dist)
    #indx = np.argmax(src_z[src_indx])
    start = 0
    end = 12
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


def einstein_radii(xs, ys, detA, lambda_t, zl, cosmo, ax, method):
    curve_crit = ax.contour(xs, ys, detA,
                            levels=(0,), colors='r',
                            linewidths=1.5, zorder=200)
    Ncrit = len(curve_crit.allsegs[0])
    curve_crit = curve_crit.allsegs[0]
    curve_crit_tan = ax.contour(xs, ys,
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
    Input:
        mu_map: 2D magnification map
        phi_map: 2D potential map
        dsx_arc: cell size in arcsec
        Ncells: number of cells
        lp1, lp2: lens place grid coordinates
        alpha1, alpha2: 2D deflection map
        SrcPosSky: source position in Mpc
        zs: source redshift
        zl: lens redshift

    Output:
        len(mu): number of multiple images of supernova
        delta_t: Time it takes for photon to cover distance source-observer
        mu: luminosity magnification of source
    """
    # Mapping light rays from image plane to source plan
    [sp1, sp2] = [lp1 - alpha1, lp2 - alpha2]  #[arcsec]

    # Source position [arcsec]
    x = SrcPosSky[0]*u.Mpc
    y = SrcPosSky[1]*u.Mpc
    z = SrcPosSky[2]*u.Mpc
    if (y == 0.) and (z == 0.):
        beta1 = 1e-3
        beta2 = 1e-3
    else:
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
