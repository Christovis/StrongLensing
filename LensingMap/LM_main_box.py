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
import lenstools as lt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())
sys.settrace

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


def einstein_radii(xs, ys, detA, lambda_t, zl, cosmo, ax, method, ll):
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
    #x = SrcPosSky[0]*u.Mpc
    #y = SrcPosSky[1]*u.Mpc
    #z = SrcPosSky[2]*u.Mpc
    #if (y == 0.) and (z == 0.):
    #    beta1 = 1e-3
    #    beta2 = 1e-3
    #else:
    #    beta1 = ((y/x)*u.rad).to_value('arcsec')
    #    beta2 = ((z/x)*u.rad).to_value('arcsec')
    beta1 = 1e-3
    beta2 = 1e-3
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
    global l_HFID, l_fov, l_deltat, l_mu, l_srctheta, l_srcbeta, l_tancritcurves, l_einsteinradius
    l_HFID=[]; l_fov=[]; l_deltat=[]; l_mu=[]; l_srctheta=[]; l_srcbeta=[]; l_tancritcurves=[]; l_einsteinradius=[]
    return l_HFID, l_fov, l_deltat, l_mu, l_srctheta, l_srcbeta,l_tancritcurves, l_einsteinradius


def srclistinit():
    global s_deltat, s_mu, s_zs, s_alpha, s_theta, s_beta, s_tancritcurves, s_einsteinradius
    s_deltat=[]; s_mu=[]; s_zs=[]; s_alpha=[]; s_theta=[]; s_beta=[]; s_tancritcurves=[]; s_einsteinradius=[]
    return s_deltat, s_mu, s_zs, s_alpha, s_theta, s_beta, s_tancritcurves, s_einsteinradius


def lensing_signal():
    # Get command line arguments
    args = {}
    args["snapnum"]      = int(sys.argv[1])
    args["simdir"]       = sys.argv[2]
    args["dmdir"]        = sys.argv[3]
    args["ncells"]        = int(sys.argv[4])
    args["outbase"]      = sys.argv[5]
    
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
    fname = glob.glob(args["dmdir"]+'*.h5')
    ffname = []
    for ff in range(len(fname)):
        #if (os.path.getsize(fname[ff])/(1024*1024.0)) < 1:
        #    fname[ff] = 0
        #else:
        ffname.append(fname[ff])
    ffname = np.asarray(ffname)

    # Cosmological Parameters
    cosmo = LambdaCDM(H0=s.header.hubble*100,
                      Om0=s.header.omega_m,
                      Ode0=s.header.omega_l)
    redshift = s.header.redshift
    print('Analyse sub-&halo at z=%f' % redshift)
   
    # Calculate critical surface density
    zl = redshift
    zs = 0.409
    sigma_cr = sigma_crit(zl, zs, cosmo).to_value('Msun Mpc-2')
    
    lenslistinit(); srclistinit()
    # Run through files
    for ff in range(len(ffname))[:]:
        print('\n')
        print('------------- \n Reading File: %s' % (fname[ff].split('/')[-2:]))
        dmf = h5py.File(ffname[ff], 'r')
    
        # Run through lenses
        for ll in range(len(dmf['HFID']))[:]:
            print('Works on lens %d with ID %d' % \
                    (ll, dmf['HFID'][ll]))
            # convert. box size and pixels size from ang. diam. dist. to arcsec
            FOV_arc = (dmf['FOV'][ll]/cf.Da(zl, cosmo)*u.rad).to_value('arcsec')
            dsx_arc = FOV_arc/args["ncells"]  #[arcsec] pixel size
            # initialize the coordinates of grids (light rays on lens plan)
            lpv = np.linspace(-(FOV_arc-dsx_arc)/2, (FOV_arc-dsx_arc)/2, args["ncells"])
            lp1, lp2 = np.meshgrid(lpv, lpv)  #[arcsec]

            # Calculate convergence map
            kappa = dmf['DMAP'][ll]/sigma_cr
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Calculate Deflection Maps
            alpha1, alpha2, mu_map, phi, detA, lambda_t = lt.cal_lensing_signals(
                    kappa, FOV_arc, args["ncells"]) 
            # Calculate Einstein Radii
            Ncrit, curve_crit, curve_crit_tan, Rein = lt.einstein_radii(
                    lp1, lp2, detA, lambda_t, zl, cosmo, ax, 'med')
            # Calculate Time-Delay and Magnification
            beta = np.array([0., 0., 0.])
            n_imgs, delta_t, mu, theta = lt.timedelay_magnification(
                    mu_map, phi, dsx_arc, args["ncells"],
                    lp1, lp2, alpha1, alpha2, beta, zs, zl, cosmo)
            if n_imgs > 1:
                #TODO: does n_imgs include the original
                print(' -> %d multiple lensed images' % (n_imgs))
                # Tree Branch 1
                l_HFID.append(int(dmf['HFID'][ll]))
                l_fov.append(FOV_arc)
                # Tree Branch 2
                l_srcbeta.append(beta)
                l_tancritcurves.append(curve_crit_tan)
                l_einsteinradius.append(Rein)
                # Tree Branch 3
                l_srctheta.append(theta)
                l_deltat.append(delta_t)
                l_mu.append(mu)

    
    ########## Save to File ########
    tree = plant_Tree()
    # Tree Branches of Node 1 : Lenses
    tree['HF_ID'] = l_HFID
    tree['snapnum'] = args["snapnum"]
    tree['zl'] = redshift
    tree['zs'] = zs
    tree['FOV'] = l_fov  #[arcsec] for glafic
    # Tree Branches of Node 1 : Sources
    tree['Sources']['beta'] = l_srcbeta
    tree['Sources']['TCC'] = l_tancritcurves
    tree['Sources']['Rein'] = l_einsteinradius
    for imgs in range(len(l_mu)):
        # Tree Branches of Node 2 : Multiple Images
        tree['Sources']['theta'][imgs] = l_srctheta[imgs]
        tree['Sources']['delta_t'][imgs] = l_deltat[imgs]
        tree['Sources']['mu'][imgs] = l_mu[imgs]
    label = args["simdir"].split('/')[-2].split('_')[2]
    zllabel = str(redshift).replace('.', '')[:3].zfill(3)
    zslabel = '{:<03d}'.format(int(str(zs).replace('.', '')))
    filename = args["outbase"]+'LM_%s_zl%szs%s.pickle' % (label, zllabel, zslabel)
    filed = open(filename, 'wb')
    pickle.dump(tree, filed)
    filed.close()
    plt.close(fig)


if __name__ == '__main__':
    lensing_signal()
