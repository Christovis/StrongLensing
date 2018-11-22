# File Description:
#   Converts density maps to convergence maps to calculate
#   strong lensing observables (deflection, magnification,
#   time-delays, source positions on lense-plane)
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
import lm_cfuncs as cf
import lenstools as lt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())
sys.settrace


def plant_Tree():
    """ Create Tree to store data hierarchical """
    return collections.defaultdict(plant_Tree)


def lenslistinit():
    global l_HFID,l_fov,l_deltat,l_mu,l_srctheta,l_srcbeta,l_tancritcurves,l_caustic,l_einsteinradius
    l_HFID=[]; l_fov=[]; l_deltat=[]; l_mu=[]; l_srctheta=[]; l_srcbeta=[]; l_tancritcurves=[]
    l_caustic=[]; l_einsteinradius=[]
    return l_HFID,l_fov,l_deltat,l_mu,l_srctheta,l_srcbeta,l_tancritcurves,l_caustic,l_einsteinradius


def srclistinit():
    global s_deltat, s_mu, s_zs, s_alpha, s_theta, s_beta, s_tancritcurves, s_einsteinradius
    s_deltat=[]; s_mu=[]; s_zs=[]; s_alpha=[]; s_theta=[]; s_beta=[]; s_tancritcurves=[]; s_einsteinradius=[]
    return s_deltat, s_mu, s_zs, s_alpha, s_theta, s_beta, s_tancritcurves, s_einsteinradius


def lensing_signal():
    # Get command line arguments
    args = {}
    args["snapnum"]      = int(sys.argv[1])
    args["simdir"]       = sys.argv[2]
    args["hfname"]       = sys.argv[3]
    args["dmdir"]        = sys.argv[4]
    args["ncells"]       = int(sys.argv[5])
    args["outbase"]      = sys.argv[6]
    
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
    unitlength = lt.define_unit(s.header.unitlength, args["hfname"])

    fname = glob.glob(args["dmdir"]+'*.h5')
    dmfile = []
    for ff in range(len(fname)):
        #if (os.path.getsize(fname[ff])/(1024*1024.0)) < 1:
        #    fname[ff] = 0
        #else:
        dmfile.append(fname[ff])
    dmfile = np.asarray(dmfile)

    # Cosmological Parameters
    cosmo = LambdaCDM(H0=s.header.hubble*100,
                      Om0=s.header.omega_m,
                      Ode0=s.header.omega_l)
    redshift = s.header.redshift
    print('Analyse sub-&halo at z=%f' % redshift)
   
    # Calculate critical surface density
    zl = redshift
    zs = 0.409
    sigma_cr = lt.sigma_crit(zl, zs, cosmo).to_value('Msun %s-2' % unitlength)
    
    lenslistinit(); srclistinit()
    # Run through files
    for ff in range(len(dmfile)):
        print('\n')
        print('------------- \n Reading File: %s' % (fname[ff].split('/')[-2:]))
        dmf = h5py.File(dmfile[ff], 'r')
        print('Nr. of galxies:', len(dmf['HFID']))

        # Run through lenses
        for ll in range(len(dmf['HFID'])):
            # convert. box size and pixels size from ang. diam. dist. to arcsec
            FOV_arc = (dmf['FOV'][ll]/cf.Da(zl, unitlength, cosmo) * \
                       u.rad).to_value('arcsec')
            dsx_arc = FOV_arc/args["ncells"]  #[arcsec] pixel size
            # initialize the coordinates of grids (light rays on lens plan)
            lp1, lp2, lpv = cf.make_r_coor(FOV_arc, args["ncells"])

            # Calculate convergence map
            kappa = dmf['DMAP'][ll]/sigma_cr
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Calculate Deflection Maps
            alpha1, alpha2, mu_map, phi, detA, lambda_t, lpv = lt.cal_lensing_signals(
                    kappa, FOV_arc, args["ncells"], lpv) 
            lp2, lp1 = np.meshgrid(lpv, lpv)  #[arcsec] finer resol. in centre
            # Mapping light rays from image plane to source plan
            [sp1, sp2] = [lp1 - alpha1, lp2 - alpha2]  #[arcsec]

            # Calculate Einstein Radii
            Ncrit, curve_crit_tan, caustic, Rein = lt.einstein_radii(
                    lp1, lp2, sp1, sp2, detA, lambda_t, cosmo, ax, 'med')
            # Calculate Time-Delay and Magnification
            beta = np.array([0., 0.])
            n_imgs, delta_t, mu, theta = lt.timedelay_magnification(
                    mu_map, phi, dsx_arc, args["ncells"],
                    lp1, lp2, alpha1, alpha2, beta, zs, zl, cosmo)
            if n_imgs > 1:
                #TODO: does n_imgs include the original
                print('Galaxy %d/%d got %d multiple lensed images' % \
                        (ll, len(dmf['HFID']), n_imgs))
                # Tree Branch 1
                l_HFID.append(int(dmf['HFID'][ll]))
                l_fov.append(FOV_arc)
                # Tree Branch 2
                l_srcbeta.append(beta)
                l_tancritcurves.append(curve_crit_tan)
                l_caustic.append(caustic)
                l_einsteinradius.append(Rein)
                # Tree Branch 3
                l_srctheta.append(theta)
                l_deltat.append(delta_t)
                l_mu.append(mu)

    
    ########## Save to File ########
    print('%d galaxies produce multiple images SN Ia' % (len(l_HFID)))
    tree = plant_Tree()
    # Tree Branches of Node 1 : Lenses
    tree['HF_ID'] = l_HFID
    tree['snapnum'] = args["snapnum"]
    tree['zl'] = redshift
    tree['zs'] = zs
    tree['FOV'] = l_fov  #[arcsec] for glafic
    # Tree Branches of Node 1 : Sources
    tree['Sources']['beta'] = l_srcbeta
    tree['Sources']['CAU'] = l_caustic
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
