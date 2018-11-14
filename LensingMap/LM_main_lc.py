# Python implementation using the multiprocessing module
#
from __future__ import division
import collections, resource
import os, sys, glob
import scipy, math
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
#sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/backup/')
#import testkappamap as kmap
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())
sys.settrace


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


def lensing_signal():
    # Get command line arguments
    args = {}
    args["simdir"]       = sys.argv[1]
    args["dmdir"]        = sys.argv[2]
    args["lcdir"]        = sys.argv[3]
    args["outbase"]      = sys.argv[4]
    args["ncells"]      = int(sys.argv[5])
    #args["simdir"]       = '/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_F5_kpc/'
    #args["dmdir"]        = '/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_F5_kpc/Lightcone/'
    #args["lcdir"]        = '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/Rockstar/LC_SN_L62_N512_F5_kpc'
    #args["outbase"]      = '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/L62_N512_F5_kpc/Lightcone/'
    #args["ncells"]      = 512
    
    # Names of all available Density maps
    dmfile = glob.glob(args["dmdir"]+'*.h5')
    dmfile.sort(key = lambda x: x[-4])
    # Names of all available Lightcones
    lcfile = glob.glob(args["lcdir"]+'*.h5')
    lcfile.sort(key = lambda x: x[-4])

    lenslistinit(); srclistinit()
    # Run through files
    for ff in range(len(dmfile)):
        print('\n')
        print('------------- \n Reading Files: \n%s\n%s' % \
                (dmfile[ff].split('/')[-2:], lcfile[ff].split('/')[-2:]))
        # Load density maps
        dmf = h5py.File(dmfile[ff], 'r')
        dmdf = pd.DataFrame({'HF_ID' : dmf['HF_ID'],
                             'LC_ID' : dmf['LC_ID'],
                             'fov_Mpc' : dmf['fov_Mpc']})
        s1 = pd.Series(dict(list(enumerate(dmf['density_map']))), index=dmdf.index)
        dmdf['density_map'] = s1
        dmdf = dmdf.set_index('LC_ID')

        # Load Lightcones
        lcf = h5py.File(lcfile[ff], 'r')
        lcdf = pd.DataFrame({'HF_ID' : lcf['HF_ID'].value,
                             'LC_ID' : lcf['LC_ID'].value,
                             'zl' : lcf['Halo_z'].value,
                             'vrms' : lcf['VelDisp'].value,
                             'snapnum' : lcf['snapnum'].value,
                             'fov_Mpc' : lcf['FOV'][:][1]})
        lcdf = lcdf.set_index('LC_ID')
        srcdf = {'Src_ID' : lcf['Src_ID'].value,
                 'zs' : lcf['Src_z'].value,
                 'SrcPosSky' : lcf['SrcPosSky'].value,
                 'SrcAbsMag' : lcf['SrcAbsMag'].value}

        print('The minimum Vrms is %f' % (np.min(lcdf['vrms'].values)))

        dmdf = dmdf.sort_values(by=['LC_ID'])
        lcdf = lcdf.sort_values(by=['LC_ID'])
        # sanity check
        assert len(lcdf.index.intersection(dmdf.index)) == len(dmdf.index.values)
        lcdf['density_map'] = dmdf['density_map']
        
        #lcdf = lcdf.sort_values(by=['snapnum'])
        s = read_hdf5.snapshot(45, args["simdir"])
        # Cosmological Parameters
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
    
        # Run through lenses
        print('There are %d lenses in file' % (len(lcdf.index.values)))
        for ll in range(len(lcdf.index.values)):
            lens = lcdf.iloc[ll]
            #print('working on lens %d' % lens['HF_ID'])
            # convert. box size and pixels size from ang. diam. dist. to arcsec
            FOV_arc = (lens['fov_Mpc']/cf.Da(lens['zl'], cosmo)*u.rad).to_value('arcsec')
            dsx_arc = FOV_arc/args["ncells"]  #[arcsec] pixel size
            # initialize the coordinates of grids (light rays on lens plan)
            lpv = np.linspace(-(FOV_arc-dsx_arc)/2, (FOV_arc-dsx_arc)/2, args["ncells"])
            lp1, lp2 = np.meshgrid(lpv, lpv)  #[arcsec]
       
            zs, Src_ID, SrcPosSky = lt.source_selection(
                    srcdf['Src_ID'], srcdf['zs'], srcdf['SrcPosSky'],
                    lcdf.index.values[ll])
            
            # Run through sources
            check_for_sources = 0
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for ss in range(len(Src_ID)):
                # Calculate critical surface density
                sigma_cr = lt.sigma_crit(lens['zl'],
                                         zs[ss],
                                         cosmo).to_value('Msun Mpc-2')

                # convert source position from Mpc to arcsec
                beta = lt.mpc2arc(SrcPosSky[ss])
                beta = [bb*1e-3 for bb in beta]

                # Calculate convergence map
                kappa = lens['density_map']/sigma_cr
                #fig = plt.figure()
                #ax = fig.add_subplot(111)
                
                # Calculate Deflection Maps
                alpha1, alpha2, mu_map, phi, detA, lambda_t = lt.cal_lensing_signals(
                        kappa, FOV_arc, args["ncells"]) 
                # Calculate Einstein Radii in [arcsec]
                Ncrit, curve_crit, curve_crit_tan, Rein = lt.einstein_radii(
                        lp1, lp2, detA, lambda_t, lens['zl'], cosmo, ax, 'med')
                #if Rein == 0. or math.isnan(Rein):
                #    print('!!! Rein is 0. or NaN')
                #    continue
                # Calculate Time-Delay and Magnification
                n_imgs, delta_t, mu, theta  = lt.timedelay_magnification(
                        mu_map, phi, dsx_arc, args["ncells"],
                        lp1, lp2, alpha1, alpha2, beta,
                        zs[ss], lens['zl'], cosmo)
                if n_imgs > 1:
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
                    #print(' -> %d multiple lensed images' % (n_imgs))
            if check_for_sources == 1:
                # Tree Branch 1
                l_HFID.append(int(lens['HF_ID']))
                l_haloID.append(int(lcdf.index.values[ll]))
                l_snapnum.append(int(lens['snapnum']))
                l_zl.append(lens['zl'])
                #l_haloposbox.append(HaloPosBox[ll])
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
                srclistinit()
                check_for_sources = 0
                print('Save data of lens %d' % ll)
    
    ########## Save to File ########
    tree = plant_Tree()
    ## Tree Branches of Node 1 : Lenses
    #tree['HF_ID'] = l_HFID
    #tree['snapnum'] = args["snapnum"]
    #tree['zl'] = redshift
    #tree['zs'] = zs
    ## Tree Branches of Node 1 : Sources
    #tree['Sources']['beta'] = l_srcbeta
    #tree['Sources']['TCC'] = l_tancritcurves
    #tree['Sources']['Rein'] = l_einsteinradius
    #for imgs in range(len(l_mu)):
    #    # Tree Branches of Node 2 : Multiple Images
    #    tree['Sources']['theta'][imgs] = l_srctheta[imgs]
    #    tree['Sources']['delta_t'][imgs] = l_deltat[imgs]
    #    tree['Sources']['mu'][imgs] = l_mu[imgs]

    # Tree Branches of Node 1 : Lenses
    tree['LC_ID'] = l_haloID
    tree['HF_ID'] = l_HFID
    tree['snapnum'] = l_snapnum
    tree['zl'] = l_zl
    #tree['HaloPosBox'] = l_haloposbox
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
    label = args["simdir"].split('/')[-2].split('_')[2]
    filename = args["outbase"]+'LM_%s.pickle' % (label)
    filed = open(filename, 'wb')
    pickle.dump(tree, filed)
    filed.close()
    plt.close(fig)


#args["simdir"]       = '/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/'
#args["dmdir"]        = '/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/Lightcone/'
#args["lcdir"]        = '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/Rockstar/LC_SN_L62_N512_GR_kpc'
#args["outbase"]      = '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/L62_N512_GR_kpc/Lightcone/'
#args["ncells"]      = 512
if __name__ == '__main__':
    lensing_signal()
