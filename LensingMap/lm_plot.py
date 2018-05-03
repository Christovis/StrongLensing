#!/usr/bin/env python

from __future__ import division
import sys
import os
#sys.path.append("/cosma/home/durham/hvrn44/Gadget/Halo_Scripts")
#sys.path.append("/cosma/home/durham/ndcf31/data/EAGLE_SIDM/source_code/EAGLE_SIDM/eagle/python")
#sys.path.append("/cosma/home/durham/ndcf31/python")
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
from astropy import units as u
from astropy.cosmology import WMAP9
# TeX stuff
import matplotlib as mpl
from matplotlib import rc
import matplotlib.pylab as plt
import h5py
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import LensInst as LI
import SLSNreadfile as rf

#########################################################
# PLOT SETTINGS
# 'family':'sans-serif','sans-serif':['Helvetica']
# 'family': 'serif', 'serif': ['Computer Modern']
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'weight' : 'bold', 'size'   : 18})
rc('text', usetex=True)

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5

mpl.rcParams['xtick.major.size'] = 5      # major tick size in points
mpl.rcParams['xtick.minor.size'] = 3      # minor tick size in points
mpl.rcParams['xtick.major.width'] = 1    # major tick width in points
mpl.rcParams['xtick.minor.width'] = 1    # minor tick width in points

mpl.rcParams['ytick.major.size'] = 5      # major tick size in points
mpl.rcParams['ytick.minor.size'] = 3      # minor tick size in points
mpl.rcParams['ytick.major.width'] = 1    # major tick width in points
mpl.rcParams['ytick.minor.width'] = 1    # minor tick width in points

title_size = 26
legend_size = 14
axis_label_size = 18
#####################################
# SIMULATION SETTINGS
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)

###########################################################################
# Run through simulations
for sim in range(len(sim_dir))[0:1]:
    print('Simulation Nr.: ', sim)
    # File for lens & source properties
    LensPropFile = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    print(LensPropFile)
    LC = rf.LightCone_with_SN_lens(LensPropFile, 'dictionary')
    # To get header info
    snapfile = sim_dir[sim]+'/snapdir_%03d/snap_%03d'
    LensingPath = HQ_dir[0]+'lensing/'+sim_phy[sim]+sim_name[sim]
    KappaPath = LensingPath + '/kappa/'
    AlphaPath = LensingPath + '/alpha/'
    LensPropertiesPath = LensingPath + '/lens_properties/'
    
    # Sort Lenses according to Snapshot Number (SnapNM)
    indx = np.argsort(LC['snapnum'])
    Lensz = LC['Halo_z'][indx]
    zs = LC['Src_z'][indx]
    LensRvir = LC['Rvir'][indx]
    LensMvir = LC['M200'][indx]

    # Run through lenses
    for ll in range(210)[201:202]:
        print('Reading data of lens ', ll)
        KappaFile = h5py.File(KappaPath + 'Lens_' + str(ll) + '.h5')
        kappa_plane_pos = KappaFile['lens_plane_pos'].value
        kappa = KappaFile['kappa'].value
        lens_z = KappaFile['Lens_z'].value[ll]  #[redshift]
        
        AlphaFile = h5py.File(AlphaPath + 'Lens_' + str(ll) + '.h5')
        xi0 = AlphaFile['xi0'].value
        raypos = AlphaFile['RaysPos'].value
        alpha = AlphaFile['alpha'].value
        detA = AlphaFile['detA'].value
        
        LensPropertiesFile = h5py.File(LensPropertiesPath + 'Lens_' + str(ll) + '.h5')
        Ncrit = LensPropertiesFile['Ncrit'].value
        try:
            crit_curves = LensPropertiesFile['crit_curve'].value
            print('single crit_curve')
        except:
            print('many crit_curve')
            cc = LensPropertiesFile.get('crit_curve')
            crit_curves = []
            for k in range(2):
                crit_curves.append(cc[str(k)].value)
        tan_crit_curve = LensPropertiesFile['tangential_critical_curves'].value
        einstein_angle = LensPropertiesFile['eqv_einstein_radius'].value  #[arcsec]
        print('arcsec', einstein_angle)
        einstein_radius = ((einstein_angle*u.arcsec).to_value('rad') * \
                            WMAP9.angular_diameter_distance(lens_z)).to_value('Mpc')
        print('Mpc', einstein_radius)
        ##########################################
        # MAKE PLOT
        ##########################################
        xs, ys = kappa_plane_pos[0, :], kappa_plane_pos[1, :]
        f, ax = plt.subplots()
        kappa_img = ax.imshow(np.log10(kappa).T,
                              extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                              vmin=np.log10(0.18),
                              vmax=np.log10(5),
                              cmap='jet_r',
                              origin='lower')
        for crit_curve in crit_curves:
            ax.plot(crit_curve.T[0], crit_curve.T[1], color='red', zorder=300)

        if len(tan_crit_curve)>0:
            plt.plot(tan_crit_curve.T[0], tan_crit_curve.T[1],
                     color='black', lw=2.5, zorder=200)
            circle1 = plt.Circle((0, 0), einstein_radius,
                                color='k', ls='--', fill=False)
            ax.add_artist(circle1)

        cbar = f.colorbar(kappa_img)
        cbar.set_label(r'$log(\kappa)$')
        plt.xlabel(r'$x \quad [Mpc/h]$')
        plt.ylabel(r'$y \quad [Mpc/h]$')
        f.savefig('LensMapTest_'+str(ll)+'.png', bbox_inches='tight')
        plt.clf()

