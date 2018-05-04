from __future__ import division
import sys
import os
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import h5py
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import LensInst as LI
import SLSNreadfile as rf

################################################################################
# PLOT SETTINGS
Halo_ID = 45  # which lens to plot

rc('font', **{'family': 'serif',
              'serif': ['Computer Modern'],
              'weight' : 'bold', 'size'   : 18})
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
################################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, dd, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

################################################################################
# Run through simulations
for sim in range(len(sim_dir))[0:1]:
    print('Plot lensing map for: ', sim_name[sim], 'Halo ID: ', Halo_ID)
    # LightCone file for lens & source properties
    lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    # LensingMap files
    lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
    lm_file = lm_dir+'LM_'+str(Halo_ID)+'.h5'
   
    # Load LightCone Contents
    LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')
    # Load LensingMap Contents
    LM = h5py.File(lm_file)


    Grid = KappaFile['Grid'].value
    kappa = KappaFile['kappa'].value
    zl = KappaFile['zl'].value
    
    LensPropertiesFile = h5py.File(LensPropertiesPath + 'Lens_' + str(ll) + '.h5')
    Ncrit = LensPropertiesFile['Ncrit'].value
    try:
        crit_curves = LensPropertiesFile['crit_curve'].value
        print('single crit_curve')
    except:
        cc = LensPropertiesFile.get('crit_curve')
        crit_curves = []
        for k in range(2):
            crit_curves.append(cc[str(k)].value)
        print('many crit_curve')
    tan_crit_curve = LensPropertiesFile['tangential_critical_curves'].value
    einstein_angle = LensPropertiesFile['eqv_einstein_radius'].value  #[arcsec]
    einstein_radius = ((einstein_angle*u.arcsec).to_value('rad') * \
                        cosmo.angular_diameter_distance(lens_z)).to_value('Mpc')
    
    ##########################################
    # MAKE PLOT
    ##########################################
    xs, ys = Grid[0, :], Grid[1, :]
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

