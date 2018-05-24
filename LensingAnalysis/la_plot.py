from __future__ import division
import os, os.path
import glob
import sys
import re
import numpy as np
from matplotlib import pyplot as plt, rc
import h5py
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import readlensing as rf

rc('figure', figsize=(8,6))
rc('font', size=18)
rc('lines', linewidth=3)
rc('axes', linewidth=2)
rc('xtick.major', width=2)
rc('ytick.major', width=2)

############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hd_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)
lens_dir = '/cosma5/data/dp004/dc-beck3/LensingMap/'

h = 0.6774
labels = ['FP_GR', 'FP_F6']
colour = ['r', 'b']
############################################################################
for sim in range(len(sim_dir)):
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    # LensingAnalysis file
    la_dir = HQ_dir+'LensingAnalysis/'+sim_phy[sim] + \
             'DynLensMass_'+sim_name[sim]+'.h5'
    # File for lens & source properties
    lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'

    # Load LightAnalysis Contents
    LA = h5py.File(la_dir)
    Mdyn = LA['Mdyn'].value
    Mlens = LA['Mlens'].value
    # Load LightCone Contents
    LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')
    
    # Find M200 of each Lens-Source system
    indx = [np.where(LC['Halo_ID'] == ii)[0][0] for ii in LA['Halo_ID'].value]
    M200 = LC['M200'][indx]
    
    # Select what to plot
    indx = np.where(M200 > 5e13)[0]
    #Mdyn = Mdyn[indx]    
    #Mlens = Mlens[indx]

    plt.loglog()
    plt.scatter(Mdyn, Mlens, c=colour[sim], label=labels[sim])

plt.plot([1e3, 1e12], [1e3, 1e12], '--k')
plt.xlim(1e6, 1e14)
plt.ylim(1e6, 1e12)
plt.xlabel(r'$M_{dyn} \quad [M_\odot/h$]')
plt.ylabel(r'$M_{lens} \quad [M_\odot/h]$')
plt.legend(loc=2)
plt.savefig('dyn_lens_mass.png', bbox_inches='tight')
plt.clf()
