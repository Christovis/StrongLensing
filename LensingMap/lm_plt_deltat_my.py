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
import readsnap

rc('figure', figsize=(8,6))
rc('font', size=18)
rc('lines', linewidth=3)
rc('axes', linewidth=2)
rc('xtick.major', width=2)
rc('ytick.major', width=2)

###############################################################################
# Load halo lensing properties
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hd_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)
lens_dir = '/cosma5/data/dp004/dc-beck3/LensingMap/'

h = 0.6774
labels = ['FP_GR', 'FP_F6']
colour = ['r', 'b']
###############################################################################

for sim in range(len(sim_dir))[:]:
    print('Analyse lensing map for: ', sim_name[sim])
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    # LightCone file for lens & source properties
    lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    # LensingMap files
    lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
    
    # Load LightCone Contents
    LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

    # LensMaps filenames
    lm_files = [name for name in glob.glob(lm_dir+'LM_L*')]
    
    SnapNM = LC['snapnum']
    A_E = np.zeros(len(lm_files))
    M200 = np.zeros(len(lm_files))
    SNdist = np.zeros(len(lm_files))
    first_lens = 0
    previous_SnapNM = SnapNM[first_lens]
    
    delta_t = []
    delta_mu = []
    for ll in range(len(lm_files)):
        # Load LensingMap Contents
        s = re.findall('[+-]?\d+', lm_files[ll])
        Halo_ID=s[-3]; Src_ID=s[-2]
        indx = np.where(LC['Halo_ID'][:] == int(Halo_ID))[0]
        
        LM = h5py.File(lm_files[ll])
        #centre = LC['HaloPosBox'][ll]
        #zl = LC['Halo_z'][ll]
        M200[ll] = LC['M200'][indx]  #Mvir[ll]
        if LC['M200'][indx] > 1e-50:
            t = LM['delta_t'].value
            indx_max = np.argmax(t)
            #indxs = np.argsort(t)
            t -= t[indx_max]
            t = np.absolute(t[t != 0])
            indxs = np.argsort(t)
            mu = LM['mu'].value
            mu -= mu[indx_max]
            mag = np.absolute(mu[mu != 0])
            for gg in range(len(t)):
                delta_t.append(t[gg])
                delta_mu.append(mag[gg])

    delta_t = np.asarray(delta_t)
    delta_mu = np.asarray(delta_mu)
    #print(delta_t)
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(delta_mu, delta_t)
    # , c=colour[sim]) #, label=labels[sim])#, label=labels[sim])

plt.xlabel(r'$\Delta m_{1x}= m_{1} - m_{x}$')
plt.ylabel(r'$\Delta t \quad [day]$')
plt.legend(loc=1)
plt.savefig('Mu_deltaT.png', bbox_inches='tight')
