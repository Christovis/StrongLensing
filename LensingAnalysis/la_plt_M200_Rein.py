from __future__ import division
import os, os.path
import glob
import sys
import re
import numpy as np
import astropy
from matplotlib import pyplot as plt, rc
from astropy import units as u
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

# Wiesner et al. 2012

data = np.array([
[0.4085542272682985, 5.868059679716307],
[0.6092460701552084, 11.205276069720082],
[0.9876757454615227, 9.427695086479087],
[1.1707504472048231, 8.16126967775384],
[1.4165337597871968, 11.882962621298237],
[1.4771308712078253, 7.918144697414032],
[1.8647450719246381, 8.507827334098092],
[2.133794961742011, 7.317912836535072],
[7.137645346593659, 7.195543429173515],
[17.71825537715099, 18.534180719994087]]).T
h = 0.6774

LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hd_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)
lens_dir = '/cosma5/data/dp004/dc-beck3/LensingMap/'

h = 0.6774
labels = ['FP_GR', 'FP_F6']
colour = ['r', 'b']

for sim in range(len(sim_dir))[:]:
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
            M200[ll] = LC['M200'][indx]  #Mvir[ll]
            A_E[ll] = LM['eqv_einstein_radius'].value  #[arcsec]
            
            t = LM['delta_t'].value
            indx_max = np.argmax(t)
            t -= t[indx_max]
            t = np.absolute(t[t != 0])
            for tt in t:
                delta_t.append(tt)
            
            mu = LM['mu'].value
            mu -= mu[indx_max]
            mag = np.absolute(mu[mu != 0])
            for mm in mag:
                delta_mu.append(mm)
            
        if A_E[ll] > 10:
            print('problem', ll, A_E[ll])
        #R_E[ll] = ((A_E*u.arcsec).to_value('rad') * \
        #            Planck15.angular_diameter_distance(zl)).to_value('Mpc')

    delta_t = np.asarray(delta_t)
    delta_mu = np.asarray(delta_mu)
    #plt.scatter(data[0]*1e14/h, data[1], c='k', label='Wiesner et al. 2012', s=20)
    plt.xscale("log")
    plt.scatter(M200, A_E, c=colour[sim])#, label=labels[sim])
plt.xlabel(r'$M_{vir} \quad [M_\odot/h$]')
plt.ylabel(r'$\theta_{E} \quad [arcsec]$')
plt.legend(loc=2)
plt.show()
