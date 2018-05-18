from __future__ import division
import os
import sys
import logging
import scipy
import numpy as np
import multiprocessing
from multiprocessing import Process
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy.cosmology import LambdaCDM
import h5py
import lm_tools as LI
sys.path.insert(0, '..')
import readsnap
import readlensing as rf

# Works only with Python 2.7.~
print("Python version: ", sys.version)
print("Numpy version: ", np.version.version)
print("Scipy version: ", scipy.version.version)
print("Number of CPUs: ", multiprocessing.cpu_count())

############################################################################
# Set up logging and parse arguments
#logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                    level=logging.DEBUG, datefmt='%H:%M:%S')

############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, dd, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

# Node = 16 CPUs
CPUs = 6  # Number of CPUs to use

###########################################################################
# Define Lensing Map Parameters
# lensing map parameters
xi0 = 0.001  #[Mpc], convert length scale
Ncells = 1024  # devide density map into cells
Lrays = 2.0*u.Mpc  # Length of, to calculate alpha from kappa
Nrays = 1024  # Number of, to calculate alpha from kappa
save_maps = True

###########################################################################
# protect the 'entry point' for Windows OS

if __name__ == '__main__':
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    os.system("taskset -p 0xff %d" % os.getpid())

    # Run through simulations
    for sim in range(len(sim_dir)):
        #logging.info('Create lensing map for: %s', sim_name[sim])
        # File for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        
        # Units of Simulation
        scale = rf.simulation_units(sim_dir[sim])
        # scale = 1e-3
        
        # Cosmological Parameters
        snap_tot_num = 45
        header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
        cosmo = LambdaCDM(H0=header.hubble*100,
                          Om0=header.omega_m,
                          Ode0=header.omega_l)
        h = header.hubble
        a = 1/(1 + header.redshift)

        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')
        Src_ID = LC['Src_ID']
        Src_z = LC['Src_z']

        # Sort Lenses according to Snapshot Number (snapnum)
        indx = np.argsort(LC['snapnum'])
        Halo_ID= LC['Halo_ID'][indx]
        Halo_z = LC['Halo_z'][indx]
        M200 = LC['M200'][indx]
        Rvir = LC['Rvir'][indx]
        snapnum = LC['snapnum'][indx]
        HaloPosBox = LC['HaloPosBox'][indx]

        # Devide Halos over CPUs
        lenses_per_cpu = LI.devide_halos(len(Halo_ID), CPUs)
        # Prepatre Processes to be run in parallel
        proc = []
        for cpu in range(CPUs)[1:]:
            Process(target=LI.generate_lens_map, name='Proc_%d'%cpu,
                    args=(lenses_per_cpu[cpu], LC, Halo_ID, Halo_z, Rvir,
                          snapnum, snapfile, h, scale, Ncells, Nrays,
                          Lrays, HQ_dir, sim, sim_phy, sim_name,
                          HaloPosBox, xi0, cosmo)).start()
        # Run Processes in parallel
        # Wait until every job is completed
        #for p in proc:
        #    p.join()
