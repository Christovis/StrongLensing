# Python implementation using the multiprocessing module
#
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
import pandas as pd
import lm_funcs_mp as LI
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/')
import readsnap
import readlensing as rf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())

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
LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, dd, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

# Node = 16 CPUs
CPUs = 2  # Number of CPUs to use
print('------------>', CPUs)
###########################################################################
# Define Lensing Map Parameters
# lensing map parameters
Ncells = 1024  # devide density map into cells
save_maps = True

###########################################################################

# protect the 'entry point' for Windows OS
if __name__ == '__main__':
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    os.system("taskset -p 0xff %d" % os.getpid())

    # Run through simulations
    for sim in range(len(sim_dir)):
        print(sim_name[sim])
        #logging.info('Create lensing map for: %s', sim_name[sim])
        # File for lens & source properties
        lc_file = lc_dir[sim]+hf_name+'/LC_SN_'+sim_name[sim]+'_rndseed31.h5'
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        
        # Units of Simulation
        #scale = rf.simulation_units(sim_dir[sim])
        scale = 1e-3
        
        # Cosmological Parameters
        snap_tot_num = 45
        header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
        cosmo = LambdaCDM(H0=header.hubble*100,
                          Om0=header.omega_m,
                          Ode0=header.omega_l)
        h = header.hubble
        a = 1/(1 + header.redshift)

        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, hf_name)

        # Sort Lenses according to Snapshot Number (snapnum)
        indx = np.argsort(LC['snapnum'])
        if hf_name == 'Subfind':
            Halo_HF_ID= LC['Halo_Subfind_ID'][indx]
            Rad = LC['Rvir'][indx]
        elif hf_name == 'Rockstar':
            Halo_HF_ID = LC['Halo_Rockstar_ID'][indx]
            #Rad = LC['Rvir'][indx]
            fov_Mpc = LC['FOV'][1, indx]*1.5
        Halo_ID= LC['Halo_ID'][indx]
        Halo_z = LC['Halo_z'][indx]
        snapnum = LC['snapnum'][indx]
        HaloPosBox = LC['HaloPosBox'][indx]
        #ratio = [Rad[iii]/fov_Mpc[iii] for iii in range(len(Rad))]
        #print('compare Rvir & fov_Mpx', np.mean(ratio))

        # Devide Halos over CPUs
        lenses_per_cpu = LI.devide_halos(len(Halo_ID), CPUs, 'equal')
        # Prepatre Processes to be run in parallel
        jobs = []
        manager = multiprocessing.Manager()
        results_per_cpu = manager.dict()
        
        print('Total number of halos:', len(Halo_ID), lenses_per_cpu[-1][-1])
        print('Halos per CPU: %s', [str(len(lpc)) for lpc in lenses_per_cpu])
        lenses_per_cpu = [lenses_per_cpu[cc] for cc in range(CPUs)]
        #lenses_per_cpu = [lenses_per_cpu[cc][:100] for cc in range(CPUs)]
        for cpu in range(CPUs):
            p = Process(target=LI.generate_lens_map, name='Proc_%d'%cpu,
                        args=(lenses_per_cpu[cpu], cpu, LC, Halo_HF_ID,
                              Halo_ID, Halo_z, fov_Mpc, snapnum, snapfile,
                              h, scale, Ncells, HQ_dir, sim, sim_phy, sim_name,
                              hf_name, HaloPosBox, cosmo, results_per_cpu))
            jobs.append(p)
            p.start()

        ###############################
        # New Multiprocessing
        #pool = multiprocessing.Pool(processes=CPUs)
        #pool.map(LI.generate_lens_map, [os.path.join(inDir, i) for i in lenses_per_cpu])
        #pool.close()
        #pool.join()

        # Run Processes in parallel
        # Wait until every job is completed
        print('started all jobs')
        for p in jobs:
            p.join()
