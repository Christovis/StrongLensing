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
import dm_funcs as DM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import readsnap
import read_hdf5
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
sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

# Node = 16 CPUs
CPUs = 1  # Number of CPUs to use
snapnum = 40
###########################################################################
# Define Lensing Map Parameters
Ncells = 1024  # devide density map into cells

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
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        hfdir = hf_dir[sim] + 'halos_%d.dat' % snapnum
        
        # Cosmological Parameters
        s = read_hdf5.snapshot(snapnum, sim_dir[sim])
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)

        # Load Sub-/Halo Data
        data = pd.read_csv(hfdir, sep='\s+', skiprows=16,
                           usecols=[0, 2, 4, 9, 10, 11],
                           names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        indx = np.where(data['Mvir'] > 1e11)[0]
        sh_id = data['ID'].values[indx]
        sh_vrms = data['Vrms'].values[indx]
        sh_pos = pd.concat([data['X'][indx],
                            data['Y'][indx],
                            data['Z'][indx]], axis=1).values  # [Mpc]
        del data['Mvir'], indx
        
        # Units of Simulation
        #scale = rf.simulation_units(sim_dir[sim])
        scale = 1e-3*s.header.hubble

        ## Load Particle Data
        #s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"], parttype=[0, 1, 4])
        #star_pos = s.data['Coordinates']['stars']
        #gas_pos = s.data['Coordinates']['gas']*scale
        #dm_pos = s.data['Coordinates']['dm']*scale
        #star_mass = s.data['Masses']['stars']
        #gas_mass = s.data['Masses']['gas']
        #dm_mass = s.data['Masses']['dm']
        #star_age = s.data['GFM_StellarFormationTime']['stars']
        #star_pos = star_pos[star_age >= 0]*scale  #[Mpc]
        #star_mass = star_mass[star_age >= 0]
        #del star_age

        # Devide Halos over CPUs
        lenses_per_cpu = DM.devide_halos(len(sh_id),
                                         CPUs, 'equal')
        # Prepatre Processes to be run in parallel
        jobs = []
        manager = multiprocessing.Manager()
        
        print('Total number of halos: %d' % len(sh_pos))
        #print('Halos per CPU: %s', [str(len(lpc)) for lpc in lenses_per_cpu])  # 100.000
        lenses_per_cpu = [lenses_per_cpu[cc] for cc in range(CPUs)]
        for cpu in range(CPUs):
            p = Process(target=DM.generate_lens_map, name='Proc_%d'%cpu,
                        args=(s,
                              sh_id[lenses_per_cpu[cpu]],
                              sh_vrms[lenses_per_cpu[cpu]],
                              sh_pos[lenses_per_cpu[cpu], :],
                              cpu, s.header.redshift, scale, Ncells,
                              HQ_dir, sim, sim_phy, sim_name, hf_name, cosmo))
            jobs.append(p)
            p.start()

        # Run Processes in parallel
        # Wait until every job is completed
        print('started all jobs')
        for p in jobs:
            p.join()
