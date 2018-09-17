# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging
from glob import glob
import subprocess
import numpy as np
#from scipy.ndimage.filters import gaussian_filter
#from astropy.cosmology import LambdaCDM
import pandas as pd
#import dm_funcs as DM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
#import readlensing as rf

# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import parallel_sort as ps
from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


def devide_subhalos(id_in, vrms_in, pos_in, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    #TODO: improve with
    # np.array_split(test,size,axis=0) or
    # [container[_i::count] for _i in range(count)]
    _boundary = [np.min(pos_in[:, 0]), np.max(pos_in[:, 0])*1.01]
    _boundary = np.linspace(_boundary[0], _boundary[1], comm_size+1)
    _inds = np.digitize(pos_in[:, 0], _boundary)
    id_out=[]; vrms_out=[]; pos_out=[]
    #np.array_split(test,size,axis=0)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        id_out.append(id_in[binds])
        vrms_out.append(vrms_in[binds])
        pos_out.append(pos_in[binds, :])
    return id_out, vrms_out, pos_out


def devide_particles(mass_in, pos_in, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    #TODO: improving by merger with devide_subhalos
    _boundary = [np.min(pos_in[:, 0]), np.max(pos_in[:, 0])*1.01]
    _boundary = np.linspace(_boundary[0], _boundary[1], comm_size+1)
    _inds = np.digitize(pos_in[:, 0], _boundary)
    mass_out=[]; pos_out=[]
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        mass_out.append(mass_in[binds])
        pos_out.append(pos_in[binds, :])
    return mass_out, pos_out


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["snapnum"]      = sys.argv[3]
        args["ncells"]       = sys.argv[4]
        args["outbase"]      = sys.argv[5]
    args = comm.bcast(args)
    
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Sort Sub-&Halos over Processes
        df = pd.read_csv(args["hfdir"]+'halos_%s.dat' % args["snapnum"],
                         sep='\s+', skiprows=16,
                         usecols=[0, 2, 4, 9, 10, 11],
                         names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        df = df[df['Mvir'] > 1e11]
        sh_id = df['ID'].values.astype('float64')
        sh_vrms = df['Vrms'].values.astype('float64')
        sh_pos = pd.concat([df['X'], df['Y'], df['Z']], axis=1).values.astype('float64')
        print('There are in total %d Sub-&Halos' % len(df.index))
        del df
       
        split_1d = np.array_split(sh_id, comm_size)
        split_2d = np.array_split(sh_pos, comm_size)
        
        split_size_1d = [len(split_1d[i]) for i in range(len(split_1d))]
        split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1]
        
        split_size_2d = [len(split_2d[i]) for i in range(len(split_2d))]
        split_disp_2d = np.insert(np.cumsum(split_size_2d), 0, 0)[0:-1]

        # Sort Particles over Processes
        #s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
        #s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"], parttype=[1])
        #s.read(["Coordinates", "Masses"], parttype=[1])
        #scale = 1e-3*s.header.hubble
        #dm_pos = s.data['Coordinates']['dm']*scale
        #dm_mass = s.data['Masses']['dm']
        #star_pos = s.data['Coordinates']['stars']
        #gas_pos = s.data['Coordinates']['gas']*scale
        #star_mass = s.data['Masses']['stars']
        #gas_mass = s.data['Masses']['gas']
        #star_age = s.data['GFM_StellarFormationTime']['stars']
        #star_pos = star_pos[star_age >= 0]*scale  #[Mpc]
        #star_mass = star_mass[star_age >= 0]
        #del star_age
        #[dm_mass, dm_pos] = devide_particles(dm_mass, dm_pos, comm_size)
        #[gas_mass, gas_pos] = devide_particles(gas_mass, gas_pos, comm_size)
        #[star_mass, star_pos] = devide_particles(star_mass, star_pos, comm_size)
    else:
        sh_id=None; sh_vrms=None; sh_pos=None
        dm_mass=None; dm_pos=None
        #gas_mass=None; gas_pos=None
        #star_mass=None; star_pos=None

    # Devide Data over Processes
    #sh_pos = comm.Scatterv(sh_pos, root=0)
    #sh_id = comm.Scatterv(sh_id, root=0)
    #sh_vrms = comm.Scatterv(sh_vrms, root=0)
    comm.Scatterv([sh_pos, split_size, split_disp, MPI.DOUBLE],
                  sh_pos_local, root=0)
    comm.Scatterv([sh_pos, split_size, split_disp, MPI.DOUBLE],
                  sh_id_local,root=0)
    
    
    #SH = {'id' : sh_id,
    #      'vrms' : sh_vrms,
    #      'pos' : sh_pos}
    #PA = {'mass' : dm_mass,
    #      'pos' : dm_pos}

    print(': Proc. %d got %d Sub-&Halos' %(comm_rank, len(sh_id)))
    print(': Proc. %d got %d Star' %(comm_rank, len(dm_mass)))

    ## Run over Sub-&Halos
    #for ll in range(len(sh_id)):


if __name__ == "__main__":
    create_density_maps()

#        # Simulation Snapshots
#        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
#        hfdir = hf_dir[sim] + 'halos_%d.dat' % snapnum
#        
#        # Units of Simulation
#        #scale = rf.simulation_units(sim_dir[sim])
#        scale = 1
#        
#        # Cosmological Parameters
#        snap_tot_num = 45
#        s = read_hdf5.snapshot(snap_tot_num, sim_dir[sim])
#        cosmo = LambdaCDM(H0=s.header.hubble*100,
#                          Om0=s.header.omega_m,
#                          Ode0=s.header.omega_l)
#
#        # Load Sub-/Halo Data
#        data = pd.read_csv(hfdir, sep='\s+', skiprows=16,
#                           usecols=[0, 2, 4, 9, 10, 11],
#                           names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
#        indx = np.where(data['Mvir'] > 1e11)[0]
#        sh_id = data['ID'][indx]
#        sh_vrms = data['Vrms'][indx]
#        sh_pos = pd.concat([data['X'][indx],
#                            data['Y'][indx],
#                            data['Z'][indx]], axis=1).values*scale
#        del data['Mvir'], indx
#
#        ## Load Particle Data
#        #s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"], parttype=[0, 1, 4])
#        #star_pos = s.data['Coordinates']['stars']
#        #gas_pos = s.data['Coordinates']['gas']*scale
#        #dm_pos = s.data['Coordinates']['dm']*scale
#        #star_mass = s.data['Masses']['stars']
#        #gas_mass = s.data['Masses']['gas']
#        #dm_mass = s.data['Masses']['dm']
#        #star_age = s.data['GFM_StellarFormationTime']['stars']
#        #star_pos = star_pos[star_age >= 0]*scale  #[Mpc]
#        #star_mass = star_mass[star_age >= 0]
#        #del star_age
#
#        # Devide Halos over CPUs
#        lenses_per_cpu = DM.devide_halos(len(sh_pos),
#                                         CPUs, 'equal')
#        # Prepatre Processes to be run in parallel
#        jobs = []
#        manager = multiprocessing.Manager()
#        
#        lenses_per_cpu = [lenses_per_cpu[cc] for cc in range(CPUs)]
#        for cpu in range(CPUs):
#            p = Process(target=DM.generate_lens_map, name='Proc_%d'%cpu,
#                        args=(s,
#                              sh_id[lenses_per_cpu[cpu]],
#                              sh_vrms[lenses_per_cpu[cpu]],
#                              sh_pos[lenses_per_cpu[cpu], :],
#                              cpu, s.header.redshift, scale, Ncells,
#                              HQ_dir, sim, sim_phy, sim_name, hf_name, cosmo))
#            jobs.append(p)
#            p.start()
#
#        # Run Processes in parallel
#        # Wait until every job is completed
#        for p in jobs:
#            p.join()
