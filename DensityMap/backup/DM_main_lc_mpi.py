# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging
from glob import glob
import subprocess
import numpy as np
from sklearn.neighbors import KDTree
import h5py
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
sys.path.insert(0, './lib/')
import process_division as procdiv
import density_maps as dmaps

# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


def whichhalofinder(repo):
    if 'Rockstar' in repo:
        halofinder = 'Rockstar'
    elif 'Subfind' in repo:
        halofinder = 'Subfind'
    else:
        print('InputError: No such Halo Finder')
    return halofinder


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        print(':Registered %d processes' % comm_size)
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["lcdir"]        = sys.argv[3]
        args["ncells"]       = int(sys.argv[4])
        args["outbase"]      = sys.argv[5]
    args = comm.bcast(args)
    label = args["simdir"].split('/')[-2].split('_')[2]
   
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Characteristics
        hflabel = whichhalofinder(args["lcdir"])

        # Load LightCone Contents
        lchdf = h5py.File(args["lcdir"], 'r')
        lcdfhalo = pd.DataFrame(
                {'HF_ID' : lchdf['Halo_Rockstar_ID'].value,
                 'ID' : lchdf['Halo_ID'].value,
                 'Halo_z' : lchdf['Halo_z'].value,
                 'snapnum' : lchdf['snapnum'].value,
                 #'Vrms' : lchdf['VelDisp'].value,
                 #'fov_Mpc' : lchdf['FOV'][:, 1],
                 ('HaloPosBox', 'X') : lchdf['HaloPosBox'][:, 0],
                 ('HaloPosBox', 'Y') : lchdf['HaloPosBox'][:, 1],
                 ('HaloPosBox', 'Z') : lchdf['HaloPosBox'][:, 2],})

        nhalo_per_snapshot = lcdfhalo.groupby('snapnum').count()['HF_ID']

        print('Number of Sub-&Halos in Snapshot:')
        print(nhalo_per_snapshot.values)
        print(np.sum(nhalo_per_snapshot.values))
        print(nhalo_per_snapshot.index.values)
        print(nhalo_per_snapshot.values[0])

        if nhalo_per_snapshot.values[0] > comm_size:
            hist_edges =  procdiv.histedges_equalN(lcdfhalo[('HaloPosBox', 'X')],
                                                   comm_size)
            SH = procdiv.cluster_subhalos(lcdfhalo['ID'].values,
                                          #lcdfhalo['Vrms'],
                                          #lcdfhalo['fov_Mpc'],
                                          lcdfhalo[('HaloPosBox', 'X')].values,
                                          lcdfhalo[('HaloPosBox', 'X')].values,
                                          lcdfhalo[('HaloPosBox', 'Y')].values,
                                          lcdfhalo[('HaloPosBox', 'Z')].values,
                                          hist_edges, comm_size)
            print('dict test', SH.keys())
        elif nhalo_per_snapshot.values[0] < comm_size:
            pass

        # Define Cosmology
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
        cosmosim = {'omega_M_0' : s.header.omega_m,
                    'omega_lambda_0' : s.header.omega_l,
                    'omega_k_0' : 0.0,
                    'h' : s.header.hubble}
        redshift = s.header.redshift
        print(': Redshift: %f' % redshift)

        # Sort Sub-&Halos over Processes
        df = pd.read_csv(args["hfdir"]+'halos_%d.dat' % args["snapnum"],
                         sep='\s+', skiprows=16,
                         usecols=[0, 2, 4, 9, 10, 11],
                         names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        df = df[df['Mvir'] > 5e11]
        sh_id = df['ID'].values.astype('float64')
        sh_vrms = df['Vrms'].values.astype('float64')
        sh_x = df['X'].values.astype('float64')
        sh_y = df['Y'].values.astype('float64')
        sh_z = df['Z'].values.astype('float64')
        del df
        hist_edges =  procdiv.histedges_equalN(sh_x, comm_size)
        SH = cluster_subhalos(sh_id, sh_vrms, sh_x, sh_y, sh_z, hist_edges, comm_size)
      
        # Load simulation
        s = read_hdf5.snapshot(45, args["simdir"])
        s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
               parttype=[0, 1, 4])
        scale = 1e-3*s.header.hubble
        
        # Calculate overlap for particle cuboids
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(np.percentile(sh_vrms, 90)/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        alpha = 6  # multiplied by 4 because of Oguri&Marshall
        overlap = 0.5*alpha*fov_rad*sh_dist  #[Mpc] half of field-of-view
        print('Cuboids overlap is: %f [Mpc]' % overlap)

        # Sort Particles over Processes
        ## Dark Matter
        dm_mass = (s.data['Masses']['dm']).astype('float64')
        dm_x = (s.data['Coordinates']['dm'][:, 0]*scale).astype('float64')
        dm_y = (s.data['Coordinates']['dm'][:, 1]*scale).astype('float64')
        dm_z = (s.data['Coordinates']['dm'][:, 2]*scale).astype('float64')
        dm_mass, dm_x, dm_y, dm_z, dm_split_size_1d, dm_split_disp_1d = cluster_particles(
                dm_mass, dm_x, dm_y, dm_z, hist_edges, comm_size)
        ## Gas
        gas_mass = (s.data['Masses']['gas']).astype('float64')
        gas_x = (s.data['Coordinates']['gas'][:, 0]*scale).astype('float64')
        gas_y = (s.data['Coordinates']['gas'][:, 1]*scale).astype('float64')
        gas_z = (s.data['Coordinates']['gas'][:, 2]*scale).astype('float64')
        gas_mass, gas_x, gas_y, gas_z, gas_split_size_1d, gas_split_disp_1d = cluster_particles(gas_mass, gas_x, gas_y, gas_z, hist_edges, comm_size)
        ## Stars
        star_mass = (s.data['Masses']['stars']).astype('float64')
        star_x = (s.data['Coordinates']['stars'][:, 0]*scale).astype('float64')
        star_y = (s.data['Coordinates']['stars'][:, 1]*scale).astype('float64')
        star_z = (s.data['Coordinates']['stars'][:, 2]*scale).astype('float64')
        star_age = s.data['GFM_StellarFormationTime']['stars']
        star_x = star_x[star_age >= 0]  #[Mpc]
        star_y = star_y[star_age >= 0]  #[Mpc]
        star_z = star_z[star_age >= 0]  #[Mpc]
        star_mass = star_mass[star_age >= 0]
        del star_age
        star_mass, star_x, star_y, star_z, star_split_size_1d, star_split_disp_1d = cluster_particles(star_mass, star_x, star_y, star_z, hist_edges, comm_size)

    else:
        c=None; alpha=None; overlap=None
        cosmosim=None; cosmo=None; redshift=None; hist_edges=None;
        SH=None;
        dm_mass=None; dm_x=None; dm_y=None; dm_z=None
        dm_split_size_1d=None; dm_split_disp_1d=None
        gas_mass=None; gas_x=None; gas_y=None; gas_z=None
        gas_split_size_1d=None; gas_split_disp_1d=None
        star_mass=None; star_x=None; star_y=None; star_z=None
        star_split_size_1d=None; star_split_disp_1d=None
      
    # Broadcast variables over all processors
    sh_split_size_1d = comm.bcast(SH['sh_split_size_1d'], root=0)
    sh_split_disp_1d = comm.bcast(SH['sh_split_disp_1d'], root=0)
    dm_split_size_1d = comm.bcast(dm_split_size_1d, root=0)
    dm_split_disp_1d = comm.bcast(dm_split_disp_1d, root=0)
    gas_split_size_1d = comm.bcast(gas_split_size_1d, root=0)
    gas_split_disp_1d = comm.bcast(gas_split_disp_1d, root=0)
    star_split_size_1d = comm.bcast(star_split_size_1d, root=0)
    star_split_disp_1d = comm.bcast(star_split_disp_1d, root=0)
    c = comm.bcast(c, root=0)
    alpha = comm.bcast(alpha, root=0)
    overlap = comm.bcast(overlap, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    redshift = comm.bcast(redshift, root=0)
    hist_edges = comm.bcast(hist_edges, root=0)

    # Initiliaze variables for each processor
    sh_id_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_vrms_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_x_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_y_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_z_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    dm_mass_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_x_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_y_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_z_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    gas_mass_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_x_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_y_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_z_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    star_mass_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_x_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_y_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_z_local = np.zeros((int(star_split_size_1d[comm_rank])))
    
    # Devide Data over Processes
comm.Scatterv([SH['ID'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_id_local, root=0)
    comm.Scatterv([SH['Vrms'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_vrms_local, root=0)
    comm.Scatterv([SH['X'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_x_local,root=0) 
    comm.Scatterv([SH['Y'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_y_local,root=0) 
    comm.Scatterv([SH['Z'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_z_local,root=0) 
    
    comm.Scatterv([dm_x, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_x_local, root=0) 
    comm.Scatterv([dm_y, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_y_local, root=0) 
    comm.Scatterv([dm_z, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_z_local, root=0) 
    comm.Scatterv([dm_mass, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_mass_local,root=0) 

    comm.Scatterv([gas_x, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_x_local, root=0) 
    comm.Scatterv([gas_y, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_y_local, root=0) 
    comm.Scatterv([gas_z, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_z_local, root=0) 
    comm.Scatterv([gas_mass, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_mass_local,root=0) 

    comm.Scatterv([star_x, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_x_local, root=0)
    comm.Scatterv([star_y, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_y_local, root=0)
    comm.Scatterv([star_z, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_z_local, root=0)
    comm.Scatterv([star_mass, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_mass_local,root=0)
    print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))

    comm.Barrier()

    SH = {"ID"   : sh_id_local,
          "Vrms" : sh_vrms_local,
          "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    DM = {"Mass"  : dm_mass_local,
          "Pos" : np.transpose([dm_x_local, dm_y_local, dm_z_local])}
    Gas = {"Mass"  : gas_mass_local,
           "Pos" : np.transpose([gas_x_local, gas_y_local, gas_z_local])}
    Star = {"Mass"  : star_mass_local,
            "Pos" : np.transpose([star_x_local, star_y_local, star_z_local])}
    
    sigma_tot=[]; subhalo_id=[]; FOV=[]
    ## Run over Sub-&Halos
    for ll in range(len(SH['ID'])):
        #TODO: for z=0 sh_dist=0!!!
        
        # Check cuboid boundary condition,
        # that all surface densities are filled with particles
        if ((SH['Pos'][ll][0]-hist_edges[comm_rank] < overlap) or
                (hist_edges[comm_rank+1]-overlap < \
                 SH['Pos'][ll][0]-hist_edges[comm_rank])):
            if fov_Mpc*0.45 > overlap:
                print("FOV is bigger than cuboids overlap: %f > %f" % \
                        (fov_Mpc*0.45, overlap))
                continue

        smlpixel = 20  # maximum smoothing pixel length
        indx = dmaps.select_particles(Gas['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
        gas_sigma = dmaps.projected_density_pmesh_adaptive(
                Gas['Pos'][indx,:], Gas['Mass'][indx],
                SH['Pos'][ll], #*a/h,
                fov_Mpc,
                args["ncells"],
                hmax=smlpixel,
                particle_type=0)
        indx = dmaps.select_particles(Star['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
        star_sigma = dmaps.projected_density_pmesh_adaptive(
                Star['Pos'][indx,:],Star['Mass'][indx],
                SH['Pos'][ll], #*a/h,
                fov_Mpc,
                args["ncells"],
                hmax=smlpixel,
                particle_type=4)
        indx = dmaps.select_particles(DM['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
        dm_sigma = dmaps.projected_density_pmesh_adaptive(
                DM['Pos'][indx,:], DM['Mass'][indx],
                SH['Pos'][ll],
                fov_Mpc,  #[Mpc]
                args["ncells"],
                hmax=smlpixel,
                particle_type=1)
        sigmatotal = dm_sigma+gas_sigma+star_sigma
       
        # Make sure that density-map if filled
        while 0.0 in sigmatotal:
            smlpixel += 5
            dm_sigma = dmaps.projected_density_pmesh_adaptive(
                    DM['Pos'][indx,:], DM['Mass'][indx],
                    SH['Pos'][ll],
                    fov_Mpc,  #[Mpc]
                    args["ncells"],
                    hmax=smlpixel,
                    particle_type=1)
            sigmatotal = dm_sigma+gas_sigma+star_sigma
        
        sigma_tot.append(sigmatotal)
        subhalo_id.append(int(SH['ID'][ll]))
        FOV.append(fov_Mpc)
    
    fname = args["outbase"]+'z_'+str(args["snapnum"])+'/'+'DM_'+label+'_'+str(comm_rank)+'.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('density_map', data=sigma_tot)
    hf.create_dataset('subhalo_id', data=np.asarray(subhalo_id))
    hf.create_dataset('fov_width', data=np.asarray(FOV))
    #RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    hf.close()


if __name__ == "__main__":
    create_density_maps()

