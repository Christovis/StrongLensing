# File Description:
#   Creates density maps of sub- & halos in snapshot.
#   Selection criteria for types of sub- & halos is
#   defined in SubHalos.py
#
from __future__ import division
import os, sys, logging
from glob import glob
import numpy as np
from sklearn.neighbors import KDTree
import h5py
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
sys.path.insert(0, './lib/')
#import testdensitymap as tmap
import process_division as procdiv
import density_maps as dmaps
from SubHalos import subhalo_data
from SubHalos import particle_data

# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        print(':Registered %d processes' % comm_size)
        args["simdir"]       = sys.argv[1]
        args["hfname"]       = sys.argv[2]
        args["hfdir"]        = sys.argv[3]
        args["snapnum"]      = int(sys.argv[4])
        args["ncells"]       = int(sys.argv[5])
        args["smlpixel"]       = int(sys.argv[6])
        args["outbase"]      = sys.argv[7]
    args = comm.bcast(args)
    label = args["simdir"].split('/')[-2].split('_')[2]
   
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Load simulation
        s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
        s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
               parttype=[0, 1, 4, 5])
       
        unitlength = dmaps.define_unit(s.header.unitlength)
        # Define Cosmology
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
        redshift = s.header.redshift
        print(': Redshift: %f' % redshift)
        
        # Sort Sub-&Halos over Processes
        SH = subhalo_data(args["hfdir"], args["hfname"], args["snapnum"],
                          s.header.hubble, s.header.unitlength)
        hist_edges =  procdiv.histedges_equalN(SH['X'], comm_size)
        SH = procdiv.cluster_subhalos_box(SH, hist_edges, comm_size)
      
        # Calculate overlap for particle cuboids
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(np.percentile(SH['Vrms'], 90)/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value(unitlength)
        alpha = 2  # multiplied by 4 because of Oguri&Marshall
        overlap = 0.5*alpha*fov_rad*sh_dist  # half of field-of-view
        print('Cuboids overlap is: %f [%s]' % (overlap, unitlength))

        # Sort Particles over Processes
        DM, Gas, Star, BH = particle_data(s.data, s.header.hubble, unitlength)
        DM = procdiv.cluster_particles(DM, hist_edges, comm_size)
        Gas = procdiv.cluster_particles(Gas, hist_edges, comm_size)
        Star = procdiv.cluster_particles(Star, hist_edges, comm_size)
        BH = procdiv.cluster_particles(BH, hist_edges, comm_size)
        
    else:
        c=None; alpha=None; overlap=None; unitlength=None
        cosmo=None; redshift=None; hist_edges=None
        SH = {'ID':None, 'Vrms':None, 'X':None, 'Y':None, 'Z':None,
              'split_size_1d':None, 'split_disp_1d':None}
        DM = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
              'split_size_1d':None, 'split_disp_1d':None}
        Gas = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
               'split_size_1d':None, 'split_disp_1d':None}
        Star = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
                'split_size_1d':None, 'split_disp_1d':None}
        BH = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
              'split_size_1d':None, 'split_disp_1d':None}
      
    # Broadcast variables over all processors
    sh_split_size_1d = comm.bcast(SH['split_size_1d'], root=0)
    sh_split_disp_1d = comm.bcast(SH['split_disp_1d'], root=0)
    dm_split_size_1d = comm.bcast(DM['split_size_1d'], root=0)
    dm_split_disp_1d = comm.bcast(DM['split_disp_1d'], root=0)
    gas_split_size_1d = comm.bcast(Gas['split_size_1d'], root=0)
    gas_split_disp_1d = comm.bcast(Gas['split_disp_1d'], root=0)
    star_split_size_1d = comm.bcast(Star['split_size_1d'], root=0)
    star_split_disp_1d = comm.bcast(Star['split_disp_1d'], root=0)
    bh_split_size_1d = comm.bcast(BH['split_size_1d'], root=0)
    bh_split_disp_1d = comm.bcast(BH['split_disp_1d'], root=0)
    c = comm.bcast(c, root=0)
    unitlength = comm.bcast(unitlength, root=0)
    alpha = comm.bcast(alpha, root=0)
    overlap = comm.bcast(overlap, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    redshift = comm.bcast(redshift, root=0)
    hist_edges = comm.bcast(hist_edges, root=0)

    SH = procdiv.scatter_subhalos(SH, sh_split_size_1d,
                                  comm_rank, comm, root_proc=0)
    DM = procdiv.scatter_particles(DM, dm_split_size_1d,
                                   comm_rank, comm, root_proc=0)
    Gas = procdiv.scatter_particles(Gas, gas_split_size_1d,
                                    comm_rank, comm, root_proc=0)
    Star = procdiv.scatter_particles(Star, star_split_size_1d,
                                     comm_rank, comm, root_proc=0)
    BH = procdiv.scatter_particles(BH, star_split_size_1d,
                                   comm_rank, comm, root_proc=0)
    
    print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))
    
    sigma_tot=[]; subhalo_id=[]; FOV=[]
    ## Run over Sub-&Halos
    for ll in range(len(SH['ID'])):
        # Define field-of-view edge-length
        fov_rad = 4*np.pi*(SH['Vrms'][ll]/c)**2
        #TODO: for z=0 sh_dist=0!!!
        sh_dist = (cosmo.comoving_distance(redshift)).to_value(unitlength)
        alpha = 1.4
        fov = alpha*fov_rad*sh_dist  #[kpc] edge-length of box
        
        # Check cuboid boundary condition,
        # that all surface densities are filled with particles
        if ((SH['Pos'][ll][0]-hist_edges[comm_rank] < overlap) or
                (hist_edges[comm_rank+1]-overlap < \
                 SH['Pos'][ll][0]-hist_edges[comm_rank])):
            if fov*0.45 > overlap:
                print("FOV is bigger than cuboids overlap: %f > %f" % \
                        (fov*0.45, overlap))
                continue

        ## BH
        pos, indx = dmaps.select_particles(
                BH['Pos'], SH['Pos'][ll], #*a/h,
                fov, 'box')
        bh_sigma = dmaps.projected_density_pmesh(
                pos, BH['Mass'][indx], fov, args["ncells"])

        ## Gas
        pos, indx = dmaps.select_particles(
                Gas['Pos'], SH['Pos'][ll], #*a/h,
                fov, 'box')
        gas_sigma = dmaps.projected_density_pmesh_adaptive(
                pos, Gas['Mass'][indx], fov,  args["ncells"],
                hmax=args["smlpixel"])
        ## Star
        pos, indx = dmaps.select_particles(
                Star['Pos'], SH['Pos'][ll], #*a/h,
                fov, 'box')
        star_sigma = dmaps.projected_density_pmesh_adaptive(
                pos, Star['Mass'][indx], fov, args["ncells"],
                hmax=args["smlpixel"])
        ## DM
        pos, indx = dmaps.select_particles(
                DM['Pos'], SH['Pos'][ll], #*a/h,
                fov, 'box')
        dm_sigma = dmaps.projected_density_pmesh_adaptive(
                pos, DM['Mass'][indx], fov, args["ncells"],
                hmax=args["smlpixel"])
        sigmatotal = dm_sigma+gas_sigma+star_sigma
       
        # Make sure that density-map if filled
        extention = 0
        while (0.0 in sigmatotal) and (extention < 60):
            extention += 5
            dm_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, DM['Mass'][indx], fov, args["ncells"],
                    hmax=args["smlpixel"]+extention)
            sigmatotal = dm_sigma+gas_sigma+star_sigma

        #tmap.plotting(sigmatotal, args["ncells"], fov, 0.57)
        sigma_tot.append(sigmatotal)
        subhalo_id.append(int(SH['ID'][ll]))
        FOV.append(fov)
    
    fname = args["outbase"]+'z_'+str(args["snapnum"])+'/'+'DM_'+label+'_'+str(comm_rank)+'.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('DMAP', data=sigma_tot)              # density map in unit of simulation
    hf.create_dataset('HFID', data=np.asarray(subhalo_id)) # Rockstar sub-&halo id
    hf.create_dataset('FOV', data=np.asarray(FOV))         # field-of-view in units #[kpc, Mpc]
    #RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    hf.close()


if __name__ == "__main__":
    create_density_maps()

