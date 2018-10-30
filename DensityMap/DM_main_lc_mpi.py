# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging, time
from glob import glob
import subprocess
import numpy as np
import h5py
import pandas as pd
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
sys.path.insert(0, './test/')
import testdensitymap as tmap

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
    time_start = time.time()
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["lcdir"]        = sys.argv[3]
        args["ncells"]       = int(sys.argv[4])
        args["walltime"]       = int(sys.argv[5])
        args["outbase"]      = sys.argv[6]
    args = comm.bcast(args, root=0)
    label = args["simdir"].split('/')[-2].split('_')[2]
    hflabel = whichhalofinder(args["lcdir"])

    # Load LightCone Contents
    if comm_rank == 0:
        lchdf = h5py.File(args["lcdir"], 'r')
        dfhalo = pd.DataFrame(
                {'HF_ID' : lchdf['Halo_Rockstar_ID'].value,
                 'ID' : lchdf['Halo_ID'].value,
                 'Halo_z' : lchdf['Halo_z'].value,
                 'snapnum' : lchdf['snapnum'].value,
                 'Vrms' : lchdf['VelDisp'].value,
                 'fov_Mpc' : lchdf['FOV'][:][1],
                 ('HaloPosBox', 'X') : lchdf['HaloPosBox'][:, 0],
                 ('HaloPosBox', 'Y') : lchdf['HaloPosBox'][:, 1],
                 ('HaloPosBox', 'Z') : lchdf['HaloPosBox'][:, 2]})
        nhalo_per_snapshot = dfhalo.groupby('snapnum').count()['HF_ID']
        snapshots = dfhalo.groupby('snapnum').count().index.values
        dfhalo = dfhalo.sort_values(by=['snapnum'])
    else:
        nhalo_per_snapshot=None
    nhalo_per_snapshot = comm.bcast(nhalo_per_snapshot, root=0)

    sigma_tot=[]; out_hfid=[]; out_lcid=[]; out_redshift=[];
    out_snapshot=[]; out_vrms=[]; out_fov=[]
    ## Run over Snapshots
    for ss in range(len(nhalo_per_snapshot))[-2:]:
        print('Snapshot %d of %d' % (ss, len(nhalo_per_snapshot)))
        
        if comm_rank == 0:
            dfhalosnap = dfhalo.loc[dfhalo['snapnum'] == snapshots[ss]]
            # Load simulation
            s = read_hdf5.snapshot(snapshots[ss], args["simdir"])
            s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
                   parttype=[0, 1, 4, 5])
            scale = 1e-3*s.header.hubble
            cosmo = LambdaCDM(H0=s.header.hubble*100,
                              Om0=s.header.omega_m,
                              Ode0=s.header.omega_l)
            print(': Redshift: %f' % s.header.redshift)
           
            sh_hfid= dfhalosnap['HF_ID'].values
            sh_id = dfhalosnap['ID'].values
            sh_red = dfhalosnap['Halo_z'].values
            sh_snap = dfhalosnap['snapnum'].values
            sh_vrms = dfhalosnap['Vrms'].values
            sh_fov = dfhalosnap['fov_Mpc'].values
            sh_x = dfhalosnap[('HaloPosBox', 'X')].values
            sh_y = dfhalosnap[('HaloPosBox', 'Y')].values
            sh_z = dfhalosnap[('HaloPosBox', 'Z')].values
            hist_edges =  procdiv.histedges_equalN(sh_x, comm_size)
            SH = procdiv.cluster_subhalos_lc(sh_hfid, sh_id, sh_red, sh_snap,
                                             sh_vrms, sh_fov, sh_x, sh_y, sh_z,
                                             hist_edges, comm_size)
            
            ## Dark Matter
            DM = {'Mass' : (s.data['Masses']['dm']).astype('float64'),
                  'Pos' : (s.data['Coordinates']['dm']*scale).astype('float64')}
            ## Gas
            Gas = {'Mass' : (s.data['Masses']['gas']).astype('float64'),
                   'Pos' : (s.data['Coordinates']['gas']*scale).astype('float64')}
            ## Stars
            age = (s.data['GFM_StellarFormationTime']['stars']).astype('float64')
            Star = {'Mass' : (s.data['Masses']['stars'][age >= 0]).astype('float64'),
                    'Pos' : (s.data['Coordinates']['stars'][age >= 0, :]*scale).astype('float64')}
            ## BH
            BH = {'Mass' : (s.data['Masses']['bh']).astype('float64'),
                  'Pos' : (s.data['Coordinates']['bh']*scale).astype('float64')}
            
            
            # Calculate overlap for particle cuboids
            c = (const.c).to_value('km/s')
            fov_rad = 4*np.pi*(np.percentile(SH['Vrms'], 90)/c)**2
            sh_dist = (cosmo.comoving_distance(s.header.redshift)).to_value('Mpc')
            alpha = 6  # multiplied by 4 because of Oguri&Marshall
            overlap = 0.5*alpha*fov_rad*sh_dist  #[Mpc] half of field-of-view
            print('Cuboids overlap is: %f [Mpc]' % overlap)
            
            DM = procdiv.cluster_particles(DM, hist_edges, comm_size)
            Gas = procdiv.cluster_particles(Gas, hist_edges, comm_size)
            Star = procdiv.cluster_particles(Star, hist_edges, comm_size)
            BH = procdiv.cluster_particles(BH, hist_edges, comm_size)
        else:
            overlap=None; hist_edges=None
            SH = {'HF_ID':None, 'ID':None, 'redshift':None, 'snapshot':None,
                  'Vrms':None, 'fov_Mpc':None, 'X':None, 'Y':None, 'Z':None,
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
        overlap = comm.bcast(overlap, root=0)
        hist_edges = comm.bcast(hist_edges, root=0)
        sh_split_size_1d = comm.bcast(SH['split_size_1d'], root=0)
        dm_split_size_1d = comm.bcast(DM['split_size_1d'], root=0)
        gas_split_size_1d = comm.bcast(Gas['split_size_1d'], root=0)
        star_split_size_1d = comm.bcast(Star['split_size_1d'], root=0)
        bh_split_size_1d = comm.bcast(BH['split_size_1d'], root=0)

        SH = procdiv.scatter_subhalos_lc(SH, sh_split_size_1d,
                                         comm_rank, comm, root_proc=0)
        DM = procdiv.scatter_particles(DM, dm_split_size_1d,
                comm_rank, comm, root_proc=0)
        Gas = procdiv.scatter_particles(Gas, gas_split_size_1d,
                comm_rank, comm, root_proc=0)
        Star = procdiv.scatter_particles(Star, star_split_size_1d,
                comm_rank, comm, root_proc=0)
        BH = procdiv.scatter_particles(BH, bh_split_size_1d,
                comm_rank, comm, root_proc=0)

        print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))

        ## Run over Sub-&Halos
        for ll in range(len(SH['ID'])):
            print('Lens %d' % (ll))
            #TODO: for z=0 sh_dist=0!!!
            
            smlpixel = 20  # maximum smoothing pixel length
            ## BH
            pos, indx = dmaps.select_particles(
                    BH['Pos'], SH['Pos'][ll], #*a/h,
                    SH['fov_Mpc'][ll], 'box')
            bh_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, BH['Mass'][indx],
                    SH['fov_Mpc'][ll],
                    args["ncells"],
                    hmax=smlpixel)
            ## Star
            pos, indx = dmaps.select_particles(
                    Star['Pos'], SH['Pos'][ll], #*a/h,
                    SH['fov_Mpc'][ll], 'box')
            star_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, Star['Mass'][indx],
                    SH['fov_Mpc'][ll],
                    args["ncells"],
                    hmax=smlpixel)
            ## Gas
            pos, indx = dmaps.select_particles(
                    Gas['Pos'], SH['Pos'][ll],  #*a/h
                    SH['fov_Mpc'][ll], 'box')
            gas_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, Gas['Mass'][indx],
                    SH['fov_Mpc'][ll],
                    args["ncells"],
                    hmax=smlpixel)
            ## DM
            pos, indx = dmaps.select_particles(
                    DM['Pos'], SH['Pos'][ll],  #*a/h
                    SH['fov_Mpc'][ll],  'box')
            dm_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, DM['Mass'][indx],
                    SH['fov_Mpc'][ll],  #[Mpc]
                    args["ncells"],
                    hmax=smlpixel)
            sigmatotal = dm_sigma+gas_sigma+star_sigma+bh_sigma


            # Make sure that density-map if filled
            while 0.0 in sigmatotal:
                smlpixel += 5
                dm_sigma = dmaps.projected_density_pmesh_adaptive(
                        pos, DM['Mass'][indx],
                        SH['fov_Mpc'][ll],  #[Mpc]
                        args["ncells"],
                        hmax=smlpixel)
                sigmatotal = dm_sigma+gas_sigma+star_sigma+bh_sigma
            
            #tmap.plotting(sigmatotal, args["ncells"],
            #              SH['fov_Mpc'][ll], SH['redshift'][ll])
            sigma_tot.append(sigmatotal)
            out_hfid.append(SH['HF_ID'][ll])
            out_lcid.append(SH['ID'][ll])
            out_redshift.append(SH['redshift'][ll])
            out_snapshot.append(SH['snapshot'][ll])
            out_vrms.append(SH['Vrms'][ll])
            out_fov.append(SH['fov_Mpc'][ll])
            if args["walltime"] - (time_start - time.time())/(60*60) < 0.25:
                fname = args["outbase"]+'DM_'+label+'_lc.h5'
                hf = h5py.File(fname, 'w')
                hf.create_dataset('density_map', data=sigma_tot)
                hf.create_dataset('HF_ID', data=np.asarray(out_hfid))
                hf.create_dataset('LC_ID', data=np.asarray(out_lcid))
                hf.create_dataset('redshift', data=np.asarray(out_redshift))
                hf.create_dataset('snapshot', data=np.asarray(out_snapshot))
                hf.create_dataset('Vrms', data=np.asarray(out_vrms))
                hf.create_dataset('fov_Mpc', data=np.asarray(out_fov))
                hf.close()
   
    # Gather the Results
    #comm.Barrier()
    #comm.Gather(out_hfid, [rootout_hfid,split_sizes,displacements,MPI.DOUBLE], root=0)

    fname = args["outbase"]+'DM_'+label+'_lc.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('density_map', data=sigma_tot)
    hf.create_dataset('HF_ID', data=np.asarray(out_hfid))
    hf.create_dataset('LC_ID', data=np.asarray(out_lcid))
    hf.create_dataset('redshift', data=np.asarray(out_redshift))
    hf.create_dataset('snapshot', data=np.asarray(out_snapshot))
    hf.create_dataset('Vrms', data=np.asarray(out_vrms))
    hf.create_dataset('fov_Mpc', data=np.asarray(out_fov))
    #RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    hf.close()


if __name__ == "__main__":
    create_density_maps()

