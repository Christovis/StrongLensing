# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging, time
from glob import glob
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
from SubHalos import subhalo_data
from SubHalos import particle_data

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


def create_density_maps():
    time_start = time.time()
    # Get command line arguments
    args = {}
    args["simdir"]       = sys.argv[1]
    args["lcdir"]        = sys.argv[2]
    args["ncells"]       = int(sys.argv[3])
    args["sml"]          = int(sys.argv[4])
    args["walltime"]     = int(sys.argv[5])
    args["outbase"]      = sys.argv[6]
    label = args["simdir"].split('/')[-2].split('_')[2]
    lclabel = args["lcdir"].split('/')[-1][-4]
    # Characteristics
    hflabel = whichhalofinder(args["lcdir"])

    # Load LightCone Contents
    lchdf = h5py.File(args["lcdir"], 'r')
    dfhalo = pd.DataFrame(
            {'HF_ID' : lchdf['HF_ID'].value,
             'LC_ID' : lchdf['LC_ID'].value,
             'Halo_z' : lchdf['Halo_z'].value,
             'snapnum' : lchdf['snapnum'].value,
             'Vrms' : lchdf['VelDisp'].value,
             'fov_Mpc' : lchdf['FOV'][:][1],
             ('HaloPosBox', 'X') : lchdf['HaloPosBox'][:, 0],
             ('HaloPosBox', 'Y') : lchdf['HaloPosBox'][:, 1],
             ('HaloPosBox', 'Z') : lchdf['HaloPosBox'][:, 2]})
    
    if len(dfhalo.index.values) > 2000:
        # Limit number of halos, to keep comp. cost down
        dfhalo = dfhalo.sample(n=2000)
    print('There are %d galaxies in this lightcone' % len(dfhalo.index.values))
    
    nhalo_per_snapshot = dfhalo.groupby('snapnum').count()['HF_ID']
    print('devided over lightcone as:')
    print(nhalo_per_snapshot)
    snapshots = dfhalo.groupby('snapnum').count().index.values
    dfhalo = dfhalo.sort_values(by=['snapnum'])

    sigma_tot=[]; out_hfid=[]; out_lcid=[]; out_redshift=[]; out_snapnum=[]
    out_vrms=[]; out_fov=[]

    ## Run over Snapshots
    for ss in range(len(nhalo_per_snapshot)):
        print('Snapshot %d of %d' % (ss, len(nhalo_per_snapshot)))
        dfhalosnap = dfhalo.loc[dfhalo['snapnum'] == snapshots[ss]]
        
        # Load simulation
        s = read_hdf5.snapshot(snapshots[ss], args["simdir"])
        s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
               parttype=[0, 1, 4, 5])
        scale = 1e-3*s.header.hubble
        print(': Redshift: %f' % s.header.redshift)
        
        DM, Gas, Star, BH = particle_data(s.data, h, 'kpc')
        ## Run over Sub-&Halos
        for ll in range(len(dfhalosnap.index)):
            print('Lens %d of %d' % (ll, len(dfhalosnap.index)))
            #TODO: for z=0 sh_dist=0!!!
            
            # Define Cosmology
            cosmo = LambdaCDM(H0=s.header.hubble*100,
                              Om0=s.header.omega_m,
                              Ode0=s.header.omega_l)
            cosmosim = {'omega_M_0' : s.header.omega_m,
                        'omega_lambda_0' : s.header.omega_l,
                        'omega_k_0' : 0.0,
                        'h' : s.header.hubble}
            
            smlpixel = args["sml"]  # maximum smoothing pixel length
            shpos = dfhalosnap.filter(regex='HaloPosBox').iloc[ll].values
            #time_start = time.time()
            ## BH
            pos, indx = dmaps.select_particles(
                    BH['Pos'], shpos, #*a/h,
                    dfhalosnap['fov_Mpc'].values[ll], 'box')
            bh_sigma = dmaps.projected_density_pmesh(
                    pos, BH['Mass'][indx],
                    dfhalosnap['fov_Mpc'].values[ll],
                    args["ncells"])
            ## Star
            pos, indx = dmaps.select_particles(
                    Gas['Pos'], shpos, #*a/h,
                    dfhalosnap['fov_Mpc'].values[ll], 'box')
            gas_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, Gas['Mass'][indx],
                    dfhalosnap['fov_Mpc'].values[ll],
                    args["ncells"],
                    hmax=smlpixel)
            ## Gas
            pos, indx = dmaps.select_particles(
                    Star['Pos'], shpos,  #*a/h
                    dfhalosnap['fov_Mpc'].values[ll], 'box')
            star_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, Star['Mass'][indx],
                    dfhalosnap['fov_Mpc'].values[ll],
                    args["ncells"],
                    hmax=smlpixel)
            ## DM
            pos, indx = dmaps.select_particles(
                    DM['Pos'], shpos,  #*a/h
                    dfhalosnap['fov_Mpc'].values[ll],  'box')
            dm_sigma = dmaps.projected_density_pmesh_adaptive(
                    pos, DM['Mass'][indx],
                    dfhalosnap['fov_Mpc'].values[ll],  #[Mpc]
                    args["ncells"],
                    hmax=smlpixel)
            sigmatotal = dm_sigma+gas_sigma+star_sigma+bh_sigma

            # Make sure that density-map if filled
            extention = 0
            while 0.0 in sigmatotal and (extention < 60):
                extention += 5
                dm_sigma = dmaps.projected_density_pmesh_adaptive(
                        pos, DM['Mass'][indx],
                        dfhalosnap['fov_Mpc'].values[ll],  #[Mpc]
                        args["ncells"],
                        hmax=smlpixel+extention)
                sigmatotal = dm_sigma+gas_sigma+star_sigma+bh_sigma

            sigma_tot.append(sigmatotal)
            out_hfid.append(dfhalosnap['HF_ID'].values[ll])
            out_lcid.append(dfhalosnap['LC_ID'].values[ll])
            out_fov.append(dfhalosnap['fov_Mpc'].values[ll])
            if args["walltime"] - (time_start - time.time())/(60*60) < 0.25:
                fname = args["outbase"]+'DM_'+label+'_lc'+str(lclabel)+'.h5'
                hf = h5py.File(fname, 'w')
                hf.create_dataset('density_map', data=sigma_tot)
                hf.create_dataset('HF_ID', data=np.asarray(out_hfid))
                hf.create_dataset('LC_ID', data=np.asarray(out_lcid))
                hf.create_dataset('fov_Mpc', data=np.asarray(out_fov))
                hf.close()
    
    fname = args["outbase"]+'DM_'+label+'_lc_'+str(lclabel)+'.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('density_map', data=sigma_tot)
    hf.create_dataset('HF_ID', data=np.asarray(out_hfid))
    hf.create_dataset('LC_ID', data=out_lcid)
    hf.create_dataset('fov_Mpc', data=np.asarray(out_fov))
    #RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    hf.close()


if __name__ == "__main__":
    create_density_maps()

