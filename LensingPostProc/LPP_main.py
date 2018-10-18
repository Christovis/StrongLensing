from __future__ import division
import os, sys, glob, logging
import numpy as np
import pickle
import pandas as pd
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import cfuncs as cf
import LPP_funcs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf
import readsnap
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
#import lm_funcs_mp # Why do I need to load this???
import LM_main
from LM_main import plant_Tree # Why do I need to load this???


################################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')

################################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, hfname, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

################################################################################
# Run through simulations
for sim in range(len(sim_dir)):
    # File for lensing-maps
    lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+hfname+'/'+sim_name[sim]+'/'
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'

    # Units of Simulation
    scale = rf.simulation_units(sim_dir[sim])

    # Cosmological Parameters
    s = read_hdf5.snapshot(45, sim_dir[sim])
    cosmo = LambdaCDM(H0=s.header.hubble*100,
                      Om0=s.header.omega_m,
                      Ode0=s.header.omega_l)
    h = s.header.hubble
   
    HF_ID=[]; HaloLCID=[]; SrcID=[]; Mdyn=[]; Mlens=[];
    # Run through LensingMap output files 
    #for lm_file in glob.glob(lm_dir+'LM_1_Proc_*_0.pickle'):
    for lm_file in glob.glob(lm_dir+'LM_new_GR.pickle'):
        # Load LensingMap Contents
        LM = pickle.load(open(lm_file, 'rb'))
       
        previous_snapnum = -1
        print('::::::::::', len(LM['HF_ID'][:]))
        # Run through lenses
        for ll in range(len(LM['Halo_ID'])):
            # Load Lens properties
            HaloHFID = int(LM['HF_ID'][ll]) #int(LM['Rockstar_ID'][ll])
            HaloPosBox = LM['HaloPosBox'][ll]
            HaloVel = LM['HaloVel'][ll]
            snapnum = LM['snapnum'][ll]
            zl = LM['zl'][ll]

            # Only load new particle data if lens is at another snapshot
            if (previous_snapnum != snapnum):
                rks_file = '/cosma5/data/dp004/dc-beck3/rockstar/'+sim_phy[sim]+ \
                           sim_name[sim]+'/halos_' + str(snapnum)+'.dat'
                hdata = pd.read_csv(rks_file, sep='\s+', skiprows=np.arange(1, 16))
                # Load Particle Properties
                # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
                snap = snapfile % (snapnum, snapnum)
                #gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*scale
                #gas_vel = readsnap.read_block(snap, 'VEL ', parttype=0)
                star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale
                star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
                star_vel = readsnap.read_block(snap, 'VEL ', parttype=4)
                star_mass = readsnap.read_block(snap, 'MASS', parttype=4)*1e10/h
                star_pos = star_pos[star_age >= 0]
                star_vel = star_vel[star_age >= 0]
                star_mass = star_mass[star_age >= 0]
                del star_age
                #star_pos = np.vstack((star_pos, gas_pos))
                #del gas_pos
                #star_vel = np.vstack((star_vel, gas_vel))
                #del gas_vel
            previous_snapnum = snapnum

            # Load Halo Properties
            indx = hdata['#ID'][hdata['#ID'] == HaloHFID].index[0]
            Vrms = hdata['Vrms'][indx]*(u.km/u.s)  #[km/s]
            Rvir = hdata['Rvir'][indx]*u.kpc
            #Rhalfmass = hdata['Halfmass_Radius'][indx]*u.kpc
            #hpos = pd.concat([df['X']*scale, df['Y']*scale, df['Z']*scale],
            #                 axis=1).loc[[indx]].values
            hvel = pd.concat([hdata['VX'], hdata['VY'], hdata['VZ']],
                             axis=1).loc[[indx]].values
            epva = pd.concat([hdata['A[x]'], hdata['A[y]'], hdata['A[z]']],
                             axis=1).loc[[indx]].values
            epvb = pd.concat([hdata['B[x]'], hdata['B[y]'], hdata['B[z]']],
                             axis=1).loc[[indx]].values
            epvc = pd.concat([hdata['C[x]'], hdata['C[y]'], hdata['C[z]']],
                             axis=1).loc[[indx]].values
            
            ####----> Add keys <----####
            ## Stellar Half Mass Radius
            Rshm = cf.call_stellar_halfmass(star_pos[:, 0], star_pos[:, 1],
                                            star_pos[:, 2], HaloPosBox[0],
                                            HaloPosBox[1], HaloPosBox[2],
                                            star_mass, Rvir.to_value('Mpc'))*u.Mpc
            if Rshm == 0.0:
                continue
            
            ## Stellar Half Light Radius
            ### https://arxiv.org/pdf/1804.04492.pdf $3.3

            ## Mass dynamical
            star_indx = lppf.check_in_sphere(HaloPosBox, star_pos,
                                             Rshm.to_value('kpc'))
            if len(star_indx[0]) > 100:
                slices = np.vstack((epva/np.linalg.norm(epva),
                                    epvb/np.linalg.norm(epvb),
                                    epvc/np.linalg.norm(epvc)))
                mdynn = lppf.mass_dynamical(Rshm, star_vel[star_indx],
                                            HaloPosBox, hvel[0], slices)
            else:
                continue
            if np.isnan(mdynn):
                continue

            ## Mass strong lensing
            # Run through sources
            for ss in range(len(LM['Sources']['Src_ID'][ll])):
                n_imgs = len(LM['Sources']['mu'][ll][ss])
                #print('The source has %d lensed images' % n_imgs,
                #      LM['Sources']['theta'][ll][ss])
                if n_imgs == 1:
                    continue
                zs = LM['Sources']['zs'][ll][ss]
                Rein = LM['Sources']['Rein'][ll][ss]*u.kpc
                if Rein == 0.0:
                    continue
                #print('-> \t The Einstein Radius is: %.3f' % Rein.to_value('kpc'))
                Mlens.append(lppf.mass_lensing(Rein, zl, zs, cosmo))
                Mdyn.append(mdynn)
                HF_ID.append(HaloHFID)
                HaloLCID.append(LM['Halo_ID'][ll])
                SrcID.append(LM['Sources']['Src_ID'][ll][ss])
    Mlens = np.asarray(Mlens)
    Mdyn = np.asarray(Mdyn)
    HF_ID = np.asarray(HF_ID)
    HaloLCID = np.asarray(HaloLCID)
    SrcID = np.asarray(SrcID)
    lpp_dir = HQ_dir+'/LensingPostProc/'+sim_phy[sim]+hfname+'/'
    hf = h5py.File(lpp_dir+'LPP_'+sim_name[sim]+'_Rshm_all.h5', 'w')
    hf.create_dataset('Halo_HF_ID', data=HF_ID)  # Rockstar ID
    hf.create_dataset('Halo_LC_ID', data=HaloLCID)
    hf.create_dataset('Src_ID', data=SrcID)
    hf.create_dataset('MassDynamical', data=Mdyn)
    hf.create_dataset('MassLensing', data=Mlens)
    hf.close()
        ## Magnitude of SNIa multiple images
        # Run through sources
        #for ss in range(len(LM['Sources']['Src_ID'][ll])):
        #    zs = LM['Sources']['zs'][ll][ss]
        #    Rein = LM['Sources']['Rein'][ll][ss]*u.kpc
        #    t = LM['Sources']['delta_t'][ll][ss]
        #    f = LM['Sources']['mu'][ll][ss]
        #    indx_max = np.argmax(t)
        #    t -= t[indx_max]
        #    t = np.absolute(t[t != 0])

#    # Run through sources
#    for ss in range(len(LM['Sources']['Src_ID'][ll])):
#        zs = LM['Sources']['zs'][ll][ss]
#        Rein = LM['Sources']['Rein'][ll][ss]*u.kpc
#        Mlens = mass_lensing(Rein, zl, zs, cosmo)
#        Star_indx = check_in_sphere(HaloPosBox, Star_pos, Rhalfmass.to_value('kpc'))
#        t = LM['Sources']['delta_t'][ll][ss]
#        f = LM['Sources']['mu'][ll][ss]
#        indx_max = np.argmax(t)
#        t -= t[indx_max]
#        t = np.absolute(t[t != 0])
#        for ii in range(len(t)):
#                rank.append(ii)
#                delta_t.append(t[ii])
