from __future__ import division
import os, sys, logging
import pandas as pd
import pickle
import numpy as np
from scipy import stats
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import cfuncs as cf
import LPP_funcs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/')  # parent directory
import read_hdf5
import readlensing as rf
import readsnap
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import lm_tools  # Why do I need to load this???


################################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')

################################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, hfname, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

CPUs = 5 # Number of CPUs to use has to be the same as in LensingMap/LM_create.py

################################################################################
# Run through simulations
for sim in range(len(sim_dir)):
    # File for lens & source properties
    lc_file = lc_dir[sim]+hfname+'/LC_SN_'+sim_name[sim]+'_rndseed1.h5'
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
    
    # Prepatre Processes to be run in parallel
    jobs = []
    manager = multiprocessing.Manager()
    results_per_cpu = manager.dict()
    for cpu in range(CPUs)[:1]:
        # Load LensingMaps Contents
        lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
        LM = pickle.load(open(lm_file, 'rb'))
        p = Process(target=la.dyn_vs_lensing_mass, name='Proc_%d'%cpu,
                    args=(cpu, LC, lm_file, snapfile, h, scale,
                          HQ_dir, sim, sim_phy, sim_name, hfname,
                          cosmo, results_per_cpu))
        p.start()
        jobs.append(p)
    #for p in jobs:
    #    p.join()


        previous_snapnum = -1
        # Run through lenses
        for ll in range(len(LM['Halo_ID'])):
            # Load Lens properties
            HaloHFID = int(LM['Rockstar_ID'][ll])
            HaloPosBox = LM['HaloPosBox'][ll]
            HaloVel = LM['HaloVel'][ll]
            snapnum = LM['snapnum'][ll]
            zl = LM['zl'][ll]

            # Only load new particle data if lens is at another snapshot
            if (previous_snapnum != snapnum):
                rks_file = '/cosma5/data/dp004/dc-beck3/rockstar/'+sim_phy[sim]+ \
                           sim_name[sim]+'/halos_' + str(snapnum)+'.dat'
                df = pd.read_csv(rks_file, sep='\s+', skiprows=np.arange(1, 16))
                # Load Particle Properties
                #s = read_hdf5.snapshot(snapnum, snapfile)
                # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
                # Have to load all particles :-( -> takes too long
                #s.read(["Velocities", "Coordinates", "AGE"], parttype=-1)
                #Star_pos = s.data['Velocities']['stars']*scale
                snap = snapfile % (snapnum, snapnum)
                star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale
                star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
                star_vel = readsnap.read_block(snap, 'VEL ', parttype=4)
                star_mass = readsnap.read_block(snap, 'MASS', parttype=4)*1e10/h
                star_pos = star_pos[star_age >= 0]
                star_vel = star_vel[star_age >= 0]
                star_mass = star_mass[star_age >= 0]
                del star_age
            previous_snapnum = snapnum

            # Load Halo Properties
            indx = df['#ID'][df['#ID'] == HaloHFID].index[0]
            Vrms = df['Vrms'][indx]*(u.km/u.s)  #[km/s]
            Rvir = df['Rvir'][indx]*u.kpc
            Rhalfmass = df['Halfmass_Radius'][indx]*u.kpc
            #hpos = pd.concat([df['X']*scale, df['Y']*scale, df['Z']*scale],
            #                 axis=1).loc[[indx]].values
            hvel = pd.concat([df['VX'], df['VY'], df['VZ']],
                             axis=1).loc[[indx]].values
            epva = pd.concat([df['A[x]'], df['A[y]'], df['A[z]']],
                             axis=1).loc[[indx]].values
            epvb = pd.concat([df['B[x]'], df['B[y]'], df['B[z]']],
                             axis=1).loc[[indx]].values
            epvc = pd.concat([df['C[x]'], df['C[y]'], df['C[z]']],
                             axis=1).loc[[indx]].values
            
            ####----> Add keys <----####
            ## Stellar Half Mass Radius
            Rshm = cf.call_stellar_halfmass(star_pos[:, 0], star_pos[:, 1],
                                            star_pos[:, 2], HaloPosBox[0],
                                            HaloPosBox[1], HaloPosBox[2],
                                            star_mass, Rvir.to_value('Mpc'))*u.Mpc
            
            ## Stellar Half Light Radius
            ### https://arxiv.org/pdf/1804.04492.pdf $3.3

            ## Mass dynamical
            star_indx = lppf.check_in_sphere(HaloPosBox, star_pos,
                                             Rshm.to_value('kpc'))
            if len(Star_indx[0]) > 50:
                slices = np.vstack((avec/np.linalg.norm(avec),
                                    bvec/np.linalg.norm(bvec),
                                    cvec/np.linalg.norm(cvec)))
                Mdyn = lppf.mass_dynamical(Rshm, star_vel[star_indx],
                                           HaloPosBox, hvel, sigma)
                results.append([LM['Halo_ID'][ll], LM['Sources']['Src_ID'][ll][ss],
                                Mdyn, Mlens])
        
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
