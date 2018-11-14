from __future__ import division
import os, sys, glob, logging
import numpy as np
import pickle
import pandas as pd
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/lib/')
import cfuncs as cf
import lppfuncs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf
import readsnap
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import LM_main
from LM_main import plant_Tree # Why do I need to load this???

# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')


def lensing_signal():
    # Get command line arguments
    args = {}
    args["snapnum"]      = int(sys.argv[1])
    args["simdir"]       = sys.argv[2]
    args["ladir"]        = sys.argv[3]
    args["rksdir"]        = sys.argv[4]
    args["outbase"]      = sys.argv[5]
    args["radius"]      = sys.argv[6]
    #args["zs"]      = sys.argv[7]

    snapfile = args["simdir"]+'snapdir_%03d/snap_%03d'
    print(snapfile)
    # Units of Simulation
    scale = rf.simulation_units(args["simdir"])

    # Cosmological Parameters
    s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
    cosmo = LambdaCDM(H0=s.header.hubble*100,
                      Om0=s.header.omega_m,
                      Ode0=s.header.omega_l)
    h = s.header.hubble
    zl = s.header.redshift
    print('Analyse sub-&halo at z=%f' % zl)
   
    Lens = {"ID" : [],
            #"LCID" : [],
            #"SrcID" : [],
            "Mdyn" : [],
            "Vrms_stellar" : [],
            "Vrms_rks" : [],
            "Mlens" : [],
            "Rein" : []}
    # Run through LensingMap output files 
    for lm_file in glob.glob(args["ladir"]+"*"+"150.pickle"):
        # Load LensingMap Contents
        LM = pickle.load(open(lm_file, 'rb'))
        print('Processing the following file: \n %s' % (lm_file))
        print('which contains %d lenses' % len(LM['HF_ID'][:]))
        print('with max. einst. radius: %f', np.max(LM['Sources']['Rein'][:]))
        label = args["simdir"].split('/')[-2].split('_')[-2]

        previous_snapnum = -1
        # Run through lenses
        for ll in range(len(LM['HF_ID'])):
            # Load Lens properties
            HaloHFID = int(LM['HF_ID'][ll]) #int(LM['Rockstar_ID'][ll])
            #HaloPosBox = LM['HaloPosBox'][ll]
            snapnum = LM['snapnum']  #[ll]
            zl = LM['zl']  #[ll]
            print('Lens ID: %d' % (HaloHFID))

            # Only load new particle data if lens is at another snapshot
            if (previous_snapnum != snapnum):
                hdata = pd.read_csv(args["rksdir"], sep='\s+', skiprows=np.arange(1, 16))
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
            HPos = [hdata['X'][indx], hdata['Y'][indx], hdata['Z'][indx]]
            Vrms = hdata['Vrms'][indx]*(u.km/u.s)  #[km/s]
            hvel = pd.concat([hdata['VX'], hdata['VY'], hdata['VZ']],
                             axis=1).loc[[indx]].values
            epva = pd.concat([hdata['A[x]'], hdata['A[y]'], hdata['A[z]']],
                             axis=1).loc[[indx]].values
            epvb = pd.concat([hdata['B[x]'], hdata['B[y]'], hdata['B[z]']],
                             axis=1).loc[[indx]].values
            epvc = pd.concat([hdata['C[x]'], hdata['C[y]'], hdata['C[z]']],
                             axis=1).loc[[indx]].values
            
            ####----> Add keys <----####
            if args["radius"] == 'Rshm':
                #Rhalfmass = hdata['Halfmass_Radius'][indx]*u.kpc
                # Stellar Half Mass Radius
                Rad_dyn = cf.call_stellar_halfmass(
                                star_pos[:, 0], star_pos[:, 1],
                                star_pos[:, 2], HPos[0],
                                HPos[1], HPos[2],
                                star_mass, Rvir.to_value('Mpc'))*u.Mpc
                if Rshm == 0.0:
                    continue
                
                ## Stellar Half Light Radius
                ### https://arxiv.org/pdf/1804.04492.pdf $3.3
            else:
                Rad_dyn = hdata['Rvir'][indx]*u.kpc

            ## Mass dynamical
            star_indx = lppf.check_in_sphere(HPos, star_pos,
                                             Rad_dyn.to_value('kpc'))
            if len(star_indx[0]) > 100:
                slices = np.vstack((epva/np.linalg.norm(epva),
                                    epvb/np.linalg.norm(epvb),
                                    epvc/np.linalg.norm(epvc)))
                mdynn, vrms_s = lppf.mass_dynamical(Rad_dyn, star_vel[star_indx],
                                                  HPos, hvel[0], slices)
            else:
                print('Not enough particles for Mdyn')
                continue
            if np.isnan(mdynn):
                continue

            ## Mass strong lensing
            # Run through sources
            #for ss in range(len(LM['Sources']['Src_ID'][ll])):
            #print('The source has %d lensed images' % n_imgs,
            #      LM['Sources']['theta'][ll][ss])
            zs = LM['zs']
            Rein = LM['Sources']['Rein'][ll]*u.kpc
            if Rein == 0.0:
                continue
            #print('-> \t The Einstein Radius is: %.3f' % Rein.to_value('kpc'))
            Lens["Mlens"].append(lppf.mass_lensing(Rein, zl, zs, cosmo))
            Lens["Rein"].append(Rein)
            Lens["Mdyn"].append(mdynn)
            Lens["Vrms_stellar"].append(vrms_s)
            Lens["Vrms_rks"].append(Vrms)
            Lens["ID"].append(HaloHFID)
            #HaloLCID.append(LM['Halo_ID'][ll])
            #SrcID.append(LM['Sources']['Src_ID'][ll])
    
    df = pd.DataFrame.from_dict(Lens)
    print('Saving %d lenses to .hdf5' % (len(df.index)))
    zllabel = str(zl).replace('.', '')[:3].zfill(3)
    zslabel = '{:<03d}'.format(int(str(1.5).replace('.', '')))
    fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
            (label, args["radius"], zllabel, zslabel))
    df.to_hdf(fname, key='df', mode='w')

    #hf = h5py.File(args["outbase"]+'LPP_'+label+'_'+args["radius"]+'_Box.h5', 'w')
    #hf.create_dataset('Halo_HF_ID', data=HF_ID)  # Rockstar ID
    #hf.create_dataset('Halo_LC_ID', data=HaloLCID)
    #hf.create_dataset('MassDynamical', data=Mdyn)
    #hf.create_dataset('MassLensing', data=Mlens)
    #hf.create_dataset('Src_ID', data=SrcID)
    #hf.close()
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

if __name__ == '__main__':
    lensing_signal()
