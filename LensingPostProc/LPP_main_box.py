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
import lens
import lpp_cfuncs as cf
import lpp_pyfuncs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import LM_main_box
from LM_main_box import plant_Tree # Why do I need to load this???

# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')


def lensing_signal():
    # Get command line arguments
    args = {}
    args["snapnum"]      = int(sys.argv[1])
    args["simdir"]       = sys.argv[2]
    args["ladir"]        = sys.argv[3]
    args["hfname"]       = sys.argv[4]
    args["hfdir"]        = sys.argv[5]
    args["outbase"]      = sys.argv[6]
    args["radius"]       = sys.argv[7]
    args["lenses"]       = int(sys.argv[8])

    snapfile = args["simdir"]+'snapdir_%03d/snap_%03d'
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
    
    label = args["simdir"].split('/')[-2].split('_')[-2]

    # Stellar Data
    stars = lens.load_stars(args["snapnum"], args["simdir"])
    dm = lens.load_dm(args["snapnum"], args["simdir"])
    
    indxdrop = []  # collect indices of subhalos falling through criterias
    if args["lenses"] == 1:
        lafile = glob.glob(args["ladir"]+"*"+"_lens_"+"*"+"409.pickle")[0]
        lenses = lens.load_subhalos(args["snapnum"], args["simdir"],
                                    lafile, strong_lensing=1)
        # Run through lenses
        for ll in range(len(lenses.index.values)):
            print('Lenses: %f' % (ll/len(lenses.index.values)))
            # Select particles for profiles
            # bound
            indxstart = lenses['soffset'].values[ll]
            indxend = lenses['sNpart'].values[ll]+lenses['soffset'].values[ll]
            print('Nr. of stars in halo: %d' % (lenses['sNpart'].values[ll]))
            if lenses['sNpart'].values[ll] < 100:
                indxdrop.append(lenses.index.values[ll])
                continue
            
            halo_stars = {'Pos' : stars['Pos'][indxstart:indxend, :],
                          'Vel' : stars['Vel'][indxstart:indxend, :],
                          'Mass' : stars['Mass'][indxstart:indxend]}
            halo_stars['Pos'] = halo_stars['Pos'] - \
                                lenses['Pos'].values[ll]
            halo_stars['Vel'] = halo_stars['Vel'] - \
                                lenses['Vel'].values[ll]
            
            
            indxstart = lenses['dmoffset'].values[ll]
            indxend = lenses['dmNpart'].values[ll]+lenses['dmoffset'].values[ll]
            print('Nr. of dm in halo: %d' % (lenses['dmNpart'].values[ll]))
            halo_dm = {'Pos' : dm['Pos'][indxstart:indxend, :],
                       'Vel' : dm['Vel'][indxstart:indxend, :],
                       'Mass' : dm['Mass'][indxstart:indxend]}
            halo_dm['Pos'] = halo_dm['Pos'] - \
                                lenses['Pos'].values[ll]
            halo_dm['Vel'] = halo_dm['Vel'] - \
                                lenses['Vel'].values[ll]

            
            ## Mass strong lensing
            Rein_arc = lenses['Rein'].values[ll]*u.arcsec
            Rein = Rein_arc.to_value('rad') * \
                    cosmo.angular_diameter_distance(zl).to('kpc')
            lenses['Mlens'].values[ll] = lppf.mass_lensing(Rein,
                                                   lenses['ZL'].values[ll],
                                                   lenses['ZS'].values[ll],
                                                   cosmo)
            
            # Ellipticity & Prolateness
            lenses['Ellipticity'][ll], lenses['Prolateness'][ll] = lppf.ellipticity_and_prolateness(halo_stars['Pos'])

            ## Vrms Profile
            lenses['VrmsProfRad'][ll], lenses['VrmsProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], halo_stars['Vel'],
                    'vrms', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])
            ## Velocity Profile
            lenses['VelProfRad'][ll], lenses['VelProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], halo_stars['Vel'],
                    'velocity', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])
            ## Density Profile
            lenses['SDensProfRad'][ll], lenses['SDensProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'density', 4, s, 0.1, 40)
            ## Mass Profile
            lenses['SMProfRad'][ll], lenses['SMProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'mass', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])
            ## Circular Velocity Profile
            lenses['SCVProfRad'][ll], lenses['SCVProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'circular_velocity', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])
            ## Density Profile
            lenses['DMDensProfRad'][ll], lenses['DMDensProfMeas'][ll] = lppf.profiles(
                    halo_dm['Pos'], halo_dm['Mass'], np.array([]),
                    'density', 4, s, 0.1, 40)
            ## Mass Profile
            lenses['DMMProfRad'][ll], lenses['DMMProfMeas'][ll] = lppf.profiles(
                    halo_dm['Pos'], halo_dm['Mass'], np.array([]),
                    'mass', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])
            ## Circular Velocity Profile
            lenses['DMCVProfRad'][ll], lenses['DMCVProfMeas'][ll] = lppf.profiles(
                    halo_dm['Pos'], halo_dm['Mass'], np.array([]),
                    'circular_velocity', 4, s, 0.1, lenses['Rstellarhalfmass'].values[ll])

        lenses = lenses.drop(indxdrop)
        print('Saving %d lenses to .hdf5' % (len(lenses.index.values)))
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(lenses['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'lens', zllabel, zslabel))
        lenses.to_hdf(fname, key='lenses', mode='w')

    if args["lenses"] == 0:
        print(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5")
        print(glob.glob(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5"))
        lafile = glob.glob(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5")[0]
        subhalos = lens.load_subhalos(args["snapnum"], args["simdir"],
                                      lafile, strong_lensing=0)
        # Run through subhalos
        for ll in range(len(subhalos.index.values)):
            print('Non-lenses: %f' % (ll/len(subhalos.index.values)))
            # Select particles for profiles
            # bound
            indxstart = subhalos['offset'].values[ll]
            indxend = subhalos['Npart'].values[ll]+subhalos['offset'].values[ll]
            print('Nr. of stars in halo: %d' % (subhalos['Npart'].values[ll]))
            if subhalos['Npart'].values[ll] < 100:
                indxdrop.append(subhalos.index.values[ll])
                continue

            halo_stars = {'Pos' : stars['Pos'][indxstart:indxend, :],
                          'Vel' : stars['Vel'][indxstart:indxend, :],
                          'Mass' : stars['Mass'][indxstart:indxend]}
            halo_stars['Pos'] = halo_stars['Pos'] - \
                                subhalos['Pos'].values[ll]
            halo_stars['Vel'] = halo_stars['Vel'] - \
                                subhalos['Vel'].values[ll]
            
            
            ## Mass strong lensing
            Rein_arc = subhalos['Rein'].values[ll]*u.arcsec
            Rein = Rein_arc.to_value('rad') * \
                    cosmo.angular_diameter_distance(zl).to('kpc')
            subhalos['Mlens'].values[ll] = lppf.mass_lensing(Rein,
                                                   subhalos['ZL'].values[ll],
                                                   subhalos['ZS'].values[ll],
                                                   cosmo)
            
            # Ellipticity & Prolateness
            subhalos['Ellipticity'][ll], subhalos['Prolateness'][ll] = lppf.ellipticity_and_prolateness(halo_stars['Pos'])

            ## Vrms Profile
            subhalos['VrmsProfRad'][ll], subhalos['VrmsProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], halo_stars['Vel'],
                    'vrms', 4, s, 0.1, subhalos['Rstellarhalfmass'].values[ll])
            ## Velocity Profile
            subhalos['VelProfRad'][ll], subhalos['VelProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], halo_stars['Vel'],
                    'velocity', 4, s, 0.1, subhalos['Rstellarhalfmass'].values[ll])
            ## Density Profile
            subhalos['DensProfRad'][ll], subhalos['DensProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'density', 4, s, 0.1, 40)
            ## Mass Profile
            subhalos['SMProfRad'][ll], subhalos['SMProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'mass', 4, s, 0.1, subhalos['Rstellarhalfmass'].values[ll])
            ## Circular Velocity Profile
            subhalos['SCVProfRad'][ll], subhalos['SCVProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'circular_velocity', 4, s, 0.1, subhalos['Rstellarhalfmass'].values[ll])

        subhalos = subhalos.drop(indxdrop)
        print('Saving %d Subhalos to .hdf5' % (len(subhalos.index.values)))
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(subhalos['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'nonlens', zllabel, zslabel))
        subhalos.to_hdf(fname, key='subhalos', mode='w')

if __name__ == '__main__':
    lensing_signal()
