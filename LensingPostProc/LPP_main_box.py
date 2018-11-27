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
    
    lm_file = glob.glob(args["ladir"]+"*"+"409.pickle")[0]
    label = args["simdir"].split('/')[-2].split('_')[-2]

    # Stellar Data
    stars = lens.load_stars(args["snapnum"], args["simdir"])
    print('Min stellar mass', np.min(stars['Mass']))
    if args["lenses"] == 1:
        lenses = lens.load_subhalos(args["snapnum"], args["simdir"],
                                    lafile, strong_lensing=1)
        # Run through lenses
        for ll in range(len(lenses.index.values)):
            print('Lenses: %f' % (ll/len(lenses.index.values)))
            # Select particles for profiles
            # bound
            indxstart = lenses['offset'].values[ll]
            indxend = lenses['Npart'].values[ll]+lenses['offset'].values[ll]
            halo_stars = {'Pos' : stars['Pos'][indxstart:indxend, :],
                          'Vel' : stars['Vel'][indxstart:indxend, :],
                          'Mass' : stars['Mass'][indxstart:indxend]}
            print('Nr. of stars in halo: %d' % len(halo_stars['Mass'][:]))
            halo_stars['Pos'] = halo_stars['Pos'] - \
                                lenses['Pos'].values[ll]
            halo_stars['Vel'] = halo_stars['Vel'] - \
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
            lenses['DensProfRad'][ll], lenses['DensProfMeas'][ll] = lppf.profiles(
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

        print('Saving %d lenses to .hdf5' % (len(lenses.index)))
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(lenses['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'lens', zllabel, zslabel))
        lenses.to_hdf(fname, key='lenses', mode='w')

    if args["lenses"] == 0:
        subhalos = lens.load_subhalos(args["snapnum"], args["simdir"],
                                    LM, strong_lensing=0)
        # Run through subhalos
        for ll in range(len(subhalos.index.values)):
            print('Subhalos: %f' % (ll/len(subhalos.index.values)))

            
            ####----> Add keys <----####
            # Select particles for profiles
            radii = [0, subhalos['Rstellarhalfmass'].values[ll]]
            indx = lens.select_particles(
                    stars['Pos'][:], subhalos['Pos'].values[ll],
                    radii[-1], 'sphere')
            if len(indx) < 100:
                continue
            ## Vrms Profile
            subhalos['VrmsProfRad'][ll], subhalos['VrmsProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], halo_stars['Vel'],
                    'vrms', 4, s, radii[0], radii[-1])
            ## Density Profile
            densradii = 40
            subhalos['DensProfRad'][ll], subhalos['DensProfMeas'][ll] = lppf.profiles(
                    halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                    'density', 4, s, radii[0], densradii)


        print('Saving %d Subhalos to .hdf5' % (len(lenses.index)))
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(lenses['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'nonlens', zllabel, zslabel))
        subhalos.to_hdf(fname, key='subhalos', mode='w')

if __name__ == '__main__':
    lensing_signal()
