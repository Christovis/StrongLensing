from __future__ import division
import os, sys, glob, logging
import numpy as np
print('Numpy is in:', np.__file__)
import pandas as pd
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingAnalysis/')
import LM_main_box  # Don't remove!!!
from LM_main_box import plant_Tree
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/lib/')
import lens as load
import lpp_cfuncs as cf
import lpp_pyfuncs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf
#sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
#import LM_main_box
#from LM_main_box import plant_Tree # Why do I need to load this???

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
    label = args["simdir"].split('/')[-2].split('_')[-2]
    
    # Units of Simulation
    scale = rf.simulation_units(args["simdir"])

    # Cosmological Parameters
    s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
    cosmo = LambdaCDM(H0=s.header.hubble*100,
                      Om0=s.header.omega_m,
                      Ode0=s.header.omega_l)
    h = s.header.hubble

    # Stellar Data
    stars = load.load_stars(args["snapnum"], args["simdir"])
    dm = load.load_dm(args["snapnum"], args["simdir"])
    gas = load.load_gas(args["snapnum"], args["simdir"])
    
    indxdrop = []  # collect indices of subhalos falling through criterias
    if args["lenses"] == 1:
        lafile = glob.glob(args["ladir"]+"*"+"_lens_"+"*"+"409.pickle")[0]
        lenses = load.load_subhalos(args["snapnum"], args["simdir"],
                                    lafile, strong_lensing=1)
        # Run through lenses
        for ll in range(len(lenses.index.values)):
            print('Lenses: %f' % (ll/len(lenses.index.values)))
            lens = lenses.iloc[ll]
            
            if isinstance(lens, (pd.core.series.Series)):
                indx = load.select_particles(
                        stars['Pos'], lens['Pos'],
                        lens['Rstellarhalfmass']*1.5,
                        'sphere')
                halo_stars = {'Pos' : stars['Pos'][indx, :],
                              'Vel' : stars['Vel'][indx, :],
                              'Mass' : stars['Mass'][indx]}
                halo_stars['Pos'] -= lens['Pos']
                
                indx = load.select_particles(
                        dm['Pos'], lens['Pos'],
                        lens['Rein'],
                        'sphere')
                halo_dm = {'Pos' : dm['Pos'][indx, :],
                           'Mass' : dm['Mass'][indx]}
                halo_dm['Pos'] -= lens['Pos']
                
                indx = load.select_particles(
                        gas['Pos'], lens['Pos'],
                        lens['Rein'],
                        'sphere')
                halo_gas = {'Pos' : gas['Pos'][indx, :],
                           'Mass' : gas['Mass'][indx]}
                halo_gas['Pos'] -= lens['Pos']


                lenses, indxdrop = load.add_properties(
                        halo_stars, halo_dm, halo_gas,
                        lens, lenses,
                        cosmo, s, indxdrop, ll, args["lenses"])
            else:
                print('!!!!!!! SOS !!!!!!!!!')

        lenses = lenses.drop(indxdrop)
        print('Saving %d lenses to .hdf5' % (len(lenses.index.values)))
        zllabel = str(lens['ZL']).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(lenses['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'lens', zllabel, zslabel))
        lenses.to_hdf(fname, key='lenses', mode='w')

    if args["lenses"] == 0:
        print(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5")
        print(glob.glob(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5"))
        lafile = glob.glob(args["ladir"]+"*"+"_nonlens_"+"*"+"409.h5")[0]
        subhalos = load.load_subhalos(args["snapnum"], args["simdir"],
                                      lafile, strong_lensing=0)
        # Run through subhalos
        for ll in range(len(subhalos.index.values)):
            print('Non-Lenses: %f' % (ll/len(subhalos.index.values)))
            subhalo = subhalos.iloc[ll]
            
            if isinstance(subhalo, (pd.core.series.Series)):
                indx = load.select_particles(
                        stars['Pos'], subhalo['Pos'],
                        subhalo['Rstellarhalfmass']*1.5,
                        'sphere')
                halo_stars = {'Pos' : stars['Pos'][indx, :],
                              'Vel' : stars['Vel'][indx, :],
                              'Mass' : stars['Mass'][indx]}
                halo_stars['Pos'] -= subhalo['Pos']
                
                #indx = load.select_particles(
                #        dm['Pos'], subhalo['Pos'],
                #        subhalo['Rstellarhalfmass']*1.5,
                #        'sphere')
                #halo_dm = {'Pos' : dm['Pos'][indx, :],
                #           'Vel' : dm['Vel'][indx, :],
                #           'Mass' : dm['Mass'][indx]}
                #halo_dm['Pos'] -= subhalo['Pos']

                subhalos, indxdrop = load.add_properties(
                        halo_stars, subhalo, subhalos,
                        cosmo, s, indxdrop, ll, args["lenses"])
            else:
                print('!!!!!!! SOS !!!!!!!!!')
            

        subhalos = subhalos.drop(indxdrop)
        print('Saving %d Subhalos to .hdf5' % (len(subhalos.index.values)))
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        zslabel = '{:<03d}'.format(int(str(subhalos['ZS'].values[0]).replace('.', '')))
        fname = (args["outbase"]+'LPPBox_%s_%s_zl%szs%s.h5' % \
                (label, 'nonlens', zllabel, zslabel))
        subhalos.to_hdf(fname, key='subhalos', mode='w')

if __name__ == '__main__':
    lensing_signal()
