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


def subhalo_data(hfdir, hfname, snapnum, h, unit):
    """
    Input:
        hfdir: halo finder output directory
        hfname: halo finder name
        snapnum: snapshot number
        h: hubble parameter
        unit: length units of halo positions in simulation
    """
    exp = np.floor(np.log10(np.abs(unit))).astype(int)

    
    if hfname == 'Rockstar':
        # [X, Y, Z] in [Mpc]
        hffile = hfdir+'halos_%d.dat' % snapnum
        df = pd.read_csv(hffile, sep='\s+', skiprows=16,
                         usecols=[0, 2, 4, 9, 10, 11],
                         names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        df = df[df['Mvir'] > 3e11]
        if exp == 23:  #[Mpc]
            pass
        elif exp == 21:  #[kpc]
            df.loc[:, 'X'] *= 1e3/h
            df.loc[:, 'Y'] *= 1e3/h
            df.loc[:, 'Z'] *= 1e3/h
        else:
            raise Exception('This unit can not be convertet for Rockstar')

    elif hfname == 'Subfind':
        s = read_hdf5.snapshot(snapnum, hfdir)
        s.group_catalog(["SubhaloIDMostbound", "SubhaloPos", "SubhaloMass",
                         "SubhaloVelDisp", "GroupFirstSub"])
        indx = s.cat['GroupFirstSub'].astype(int)
        df = pd.DataFrame({'ID'   : s.cat['SubhaloIDMostbound'][indx],
                           'Vrms' : s.cat['SubhaloVelDisp'][indx],
                           'X'    : s.cat['SubhaloPos'][indx, 0],
                           'Y'    : s.cat['SubhaloPos'][indx, 1],
                           'Z'    : s.cat['SubhaloPos'][indx, 2],
                           'Mass' : s.cat['SubhaloMass'][indx]})
        df = df[df['Mass'] > 5e11]
        if exp == 21:  #simulation in [kpc]
            pass
        elif exp == 23:  #simulation in [Mpc]
            df.loc[:, 'X'] *= 1e-3*h
            df.loc[:, 'Y'] *= 1e-3*h
            df.loc[:, 'Z'] *= 1e-3*h
        else:
            raise Exception('This unit can not be convertet for Subfind')

    SH = {'ID' : df['ID'].values.astype('float64'), #for MPI.DOUBLE datatype
          'Vrms' : df['Vrms'].values.astype('float64'),
          'X' : df['X'].values.astype('float64'),
          'Y' : df['Y'].values.astype('float64'),
          'Z' : df['Z'].values.astype('float64')}
    del df
    return SH
              

def particle_data(sdata, h, unit):
    if unit == 'kpc':  #halo-finder in [kpc]
        scale = 1
    elif unit == 'Mpc':  #halo-finder in [Mpc]
        scale = 1e-3*h
    else:
        raise Exception('This unit can not be convertet')

    DM = {'Mass' : (sdata['Masses']['dm']).astype('float64'),
          'Pos' : (sdata['Coordinates']['dm']*scale).astype('float64')}
    
    Gas = {'Mass' : (sdata['Masses']['gas']).astype('float64'),
           'Pos' : (sdata['Coordinates']['gas']*scale).astype('float64')}
    
    age = (sdata['GFM_StellarFormationTime']['stars']).astype('float64')
    Star = {'Mass' : (sdata['Masses']['stars'][age >= 0]).astype('float64'),
            'Pos' : (sdata['Coordinates']['stars'][age >= 0, :]*scale).astype('float64')}
    del age

    BH = {'Mass' : (sdata['Masses']['bh']).astype('float64'),
          'Pos' : (sdata['Coordinates']['bh']*scale).astype('float64')}

    return DM, Gas, Star, BH
