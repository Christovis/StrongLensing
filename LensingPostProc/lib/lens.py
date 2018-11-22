from __future__ import division
import os, sys, glob, logging
import numpy as np
import pickle
import pandas as pd
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import LM_main_box
from LM_main_box import plant_Tree
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/lib/')
import lpp_cfuncs as cf
import lpp_pyfuncs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf


def load_stars(snapnum, snapfile):
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.read(["Coordinates", "Masses",
            "Velocities", "GFM_StellarFormationTime"],
            parttype=[4])
    age = (s.data['GFM_StellarFormationTime']['stars']).astype('float64')
    star_pos = s.data['Coordinates']['stars'][age >= 0, :]
    star_mass = s.data['Masses']['stars'][age >= 0]
    star_vel = s.data['Velocities']['stars'][age >= 0, :]

    #df = pd.DataFrame({'Mass':star_mass})
    #s1 = pd.Series(dict(list(enumerate(star_pos))),
    #               index=df.index)
    #df['Pos'] = s1
    #s1 = pd.Series(dict(list(enumerate(star_vel))),
    #               index=df.index)
    #df['Vel'] = s1
    star = {'Pos' : star_pos,
            'Vel' : star_vel,
            'Mass' : star_mass}
    return star


def load_subhalos(snapnum, snapfile, LM):
    # load snapshot data
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.group_catalog(["SubhaloIDMostbound",
                     "SubhaloPos",
                     "SubhaloVel",
                     "SubhaloMass",
                     "SubhaloVelDisp",
                     "SubhaloHalfmassRadType",
                     "SubhaloLenType"])
    df = pd.DataFrame({'HF_ID' : s.cat['SubhaloIDMostbound'],
                       'Vrms' : s.cat['SubhaloVelDisp'],
                       'Mass' : s.cat['SubhaloMass'],
                       'Rstellarhalfmass' : s.cat['SubhaloHalfmassRadType'][:, 4],
                       'Npart' : s.cat["SubhaloLenType"][:, 4]})
    subhalo_offset = (np.cumsum(df['Npart'].values) - df['Npart'].values).astype(int)
    df['offset'] = pd.Series(subhalo_offset, index=df.index,
                             dtype=int)
    s1 = pd.Series(dict(list(enumerate(
        s.cat['SubhaloPos']
        ))), index=df.index)
    df['Pos'] = s1
    s1 = pd.Series(dict(list(enumerate(s.cat['SubhaloVel']))),
                   index=df.index)
    df['Vel'] = s1
    df = df.sort_values(by=['HF_ID'])
    df = df.set_index('HF_ID')
    
    # Find intersection
    df = df[df.index.isin(LM['HF_ID'])]
    lmdf = pd.DataFrame({'Rein' : LM['Sources']['Rein'],
                         'FOV' : LM["FOV"],
                         'HF_ID' : LM["HF_ID"]})
    lmdf['Nimg'] = pd.Series(0, index=lmdf.index)
    lmdf['theta'] = pd.Series(0, index=lmdf.index)
    lmdf['delta_t'] = pd.Series(0, index=lmdf.index)
    lmdf['mu'] = pd.Series(0, index=lmdf.index)
    lmdf['theta'] = lmdf['theta'].astype('object')
    lmdf['delta_t'] = lmdf['delta_t'].astype('object')
    lmdf['mu'] = lmdf['mu'].astype('object')
    for ll in range(len(lmdf.index.values)):
        lmdf['Nimg'][ll] = len(LM['Sources']['mu'][ll])
        lmdf['theta'][ll] = LM['Sources']['theta'][ll]
        lmdf['delta_t'][ll] = LM['Sources']['delta_t'][ll]
        lmdf['mu'][ll] = LM['Sources']['mu'][ll]
    
    lmdf = lmdf.sort_values(by=['HF_ID'])
    lmdf = lmdf.set_index('HF_ID')

    df['Rein'] = lmdf['Rein']
    df['Nimg'] = lmdf['Nimg']
    df['FOV'] = lmdf['FOV']
    df['theta'] = lmdf['theta']
    df['delta_t'] = lmdf['delta_t']
    df['mu'] = lmdf['mu']
    return df


class Lenses():
    def __init__(self, hfname, LM):
        self.halofinder = hfname
        
        if self.halofinder == 'Rockstar':
            self.df = pd.DataFrame({"HF_ID" : LM['HF_ID'][:],
                                    "LC_ID" : LM['LC_ID'][:],
                                    "SrcID" : [],
                                    "Nimgs" : [],
                                    "M200"  : [],
                                    "zl"    : [],
                                    "zs"    : [],
                                    "Mdyn_rks" : [],
                                    "Mdyn_stellar" : [],
                                    "Mlens" : [],
                                    "Vrms_stellar" : [],
                                    "Vrms_rks" : [],
                                    "Rein"  : []})
        elif self.halofinder == 'Subfind':
            print(LM['LC_ID'])
            self.df = pd.DataFrame({"HF_ID" : LM['HF_ID'][:],
                                    #"LC_ID" : LM['LC_ID'][:],
                                    "SrcID" : np.zeros(len(LM['HF_ID'][:])), 
                                    "Nimgs" : np.zeros(len(LM['HF_ID'][:])),
                                    "M200"  : np.zeros(len(LM['HF_ID'][:])),
                                    "zl"    : np.zeros(len(LM['HF_ID'][:])),
                                    "zs"    : np.zeros(len(LM['HF_ID'][:])),
                                    "Mdyn_stellar" : np.zeros(len(LM['HF_ID'][:])),
                                    "Mlens" : np.zeros(len(LM['HF_ID'][:])),
                                    "Vrms_stellar" : np.zeros(len(LM['HF_ID'][:])),
                                    "Rein"  : np.zeros(len(LM['HF_ID'][:]))})


    def load_snapshot(self, snapnum, hfdir):
        """
        Load data of subhalos and particles
        """
        self.snapshot = read_hdf5_eagle.snapshot(snapnum, part_file)
        
        # Subhalos
        if self.halofinder == 'Rockstar':
            df = pd.read_csv(args["rksdir"]+'halos_%d.dat' % snapnum,
                             sep='\s+', skiprows=np.arange(1, 16))
            df = df.sort_values(by=['#ID'])
            df = df.set_index('#ID')
            subhalo = hdata.loc[hdata['#ID'] == HFID]
            HPos = subhalo[['X', 'Y', 'Z']].values[0]
            Vrms = subhalo['Vrms'].values[0]
            M200 = subhalo['Mvir'].values[0]
            hvel = subhalo[['VX', 'VY', 'VZ']].values[0]
            epva = subhalo[['A[x]', 'A[y]', 'A[z]']].values[0]
            epvb = subhalo[['B[x]', 'B[y]', 'B[z]']].values[0]
            epvc = subhalo[['C[x]', 'C[y]', 'C[z]']].values[0]

        elif self.halofinder == 'Subfind':
            s = read_hdf5.snapshot(snapnum, hfdir)
            s.group_catalog(["SubhaloIDMostbound", "SubhaloPos", "SubhaloVel",
                             "SubhaloMass", "SubhaloVelDisp",
                             "SubhaloHalfmassRadType", "SubhaloLenType"])
            df = pd.DataFrame({'HF_ID' : self.snapshot.cat['SubhaloIDMostbound'],
                               'Vrms' : (s.cat['SubhaloVelDisp']),
                               'Mass' : s.cat['SubhaloMass'],
                               'Npart' : (self.snapshot.cat["SubhaloLenType"][:, 4])})
            subhalo_offset = (np.cumsum(df['Npart'].values) - \
                              df['Npart'].values).astype(int)
            df['offset'] = pd.Series(subhalo_offset, index=dfpart.index, dtype=int)
            s1 = pd.Series(dict(list(enumerate(
                s.cat['SubhaloPos']*self.snapshot.header.hubble*1e-3
                ))), index=df.index)
            df['Pos'] = s1
            s1 = pd.Series(dict(list(enumerate(s.cat['SubhaloVel']))), index=dmdf.index)
            df['Vel'] = s1
            df = df.sort_values(by=['HF_ID'])
            df = df.set_index('HF_ID')
            print('HF_DI test', df.index.values)
            print(self.df.index.intersection(df.index))
            
            indx = self.df.index.intersection(df.index)
            self.df["M200"]


    def intersection():
        lcdf = lcdf.set_index('LC_ID')
        lcdf.index.intersection(dmdf.index)





