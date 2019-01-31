from __future__ import division
import os, sys, glob
import numpy as np
import pickle
import pandas as pd
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingAnalysis/')
import LM_main_box
from LM_main_box import plant_Tree
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/lib/')
import lpp_cfuncs as cf
import lpp_pyfuncs as lppf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
from find_bound_particles import BoundParticleFinder
import readlensing as rf
pd.options.mode.chained_assignment = None  # Disable warning; default='warn'


def select_particles(_pos, _centre, _radius, _regiontype):
    """
    Parameters
    ----------
    _pos : np.ndarray
    _centre : np.array
    _radius : np.float
    _regiontype : str
    
    Returns
    -------
    indx : np.array
    """
    _pos = _pos - _centre
    if _regiontype == 'box':
        indx = np.where((np.abs(_pos[:, 0]) < 0.5*_radius) &
                        (np.abs(_pos[:, 1]) < 0.5*_radius) &
                        (np.abs(_pos[:, 2]) < 0.5*_radius))[0]
    elif _regiontype == 'sphere':
        _dist = np.sqrt(_pos[:, 0]**2 + \
                        _pos[:, 1]**2 + \
                        _pos[:, 2]**2)
        indx = np.where(_dist <= _radius)[0]
    return indx


def load_stars(snapnum, snapfile):
    """
    Parameters
    ----------
    snapnum : int
        snapshot number of simulation
    snapfile : str
        path to snapshot directory
    
    Returns
    -------
    indx : np.array
    """
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.read(["Coordinates", "Masses",
            "Velocities",
            "GFM_StellarFormationTime",
            "GFM_StellarPhotometrics"],
            parttype=[4])
    age = (s.data['GFM_StellarFormationTime']['stars']).astype('float64')
    star_pos = s.data['Coordinates']['stars'][age >= 0, :]
    star_vel = s.data['Velocities']['stars'][age >= 0, :]
    star_mass = s.data['Masses']['stars'][age >= 0]   #[Msol]
    star_mag = s.data['GFM_StellarPhotometrics']['stars'][age >= 0]  #[Bol. Mag]

    star = {'Pos' : star_pos,
            'Vel' : star_vel,
            'Mass' : star_mass,
            'Mag' : star_mag,
            }
    return star


def load_gas(snapnum, snapfile):
    """
    Parameters
    ----------
    snapnum : int
        snapshot number of simulation
    snapfile : str
        path to snapshot directory
    
    Returns
    -------
    indx : np.array
    """
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.read([
        "Coordinates", 
        "Masses",
        ], parttype=[0])

    gas = {'Pos' : s.data['Coordinates']['gas'],
            'Mass' : s.data['Masses']['gas'],
            }
    return gas


def load_dm(snapnum, snapfile):
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.read([
        "Coordinates",
        "Masses",
        ], parttype=[1])
    dm_pos = s.data['Coordinates']['dm']
    dm_mass = s.data['Masses']['dm']   #[Msol]

    dm = {'Pos' : dm_pos,
          'Mass' : dm_mass}
    return dm


def load_subhalos(snapnum, snapfile, lafile, strong_lensing=1):
    """
    Parameters
    ----------
    snapnum : int
        snapshot number of simulation
    snapfile : str
        path to snapshot directory
    lafile : str
        path to lensing analysis results
    subhalo_tag : boolean
        switch to ouput all subhalo acting as
        strong lenses or not

    Returns
    -------
    df : pd.DataFrame
    """
    # load snapshot data
    s = read_hdf5.snapshot(snapnum, snapfile)
    s.group_catalog(["SubhaloIDMostbound",
                     "SubhaloPos",
                     "SubhaloVel",
                     "SubhaloMass",
                     "SubhaloVelDisp",
                     "SubhaloHalfmassRadType",
                     "SubhaloLenType",
                     "GroupLenType",
                     "GroupNsubs",
                     "GroupFirstSub"])
    df = pd.DataFrame({'HF_ID' : s.cat['SubhaloIDMostbound'],
                       'Vrms' : s.cat['SubhaloVelDisp'],
                       'Mass' : s.cat['SubhaloMass'],
                       'Rstellarhalfmass' : s.cat['SubhaloHalfmassRadType'][:, 4],
                       'sNpart' : s.cat["SubhaloLenType"][:, 4].astype('int'),
                       'dmNpart' : s.cat["SubhaloLenType"][:, 1].astype('int')})
    s1 = pd.Series(dict(list(enumerate(s.cat['SubhaloPos']))), index=df.index)
    df['Pos'] = s1
    s1 = pd.Series(dict(list(enumerate(s.cat['SubhaloVel']))), index=df.index)
    df['Vel'] = s1
    df['Index'] = df.index.values
    df = df.sort_values(by=['HF_ID'])
    df = df.set_index('HF_ID')

    if strong_lensing == True:
        LA = pickle.load(open(lafile, "rb"))  #, encoding='latin1')
        print('Processing the following file: \n %s' % (lafile))
        print('which contains %d lenses' % len(LA['HF_ID'][:]))
        print('with max. einst. radius: %f', np.max(LA['Sources']['Rein'][:]))
        
        # Output only subhalos acting as gravitational strong lenses
        # Find intersection
        ladf = pd.DataFrame({
            'HF_ID' : LA["HF_ID"],
            'ZS' : LA["zs"],
            'FOV' : LA["FOV"],
            'Rein' : LA["Sources"]["Rein"],
            })
        ladf['Nimg'] = pd.Series(0, index=ladf.index)
        ladf['theta'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        ladf['delta_t'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        ladf['mu'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        ladf['TCC'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        ladf['DMAP'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        for ll in range(len(ladf.index.values)):
            ladf['Nimg'][ll] = len(LA['Sources']['mu'][ll])
            ladf['theta'][ll] = LA['Sources']['theta'][ll]
            ladf['delta_t'][ll] = LA['Sources']['delta_t'][ll]
            ladf['mu'][ll] = LA['Sources']['mu'][ll]
            ladf['TCC'][ll] = LA['Sources']['TCC'][ll]
            ladf['DMAP'][ll] = LA['DMAP'][ll]
        
        ladf = ladf.sort_values(by=['HF_ID'])
        ladf = ladf.set_index('HF_ID')  # may contain dublicates
        #df = df[df.index.isin(ladf.index.values)]
        # Attach df to ladf
        ladf = ladf.join(df, how='inner')

        ladf['ZL'] = pd.Series(s.header.redshift, index=ladf.index)
        #df['Rein'] = ladf['Rein']
        #df['ZS'] = ladf['ZS']
        #df['Nimg'] = ladf['Nimg']
        #df['FOV'] = ladf['FOV']
        #df['theta'] = ladf['theta']
        #df['delta_t'] = ladf['delta_t']
        #df['mu'] = ladf['mu']
        #df['DMAP'] = ladf['DMAP']
       
        # Find bound particles for lenses
        BPF = BoundParticleFinder(s)
        subhalo_particles = BPF.find_bound_subhalo_particles(ladf['Index'].values, 4)
        ladf['BPF'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
        for ll in range(len(ladf.index.values)):
            ladf['BPF'][ll] = subhalo_particles[ll].astype(int)

        # Initialize
        ladf['Mlens'] = pd.Series(0, index=ladf.index)
    elif strong_lensing == False:
        ladf = pd.read_hdf(lafile, key='nonlenses')
        ladf = ladf.sort_values(by=['HF_ID'])
        ladf = ladf.set_index('HF_ID')
        print('Processing the following file: \n %s' % (lafile))
        print('which contains %d subhalos' % len(ladf.index.values))
        
        # Output only subhalos do not act as gravitational strong lenses
        # Attach df to ladf
        ladf = ladf.rename(columns = {'zl':'ZL'})
        ladf = ladf.rename(columns = {'zs':'ZS'})
        ladf = ladf.join(df, how='inner')


    # Initialize
    ladf['Mdyn'] = pd.Series(0, index=ladf.index)
    ladf['Mtotal'] = pd.Series(0, index=ladf.index)
    ladf['Vrms_Rein'] = pd.Series(0, index=ladf.index)
    ladf['Vrms_Rhm'] = pd.Series(0, index=ladf.index)
    ladf['PA'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['Ellip2D'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['Eccen2D'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['Ellip3D'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['Eccen3D'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['Prolat3D'] = pd.Series(0.0, dtype=np.float, index=ladf.index)
    ladf['VelProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['VelProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['VrmsProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['VrmsProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SDensProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SDensProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SMProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SMProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SCVProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['SCVProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['DMDensProfRad'] = pd.Series(0, index=df.index).astype(np.ndarray)
    ladf['DMDensProfMeas'] = pd.Series(0, index=df.index).astype(np.ndarray)
    ladf['GDensProfRad'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['GDensProfMeas'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['power_law_index'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    ladf['power_law_profile'] = pd.Series(0, index=ladf.index).astype(np.ndarray)
    #df['DMMProfRad'] = pd.Series(0, index=df.index).astype(np.ndarray)
    #df['DMMProfMeas'] = pd.Series(0, index=df.index).astype(np.ndarray)
    #df['DMCVProfRad'] = pd.Series(0, index=df.index).astype(np.ndarray)
    #df['DMCVProfMeas'] = pd.Series(0, index=df.index).astype(np.ndarray)
    return ladf


def add_properties(halo_stars, halo_dm, halo_gas,
        lens, lenses, cosmo, s, indxdrop, ll, strong_lensing):
    """
    """
    print('nr of halo_stars', len(halo_stars['Mass']))
    if len(halo_stars['Mass']) < 100 or lens['Rein'] < 0.3:
        # min. num. of particles to compute shape and profiles
        indxdrop.append(lenses.index.values[ll])
        return lenses, indxdrop
    else:
        if strong_lensing == 1:
            Rein_arc = lens['Rein']*u.arcsec
            Rein = Rein_arc.to_value('rad') * \
                    cosmo.angular_diameter_distance(lens['ZL']).to('kpc')
            
            _dist = np.sqrt(halo_stars['Pos'][:, 0]**2 + \
                            halo_stars['Pos'][:, 1]**2 + \
                            halo_stars['Pos'][:, 2]**2)
            indx = np.where(_dist <= Rein.to_value('kpc'))[0]
            #if len(indx) > 1:
            #    pass
            #else:
            #    return lenses, indxdrop
            ## Mass strong lensing
            lenses['Mlens'].values[ll] = lppf.mass_lensing(
                    Rein.to_value('kpc'),
                    lens['ZL'], lens['ZS'],
                    cosmo)

            # Mass stellar kinematics
            vrms = lppf.vrms_at_radius(
                    halo_stars, lens['Vel'],
                    float(Rein.to_value('kpc')),
                    )
            
            lenses['Vrms_Rein'][ll] = vrms
            lenses['Mdyn'][ll] = lppf.mass_dynamical(
                    vrms*u.km/u.second, Rein)
            
            lenses['Mtotal'][ll] = np.sum(halo_stars['Mass'][indx]) + \
                                   np.sum(halo_dm['Mass']) + \
                                   np.sum(halo_gas['Mass'])

        # Ellipticity & Prolateness
        lenses['PA'][ll], lenses['Eccen2D'][ll], lenses['Ellip2D'][ll] = lppf.morphology2D(
                halo_stars['Pos'], 0)
        lenses['Ellip3D'][ll], lenses['Eccen3D'][ll], lenses['Prolat3D'][ll] = lppf.morphology3D(
                halo_stars['Pos'])
        
        ## Density Profile
        #radii = np.array([0.05, 1])*lens['Rstellarhalfmass']
        if lens['Mass'] > 1e14:
            radii = np.array([0.6, 70])
            nr_of_bins = 80
        elif lens['Mass'] > 1e13:
            radii = np.array([0.6, 7])
            nr_of_bins = 50
        elif lens['Mass'] > 1e12:
            radii = np.array([0.4, 7])
            nr_of_bins = 40
        lenses['SDensProfRad'][ll], lenses['SDensProfMeas'][ll] = lppf.profiles(
                halo_stars['Pos'], halo_stars['Mass'], np.array([]),
                'density', 4, s, radii[0], radii[1], nr_of_bins)
        lenses['DMDensProfRad'][ll], lenses['DMDensProfMeas'][ll] = lppf.profiles(
                halo_dm['Pos'], halo_dm['Mass'], np.array([]),
                'density', 4, s, radii[0], radii[1], nr_of_bins)
        lenses['GDensProfRad'][ll], lenses['GDensProfMeas'][ll] = lppf.profiles(
                halo_gas['Pos'], halo_gas['Mass'], np.array([]),
                'density', 4, s, radii[0], radii[1], nr_of_bins)
        total_density = lenses['SDensProfMeas'].values[ll] + \
                        lenses['DMDensProfMeas'].values[ll] + \
                        lenses['GDensProfMeas'].values[ll]
        lenses['power_law_index'][ll], lenses['power_law_profile'][ll] = lppf.scale_free_power_law(
                lenses['SDensProfRad'].values[ll], total_density)

        return lenses, indxdrop

#indx = load.select_particles(dm['Pos'], lens['Pos'],
#                             lens['Rstellarhalfmass'], 'sphere')
#halo_dm = {'Pos' : dm['Pos'][indx, :],
#           'Vel' : dm['Vel'][indx, :],
#           'Mass' : dm['Mass'][indx]}
#halo_dm['Pos'] -= lens['Pos']

#halo_stars = {'Pos' : stars['Pos'][lens['BPF'][0]:lens['BPF'][1], :],
#              'Vel' : stars['Vel'][lens['BPF'][0]:lens['BPF'][1], :],
#              'Mass' : stars['Mass'][lens['BPF'][0]:lens['BPF'][1]]}
#halo_stars['Pos'] = halo_stars['Pos'] - lenses['Pos'].values[ll]
#halo_stars['Vel'] = halo_stars['Vel'] - lenses['Vel'].values[ll]
#
#halo_dm = {'Pos' : dm['Pos'][lens['BPF'][0]:lens['BPF'][1], :],
#           'Vel' : dm['Vel'][lens['BPF'][0]:lens['BPF'][1], :],
#           'Mass' : dm['Mass'][lens['BPF'][0]:lens['BPF'][1]]}
#halo_dm['Pos'] = halo_dm['Pos'] - lenses['Pos'].values[ll]
#halo_dm['Vel'] = halo_dm['Vel'] - lenses['Vel'].values[ll]

#lenses['DMDensProfRad'][ll], lenses['DMDensProfMeas'][ll] = lppf.profiles(
#        halo_dm['Pos'], halo_dm['Mass'], np.array([]),
#        'density', 1, s, radii[0], radii[1])

## Mass Profile
#lenses['SMProfRad'][ll], lenses['SMProfMeas'][ll] = lppf.profiles(
#        halo_stars['Pos'], halo_stars['Mass'], np.array([]),
#        'mass', 4, s, radii[0], radii[1])
## Circular Velocity Profile
#lenses['SCVProfRad'][ll], lenses['SCVProfMeas'][ll] = lppf.profiles(
#        halo_stars['Pos'], halo_stars['Mass'], np.array([]),
#        'circular_velocity', 4, s, radii[0], radii[1])
## Mass Profile
#lenses['DMMProfRad'][ll], lenses['DMMProfMeas'][ll] = lppf.profiles(
#        halo_dm['Pos'], halo_dm['Mass'], np.array([]),
#        'mass', 1, s, radii[0], radii[1])
## Circular Velocity Profile
#lenses['DMCVProfRad'][ll], lenses['DMCVProfMeas'][ll] = lppf.profiles(
#        halo_dm['Pos'], halo_dm['Mass'], np.array([]),
#        'circular_velocity', 1, s, radii[0], radii[1])


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
            
            indx = self.df.index.intersection(df.index)
            self.df["M200"]


    def intersection():
        lcdf = lcdf.set_index('LC_ID')
        lcdf.index.intersection(dmdf.index)





