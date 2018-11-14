from __future__ import division
import os, sys, glob, logging
import numpy as np
import pickle
import pandas as pd
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
import LM_main_box
from LM_main_box import plant_Tree # Why do I need to load this???

# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')


def Euler_angles(sub_av, sub_bv, sub_cv):
    """ Based on Stark 1997 equ. 2 """
    vu = np.arcsin(sub_bv[:, 2])
    phi = np.arcsin(sub_av[:, 1])
    return vu, phi


def lensing_signal():
    # Get command line arguments
    args = {}
    args["simdir"]       = sys.argv[1]
    args["snapnum"]      = int(sys.argv[2])
    args["ladir"]        = sys.argv[3]
    args["rksdir"]        = sys.argv[4]
    args["outbase"]      = sys.argv[5]
    args["radius"]      = sys.argv[6]
    #args["zs"]      = sys.argv[7]

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

    Lens = {"HFID":[], "Vrms":[], "FOV":[], "ELIP":[], "PA":[]}
    # Run through LensingMap output files 
    for lm_file in glob.glob(args["ladir"]+"*"+"409.pickle"):
        # Load LensingMap Contents
        LM = pickle.load(open(lm_file, 'rb'))
        print('Processing the following file: \n %s' % (lm_file))
        print('which contains %d lenses' % len(LM['HF_ID'][:]))
        print('with max. einst. radius: %f', np.max(LM['Sources']['Rein'][:]))
        label = args["simdir"].split('/')[-2].split('_')[-2]

        snapnum = LM['snapnum']
        dfh = pd.read_csv(args["rksdir"]+'halos_%d.dat' % snapnum,
                          sep='\s+', skiprows=np.arange(1, 16))
       
        print(LM['FOV'])
        # Run through lenses
        for ll in range(len(LM['HF_ID'])):
            HFID = int(LM['HF_ID'][ll])
            FOV = int(LM['FOV'][ll])  #[arcsec]

            # Load Halo Properties
            indx = dfh['#ID'][dfh['#ID'] == HFID].index[0]
            HPos = [dfh['X'][indx], dfh['Y'][indx], dfh['Z'][indx]]
            Vrms = dfh['Vrms'][indx]  #[km/s]
            hvel = pd.concat([dfh['VX'], dfh['VY'], dfh['VZ']],
                             axis=1).loc[[indx]].values
            epveca = pd.concat([dfh['A[x]'], dfh['A[y]'], dfh['A[z]']],
                             axis=1).loc[[indx]].values
            epvecb = pd.concat([dfh['B[x]'], dfh['B[y]'], dfh['B[z]']],
                             axis=1).loc[[indx]].values
            epvecc = pd.concat([dfh['C[x]'], dfh['C[y]'], dfh['C[z]']],
                             axis=1).loc[[indx]].values
            
            # Ellipticity (should be between 0.0-0.8)
            epval = [np.linalg.norm(vec) for vec in np.vstack((epveca, epvecb, epvecc))]
            print('epval', epval)
            indx = np.argsort(epval)
            a = epval[indx[2]]  # major
            b = epval[indx[1]]  # intermediate
            c = epval[indx[0]]  # minor
            ellipticity = (a-b) / (2*(a+b+c))
            print(a, b, c)
            print('ellipticity', ellipticity)

            # Position-Angle [rad]
            # Based on Stark 1977; Binggeli 1980
            """
            [vu, phi] = Euler_angles(sub_av, sub_bv, sub_cv)
            j = e1**2*e2**2*np.sin(vu)**2 + \
                e1**2*np.cos(vu)*np.cos(phi)**2 + \
                e2**2*np.cos(vu)**2*np.sin(phi)**2
            l = e1**2*np.sin(phi)**2 + e2**2*np.cos(phi)**2
            k = (e1**2 - e2**2)*np.sin(phi)*np.cos(phi)*np.cos(vu)
            ep = np.sqrt((j + l + np.sqrt((j - l)**2 + 4*k**2)) /
                         (j + l - np.sqtr((j - l)**2 + 4*k**2)))
            # Observed ellipticity
            e = 1 - 1/ep
            f = e1**2*np.sin(vu)**2*np.sin(phi)**2 +
                e2**2*np.sin(vu)**2*np.cos(phi)**2 + np.cos(vu)**2
            """
            PA = 0.

            Lens['HFID'].append(HFID)
            Lens['FOV'].append(FOV)
            Lens['Vrms'].append(Vrms)
            Lens['ELIP'].append(ellipticity)
            Lens['PA'].append(PA)
        df = pd.DataFrame.from_dict(Lens)
        zllabel = str(zl).replace('.', '')[:3].zfill(3)
        fname = (args["outbase"]+'LPPBox_%s_zl%s.txt' % \
                (label, zllabel))
        df.to_csv(fname, header=True, index=None, sep=' ', mode='w')

if __name__ == '__main__':
    lensing_signal()
