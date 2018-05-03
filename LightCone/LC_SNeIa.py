#!/usr/bin/env python
import sys
import numpy as np
import random as rnd
import astropy
from astropy.cosmology import Planck15
from astropy import cosmology 
from astropy import units as u
import readsnap
from scipy.interpolate import interp1d
import time
import h5py
import CosmoDist as cd
import constants as cc
#from Supernovae import SN_Type_Ia as SN
import lc_supernovae as SN
sys.path.append('/cosma5/data/dp004/dc-beck3')
import readlensing as rf


############################################################################
data = open('/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt', 'r')
Settings = data.readlines()
sim_dir = []
sim_phy = []
sim_name = []
sim_col = []
hf_dir = []
lc_dir = []
HQ_dir = []
for k in range(len(Settings)):
    if 'Python' in Settings[k].split():
        HQdir = Settings[k+1].split()[0]
        HQ_dir.append(HQdir)
    if 'Simulation' in Settings[k].split():
        [simphy, simname] = Settings[k+1].split()
        sim_phy.append(simphy)
        sim_name.append(simname)
        [simdir, simcol, simunit] = Settings[k+2].split()
        sim_dir.append(simdir)
        sim_col.append(simcol)
        [hfdir, hf_name] = Settings[k+3].split()
        hf_dir.append(hfdir)
        lcdir = Settings[k+4].split()[0]
        lc_dir.append(lcdir)

c = 299792.458  #[km/s] speed of light
#'omega_M_0' : 0.308900,
#'omega_lambda_0' : 0.691100,
#'omega_k_0' : 0.0,
#'h' : 0.677400}
# Time interval
years = 1e1  # [yr]
###########################################################################

# Iterate through Simulations
for sim in range(len(sim_dir)):
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d.0'

    # Cosmological Constants
    num_of_snapshots = 45 
    header = readsnap.snapshot_header(snapfile % (num_of_snapshots, num_of_snapshots))
    cosmosim = {'omega_M_0' : header.omega_m,
                'omega_lambda_0' : header.omega_l,
                'omega_k_0' : 0.0,
                'h' : header.hubble}

    # Load Lightcone
    LCFile = lc_dir[sim]+'LC_'+sim_name[sim]+'.h5'
    LCHalos = rf.LightCone_without_SN(LCFile, 'dictionary')
    # Select Halos
    LCHalos = SN.select_halos(LCHalos)
    print('Number of Halos: ', len(LCHalos['M200']))

    fz = open('/cosma5/data/dp004/dc-beck3/z_lcone.txt', 'r')
    zlcone = fz.readlines()
    z_lcone = []
    for k in range(len(zlcone)):
        z_lcone.append(float(zlcone[k]))
    z_lcone = np.array(z_lcone)
    fd = open('/cosma5/data/dp004/dc-beck3/CoDi.txt', 'r')
    dist = fd.readlines()
    CoDi = []
    for k in range(len(dist)):
        CoDi.append(float(dist[k]))
    CoDi = np.array(CoDi)
    # Interpolation fct. between comoving dist. and redshift
    reddistfunc = interp1d(CoDi, z_lcone, kind='cubic')
    # convert conformal time to redshift
    agefunc, redfunc = cd.quick_age_function(20., 0., 0.001, 1, **cosmosim)

    # Redshift bins for SNeIa
    zr = np.linspace(0, 2, 1000)
    dist_zr = [cd.comoving_distance(z,0.,**cosmosim) for z in zr]#[Mpc]
    # middle redshift & distnace
    zmid = np.linspace(0.002/2, 2, 998)
    dist_mid = [cd.comoving_distance(z,0.,**cosmosim) for z in zmid]#[Mpc]
    # Comoving distance between redshifts
    dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim) for j in range(len(zr)-1)]#[Mpc]

    # Initialize Lists to save
    L_indx=[]; L_fov=[]; S_ID=[]; S_z=[]; S_possky=[]; A_E=[]; R_E=[]

    # Iterate over lenses
    for i in range(len(LCHalos['Halo_ID'])):
        # Lens properties
        l_posbox = LCHalos['HaloPosBox'][i, :]
        l_poslc = LCHalos['HaloPosLC'][i, :]
        l_z = LCHalos['redshift'][i]
        lensdist = np.sqrt(l_poslc[0]**2 + l_poslc[1]**2 + l_poslc[2]**2)  # [Mpc/h]
        
        # Find starting redshift of Box
        indx = np.where(zr > LCHalos['redshift'][i])[0]
        distmid_p = [dist_mid[idx] for idx in indx[:-2]]
        distbet_p = [dist_bet[idx] for idx in indx[:-2]]
        dist_sr = [dist_zr[idx] for idx in indx]
        
        # Calculate app. Einstein radius
        thetaE_inf = SN.Einstein_ring(LCHalos['VelDisp'][i], c, LCHalos['redshift'][i],
                                      lensdist, None, 'rad')
        u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
        l_lenspos = np.linalg.norm(l_poslc)

        # Calculate Volume for SNeIa distribution
        V, re = SN.Einstein_Volume(thetaE_inf, distmid_p, distbet_p)

        # Number of SNeIa in time and Volume
        SNIa_num = SN.SNeIa_distr(years, V, zmid, agefunc, redfunc, **cosmosim)
        [dec, SNIa_num] = np.modf(SNIa_num)
        rand = rnd.random()
        indx = np.where(dec > rnd.random())
        SNIa_num[indx] += 1

        # initilize variable to see if lens has supernova source
        checksrc = 0
        # Find Volums wih Supernovae
        indx = np.nonzero(SNIa_num)
        # Iterate over Volume
        for y in indx[0]:
            [dist_min, dist_max] = [dist_sr[y], dist_sr[y+1]]
            # Iterate over Supernovae
            for x in range(int(SNIa_num[y])):
                # Generate random SNeIa location within a cone
                # with apex-angle thetaE_inf along the l.o.s.
                radial_rnd = u_lenspos*(l_lenspos + \
                             rnd.random()*(dist_max-l_lenspos))  # [Mpc]
                srcdist = np.sqrt(radial_rnd[0]**2 + radial_rnd[1]**2 + radial_rnd[2]**2)
                src_z = reddistfunc(srcdist)
                # max. distance equa to re
                sposx = rnd.random()*re[y]
                sposy = rnd.random()*re[y]
                # Calculate Einstein Radius for Source
                Ds = Planck15.angular_diameter_distance(src_z)  #[Mpc]
                Dls = Planck15.angular_diameter_distance_z1z2(l_z, src_z)  #[Mpc]
                # only valid for point lense... not accurate
                #[thetaE, radiusE] = SN.Einstein_ring(LCHalos['VelDisp'][i],
                #                     c, l_z, Dls, Ds, 'kpc')  #[rad],[kpc]
                # Write glafic file
                S_ID.append(i);         S_z.append(src_z)
                S_possky.append([sposx, sposy])
                #A_E.append(thetaE);     R_E.append(radiusE)
            checksrc = 1
        if checksrc == 1:
            # Write glafic file
            fov = thetaE_inf*180*3600/np.pi  # convert radians to arcsec
            print(((fov*u.arcsec).to_value('rad')*Planck15.angular_diameter_distance(l_z)).to_value('Mpc'))
            print(LCHalos['Rvir'][i]*0.3*1e-3)
            print('------------------------')
            L_indx.append(i);   L_fov.append(fov)
    
    S_ID = np.asarray(S_ID);       S_z = np.asarray(S_z)
    S_possky = np.asarray(S_possky)
    A_E = np.asarray(A_E);         R_E = np.asarray(R_E)
    L_indx = np.asarray(L_indx);   L_fov = np.asarray(L_fov)
    #hf = h5py.File(lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5', 'w')
    print(lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5')
    hf.create_dataset('Halo_ID', data=L_indx)
    hf.create_dataset('snapnum', data=LCHalos['snapnum'][L_indx])
    hf.create_dataset('Halo_z', data=LCHalos['redshift'][L_indx])
    hf.create_dataset('M200', data=LCHalos['M200'][L_indx])
    hf.create_dataset('Rvir', data=LCHalos['Rvir'][L_indx])
    hf.create_dataset('Rsca', data=LCHalos['Rsca'][L_indx])
    hf.create_dataset('Rvmax', data=LCHalos['Rvmax'][L_indx])
    hf.create_dataset('Vmax', data=LCHalos['Vmax'][L_indx])
    hf.create_dataset('HaloPosBox', data=LCHalos['HaloPosBox'][L_indx])
    hf.create_dataset('HaloPosLC', data=LCHalos['HaloPosLC'][L_indx])
    hf.create_dataset('VelDisp', data=LCHalos['VelDisp'][L_indx])
    hf.create_dataset('Ellip', data=LCHalos['Ellip'][L_indx])
    hf.create_dataset('Pa', data=LCHalos['Pa'][L_indx])
    hf.create_dataset('FOV', data=L_fov)
    hf.create_dataset('Src_ID', data=S_ID)
    hf.create_dataset('Src_z', data=S_z)
    hf.create_dataset('SrcPosSky', data=S_possky)
    #hf.create_dataset('SrcPosBox', data=S_posbos)
    #hf.create_dataset('Einstein_angle', data=A_E)
    #hf.create_dataset('Einstein_radius', data=R_E)
    hf.close()
