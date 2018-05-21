#!/usr/bin/env python
import sys
import numpy as np
import random as rnd
import astropy
from astropy.cosmology import LambdaCDM
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import interp1d
import h5py
import CosmoDist as cd
#from Supernovae import SN_Type_Ia as SN
import lc_supernovae as SN
sys.path.insert(0, '..')
import readsnap
import readlensing as rf


###############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)


###############################################################################
# Define Supernova Distribution Parameters
c = const.c.to_value('km/s')
# Region around Lense...
# Time interval
years = 1e5  # [yr]
###############################################################################

# Iterate through Simulations
for sim in range(len(sim_dir)):
    # LightCone file for lens properties
    lc_file = lc_dir[sim]+'LC_'+sim_name[sim]+'.h5'
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d.0.hdf5'

    # Cosmological Parameters
    snap_tot_num = 45  # read output_list_new.txt
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    cosmo = LambdaCDM(H0=header.hubble*100,
                      Om0=header.omega_m,
                      Ode0=header.omega_l)
    # Load LightCone Contents
    LC = rf.LightCone_without_SN(lc_file, 'dictionary')
    # Select Halos
    LC = SN.select_halos(LC)
    print('Number of Halos: ', len(LC['M200']))

    # convert conformal time to redshift
    cosmosim = {'omega_M_0' : header.omega_m,
                'omega_lambda_0' : header.omega_l,
                'omega_k_0' : 0.0,
                'h' : header.hubble}
    agefunc, redfunc = cd.quick_age_function(20., 0., 0.001, 1, **cosmosim)

    # Redshift bins for SNeIa
    zr = np.linspace(0, 2, 1000)
    dist_zr = cosmo.comoving_distance(zr).to_value('Mpc')
    # middle redshift & distnace
    zmid = np.linspace((zr[1]-zr[0])/2, zr[-2] + (zr[-1]-zr[-2])/2, 999)
    #dist_mid = cosmo.comoving_distance(zmid).to_value('Mpc')
    # Comoving distance between redshifts
    dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim) for j in range(len(zr)-1)]#[Mpc]
    
    # Initialize Lists to save
    L_indx=[]; L_fov=[]; S_ID=[]; S_z=[]; S_possky=[]; L_AE=[]; R_E=[]

    # Iterate over lenses
    for i in range(len(LC['Halo_ID'])):
        # Lens properties
        l_posbox = LC['HaloPosBox'][i, :]
        l_poslc = LC['HaloPosLC'][i, :]
        zl = LC['redshift'][i]
        lensdist = np.sqrt(l_poslc[0]**2 + l_poslc[1]**2 + l_poslc[2]**2)  # [Mpc/h]
        
        # Find starting redshift of Box
        indx = np.where(zr > LC['redshift'][i])[0]
        dist_sr = [dist_zr[idx] for idx in indx]
        #distmid_p = [dist_mid[idx] for idx in indx[:-2]]
        distbet_p = [dist_bet[idx] for idx in indx[:-1]]
        # Calculate app. Einstein radius for point lense
        A_E = SN.Einstein_ring(LC['VelDisp'][i], c, LC['redshift'][i],
                               lensdist, None, 'rad')
        u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
        l_lenspos = np.linalg.norm(l_poslc)

        # Calculate Volume for SNeIa distribution
        fov_rad = 0.2*A_E  # convert radians to arcsec
        V, fov_Mpc = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)

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
                zs = z_at_value(cosmo.comoving_distance, srcdist*u.Mpc, zmax=2)
                # max. distance equa to re
                sposx = srcdist  #[Mpc]
                charge = 1 if rnd.random() < 0.5 else -1
                sposy = charge*rnd.random()*fov_Mpc[y]  #[Mpc]
                charge = 1 if rnd.random() < 0.5 else -1
                sposz = charge*rnd.random()*fov_Mpc[y]  #[Mpc]
                # Calculate Einstein Radius for Source
                Ds = cosmo.angular_diameter_distance(zs)  #[Mpc]
                Dls = cosmo.angular_diameter_distance_z1z2(zl, zs)  #[Mpc]
                # only valid for point lense... not accurate
                #[thetaE, radiusE] = SN.Einstein_ring(LC['VelDisp'][i],
                #                     c, zl, Dls, Ds, 'kpc')  #[rad],[kpc]
                # Write glafic file
                S_ID.append(i);         S_z.append(zs)
                S_possky.append([sposx, sposy, sposz])
                #A_E.append(A_E)#;     R_E.append(radiusE)
            checksrc = 1
        if checksrc == 1:
            # Write glafic file
            fov = (fov_rad*u.rad).to_value('arcsec')  # convert radians to arcsec
            A_E = (A_E*u.rad).to_value('arcsec')  # convert radians to arcsec
            L_indx.append(i);   L_fov.append(fov); L_AE.append(A_E)
    
    S_ID = np.asarray(S_ID);       S_z = np.asarray(S_z)
    S_possky = np.asarray(S_possky)
    L_AE = np.asarray(L_AE);         R_E = np.asarray(R_E)
    L_indx = np.asarray(L_indx);   L_fov = np.asarray(L_fov)
    print('->', len(L_indx), 'number of lenses')
    hf = h5py.File(lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5', 'w')
    hf.create_dataset('Halo_Rockstar_ID', data=LC['Halo_ID'][L_indx])  # not Rockstar ID
    hf.create_dataset('Halo_ID', data=L_indx)  # not Rockstar ID
    hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
    hf.create_dataset('Halo_z', data=LC['redshift'][L_indx])
    hf.create_dataset('M200', data=LC['M200'][L_indx])
    hf.create_dataset('Rvir', data=LC['Rvir'][L_indx])
    hf.create_dataset('Rsca', data=LC['Rsca'][L_indx])
    hf.create_dataset('Rvmax', data=LC['Rvmax'][L_indx])
    hf.create_dataset('Vmax', data=LC['Vmax'][L_indx])
    hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
    hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
    hf.create_dataset('VelDisp', data=LC['VelDisp'][L_indx])
    hf.create_dataset('Ellip', data=LC['Ellip'][L_indx])
    hf.create_dataset('Pa', data=LC['Pa'][L_indx])
    hf.create_dataset('FOV', data=L_fov)
    hf.create_dataset('Src_ID', data=S_ID)
    hf.create_dataset('Src_z', data=S_z)
    hf.create_dataset('SrcPosSky', data=S_possky)
    #hf.create_dataset('SrcPosBox', data=S_posbos)
    hf.create_dataset('Einstein_angle', data=L_AE)  #[rad]
    #hf.create_dataset('Einstein_radius', data=R_E)
    hf.close()
