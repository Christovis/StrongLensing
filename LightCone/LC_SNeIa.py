#!/usr/bin/env python
import sys, logging
import time
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
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/')
import readsnap
import readlensing as rf

# Fixing random state for reproducibility
rnd.seed(10)

###############################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')

###############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

###############################################################################
# Define Supernova Distribution Parameters
try:
    c = float((const.c).to('km/s')*(u.second/u.km))
except:
    c = (const.c).to_value('km/s')
assert type(c) == float, "Incorrect type of c"
# Time interval
years = 1e3  # [yr]

###############################################################################
# Iterate through Simulations
for sim in range(len(sim_dir)):
    logging.info('Seed SNIa in Light-cone for: %s' % sim_name[sim])
    # LightCone file for lens properties
    lc_file = lc_dir[sim]+hf_name+'/LC_'+sim_name[sim]+'_2.h5'
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'

    # Cosmological Parameters
    snap_tot_num = 45  # read output_list_new.txt
    cosmo = LambdaCDM(H0=0.6774*100,
                      Om0=0.3089,
                      Ode0=0.6911)
    # Load LightCone Contents
    LC = rf.LightCone_without_SN(lc_file, hf_name)
    # Select Halos
    LC = SN.select_halos(LC, hf_name)

    # convert conformal time to redshift
    cosmosim = {'omega_M_0' : 0.3089,
                'omega_lambda_0' : 0.6911,
                'omega_k_0' : 0.0,
                'h' :0.6779}
    agefunc, redfunc = cd.quick_age_function(20., 0., 0.001, 1, **cosmosim)

    # Redshift bins for SNeIa
    zr = np.linspace(0, 2, 1000)
    try:
        dist_zr = ((cosmo.comoving_distance(zr)).to('Mpc')*1/u.Mpc).astype(np.float)
        dist_zr = [float(xx) for xx in dist_zr]  # convert Quantity to np.list(float)
    except:
        dist_zr = (cosmo.comoving_distance(zr)).to_value('Mpc')
    assert type(dist_zr) == list, "Incorrect type of dist_zr"
    # middle redshift & distnace
    zmid = np.linspace((zr[1]-zr[0])/2, zr[-2] + (zr[-1]-zr[-2])/2, 999)
    # Comoving distance between redshifts
    dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim) for j in range(len(zr)-1)]#[Mpc]

    # Initialize Lists to save
    L_indx=[]; L_fov_arcsec=[]; L_fov_Mpc=[]
    S_ID=[]; S_z=[]; S_possky=[]; L_AE=[]; R_E=[]

    S_ID=np.zeros((1)); S_z=np.zeros((1)); S_skypos=np.zeros((1,3))
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
        distbet_p = [dist_bet[idx] for idx in indx[:-1]]
        # Calculate app. Einstein radius for point lense
        if hf_name == 'Subfind':
            A_E = SN.Einstein_ring(LC['VelDisp'][i]+10, c, LC['redshift'][i],
                                   lensdist, None, 'rad')
            u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
            l_lenspos = np.linalg.norm(l_poslc)

            # Calculate Volume for SNeIa distribution
            fov_rad = A_E*0.5
            V, fov_Mpc = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)
        elif hf_name == 'Rockstar':
            A_E = SN.Einstein_ring(LC['VelDisp'][i], c, LC['redshift'][i],
                                   None, None)
            u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
            l_lenspos = np.linalg.norm(l_poslc)

            # Calculate Volume for SNeIa distribution
            fov_rad = A_E  #0.5
            V, fov_Mpc = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)
        
        # Number of SNeIa in time and Volume
        SNIa_num = SN.SNIa_distr(years, V, zmid, agefunc, redfunc, **cosmosim)
        [dec, SNIa_num] = np.modf(SNIa_num)
        rand = rnd.random()
        indx = np.where(dec > rnd.random())
        SNIa_num[indx] += 1

        if np.sum(SNIa_num) == 0.0:
            continue
        else:
            pass

        # Position SNIa in Light-Cone
        ## Find Volums wih Supernovae
        indx = np.nonzero(SNIa_num)
        ## Place SNIa in Light-Cone
        [sid, sx, sy, sz] = cf.call_seed_snia(i, indx, SNIa_num, dist_sr,
                                               u_lenspos, l_lenspos, fov_Mpc)
        ## most time consuming
        sred = [z_at_value(cosmo.comoving_distance, xx*u.Mpc, zmax=2) for xx in sx]
        S_ID = np.concatenate((S_ID, sid), axis=None)
        S_z = np.concatenate((S_z, sred), axis=None)
        sskypos = np.stack((sx, sy, sz)).transpose()
        S_skypos = np.concatenate((S_skypos, sskypos))


        # SNIa abs. magnitude distribution
        SN.SNIa_magnitudes(SNIa_num)
        M_SNIa = np.random.normal(M_SNIa, sigma_SNIa, SNIa_num)

        if np.count_nonzero(SNIa_num) != 0:
            # Write glafic file
            try:
                fov_arcsec = (fov_rad*u.rad).to_value('arcsec')  # radians to arcsec
                A_E = (A_E*u.rad).to_value('arcsec')  # convert radians to arcsec
            except:
                fov_arcsec = float((fov_rad*u.rad).to('arcsec')*1/u.arcsec)
                A_E = float((A_E*u.rad).to('arcsec')*1/u.arcsec)
            assert type(fov_arcsec) == float, "Incorrect type of fov_arcsec"
            assert type(A_E) == float, "Incorrect type of A_E"
            L_indx.append(i); L_fov_arcsec.append(fov_arcsec);
            L_fov_Mpc.append(fov_Mpc[0]); L_AE.append(A_E)
    
    S_ID = np.asarray(S_ID); S_z = np.asarray(S_z)
    S_skypos = np.asarray(S_skypos)
    L_AE = np.asarray(L_AE); R_E = np.asarray(R_E)
    L_indx = np.asarray(L_indx); 
    #L_fov_arcsec = np.asarray(L_fov_arcsec)
    #L_fov_Mpc = np.asarray(L_fov_Mpc)
    L_fov_arcsec = [float(xx) for xx in L_fov_arcsec]
    L_fov_Mpc = [float(xx) for xx in L_fov_Mpc]
    hf = h5py.File(lc_dir[sim]+hf_name+'/LC_SN_'+sim_name[sim]+'_rndseed31.h5', 'w')
    if hf_name == 'Subfind':
        hf.create_dataset('Halo_Subfind_ID', data=LC['Halo_ID'][L_indx])  # Rockstar ID
        hf.create_dataset('Halo_ID', data=L_indx)  # Light-Cone ID
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('Halo_z', data=LC['redshift'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_arcsec, L_fov_Mpc])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_skypos)
        hf.create_dataset('Einstein_angle', data=L_AE)  #[arcsec]
    elif hf_name == 'Rockstar':
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('Halo_Rockstar_ID', data=LC['Halo_ID'][L_indx])  # Rockstar ID
        hf.create_dataset('Halo_ID', data=L_indx)  # not Rockstar ID
        hf.create_dataset('Halo_z', data=LC['redshift'][L_indx])
        #hf.create_dataset('M200', data=LC['M200'][L_indx])
        hf.create_dataset('Rvir', data=LC['Rvir'][L_indx])
        #hf.create_dataset('Rsca', data=LC['Rsca'][L_indx])
        #hf.create_dataset('Rvmax', data=LC['Rvmax'][L_indx])
        #hf.create_dataset('Vmax', data=LC['Vmax'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        #hf.create_dataset('HaloVel', data=LC['HaloVel'][L_indx])
        #hf.create_dataset('VelDisp', data=LC['VelDisp'][L_indx])
        #hf.create_dataset('Ellip', data=LC['Ellip'][L_indx])
        #hf.create_dataset('Pa', data=LC['Pa'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_arcsec, L_fov_Mpc])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_skypos)
        #hf.create_dataset('SrcPosBox', data=S_posbos)
        hf.create_dataset('Einstein_angle', data=L_AE)  #[rad]
        #hf.create_dataset('Einstein_radius', data=R_E)
    hf.close()
