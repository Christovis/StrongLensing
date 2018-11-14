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
import lc_supernovae as SN
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import readsnap
import readlensing as rf

# Fixing random state for reproducibility
rnd.seed(1872)  #1872, 2944, 5912, 7638

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
years = 5e3  # [yr]

###############################################################################
# Iterate through Simulations
for sim in range(len(sim_dir)):
    logging.info('Seed SNIa in Light-cone for: %s' % sim_name[sim])
    # LightCone file for lens properties
    lc_file = lc_dir[sim]+hf_name+'/LC_'+sim_name[sim]+'_1.h5'
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
        dist_zr = np.asarray(dist_zr)
    except:
        dist_zr = (cosmo.comoving_distance(zr)).to_value('Mpc')
    # middle redshift & distnace
    zmid = np.linspace((zr[1]-zr[0])/2, zr[-2] + (zr[-1]-zr[-2])/2, 999)
    # Comoving distance between redshifts
    dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim) for j in range(len(zr)-1)]#[Mpc]

    # Number of SNeIa in time and Volume
    SNIa_number_density = SN.snia_number_density(
            years, zmid, agefunc, redfunc, **cosmosim)
    print('SNIa_formation_rate', SNIa_number_density)

    # Initialize Lists to save
    L_indx=[]; L_fov_arcsec=[]; L_fov_Mpc=[]; L_AE=[]; R_E=[]
    S_ID=np.zeros((1)); S_z=np.zeros((1))
    S_possky=np.zeros((1, 3)); S_mag=np.zeros((1))
    # Iterate over lenses
    print('There are %d sub-and halos in lc' % (len(LC['HF_ID'])))
    for i in range(len(LC['HF_ID'])):
        # Lens properties
        l_posbox = LC['HaloPosBox'][i, :]
        l_poslc = LC['HaloPosLC'][i, :]
        zl = LC['redshift'][i]
        lensdist = np.sqrt(l_poslc[0]**2 + \
                           l_poslc[1]**2 + \
                           l_poslc[2]**2)  # [Mpc/h]
        
        # Find starting redshift of lens
        indx = np.where(zmid > zl)[0]
        dist_sr = dist_zr[indx]
        #dist_sr = [dist_zr[idx] for idx in indx]
        dist_sr[-1] = dist_sr[-1]*0.9  # due to end-effects
        snia_number_density_sr = SNIa_number_density[indx]
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
            A_E = SN.Einstein_ring(LC['VelDisp'][i], c, zl,
                                   None, None)
            u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
            l_lenspos = np.linalg.norm(l_poslc)

            # Calculate Volume for SNeIa distribution
            fov_rad = A_E
            # length of distbet_p
            V, fov_Mpc = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)
        
        # Number of SNeIa in time and Volume
        snia_number = V*snia_number_density_sr[:-1]
        [dec, snia_number] = np.modf(snia_number)
        indx = np.where(dec > rnd.random())
        snia_number[indx] += 1

        # Check whether halo has SNIa at all
        if np.sum(snia_number) == 0.0:
            continue

        # Position SNIa in Light-Cone
        indx = np.nonzero(snia_number)
        [sid, sred, spossky] = SN.SNIa_position(i, indx, snia_number, dist_sr,
                                                u_lenspos, l_lenspos, fov_Mpc,
                                                S_ID, S_possky)
        S_ID = np.concatenate((S_ID, sid), axis=None)
        S_z = np.concatenate((S_z, sred), axis=None)
        S_possky = np.concatenate((S_possky, spossky))

        # SNIa abs. magnitude distribution
        smag = SN.SNIa_magnitudes(snia_number, zmid)
        S_mag = np.concatenate((S_mag, smag), axis=None)

        if np.count_nonzero(snia_number) != 0:
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
    
    S_ID=np.asarray(S_ID); S_z=np.asarray(S_z)
    S_possky=np.asarray(S_possky); S_mag=np.asarray(S_mag)
    L_AE=np.asarray(L_AE); R_E=np.asarray(R_E)
    L_indx=np.asarray(L_indx); 
    print('%d galaxies have SN Ia in background' % len(L_indx))
    #L_fov_arcsec = np.asarray(L_fov_arcsec)
    #L_fov_Mpc = np.asarray(L_fov_Mpc)
    L_fov_arcsec = [float(xx) for xx in L_fov_arcsec]
    L_fov_Mpc = [float(xx) for xx in L_fov_Mpc]
    hf = h5py.File(lc_dir[sim]+hf_name+'/LC_SN_'+sim_name[sim]+'_1.h5', 'w')
    if hf_name == 'Subfind':
        hf.create_dataset('HF_ID', data=LC['HF_ID'][L_indx])  # Halo Finder ID
        hf.create_dataset('LC_ID', data=L_indx)  # Light-Cone ID
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('Halo_z', data=LC['redshift'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_arcsec, L_fov_Mpc])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_possky)
        hf.create_dataset('SrcAbsMag', data=S_mag)
        hf.create_dataset('Einstein_angle', data=L_AE)  #[arcsec]
    elif hf_name == 'Rockstar':
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('HF_ID', data=LC['HF_ID'][L_indx])  # Halo Finder ID
        hf.create_dataset('LC_ID', data=L_indx)  # Lightcone ID
        hf.create_dataset('Halo_z', data=LC['redshift'][L_indx])
        hf.create_dataset('M200', data=LC['M200'][L_indx])
        hf.create_dataset('Rvir', data=LC['Rvir'][L_indx])
        #hf.create_dataset('Rsca', data=LC['Rsca'][L_indx])
        #hf.create_dataset('Rvmax', data=LC['Rvmax'][L_indx])
        #hf.create_dataset('Vmax', data=LC['Vmax'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        #hf.create_dataset('HaloVel', data=LC['HaloVel'][L_indx])
        hf.create_dataset('VelDisp', data=LC['VelDisp'][L_indx])
        #hf.create_dataset('Ellip', data=LC['Ellip'][L_indx])
        #hf.create_dataset('Pa', data=LC['Pa'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_arcsec, L_fov_Mpc])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_possky)
        hf.create_dataset('SrcAbsMag', data=S_mag)
        #hf.create_dataset('SrcPosBox', data=S_posbos)
        hf.create_dataset('Einstein_angle', data=L_AE)  #[rad]
        #hf.create_dataset('Einstein_radius', data=R_E)
    hf.close()
