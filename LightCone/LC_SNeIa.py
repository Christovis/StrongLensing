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
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/lib/')
import CosmoDist as cd
import lc_supernovae as SN
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import readlensing as rf

# Fixing random state for reproducibility
rnd.seed(1872)  #1872, 2944, 5912, 7638


###############################################################################
# Load Simulation Specifications
#LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
#sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

###############################################################################
# Define Supernova Distribution Parameters
c = float((const.c).to('km/s')*(u.second/u.km))
assert type(c) == float, "Incorrect type of c"
print(c)
# Time interval
years = 5e2  # [yr]

###############################################################################

def lc_seed_snia():
    args = {}
    args["simname"]     = sys.argv[1]
    args["simdir"]      = sys.argv[2]
    args["hfname"]      = sys.argv[3]
    args["hfdir"]       = sys.argv[4]
    args["lcfile"]      = sys.argv[5]
    args["lcnumber"]    = int(sys.argv[6])
    args["zmax"]        = int(sys.argv[7])
    args["outdir"]      = sys.argv[8]
    
    logging.info('Seed SNIa in Light-cone for: %s' % args["simname"])
    snap_tot_num = 45
    snapfile = args["simdir"]+'snapdir_%03d/snap_%03d'
    s = read_hdf5.snapshot(snap_tot_num, args["simdir"])
    header = s.header
    exp = np.floor(np.log10(np.abs(header.unitlength))).astype(int)
    if exp == 21:  # simulation in [kpc]
        scale = 1e-3
    elif exp == 23:  # simulation in [Mpc]
        scale = 1

    # Cosmological Parameters
    cosmo = LambdaCDM(H0=header.hubble*100,
                      Om0=header.omega_m,
                      Ode0=header.omega_l)
    
    # Load LightCone Contents
    LC = rf.LightCone_without_SN(args['lcfile'], args['hfname'])

    # convert conformal time to redshift
    cosmosim = {'omega_M_0' : 0.3089,
                'omega_lambda_0' : 0.6911,
                'omega_k_0' : 0.0,
                'h' :0.6779}
    agefunc, redfunc = cd.quick_age_function(20., 0., 0.001, 1, **cosmosim)

    # Redshift bins for SNeIa
    zmid, dist_zr, dist_bet = SN.redshift_division(
            args["zmax"], cosmo, header.unitlength, cosmosim)

    # Number of SNeIa in timezmid, dist_zr, dist_bet =  and Volume
    SNIa_number_density = SN.snia_number_density(
            years, zmid, agefunc, redfunc, **cosmosim)
    print('SNIa_formation_rate max', np.max(SNIa_number_density))

    # Initialize Lists to save
    L_indx=[]; L_fov_angle=[]; L_fov=[]; L_AE=[]; R_E=[]
    S_ID=np.zeros((1)); S_z=np.zeros((1))
    S_possky=np.zeros((1, 3)); S_mag=np.zeros((1))
    # Iterate over lenses
    print('There are %d sub-and halos in lc' % (len(LC['HF_ID'])))
    for i in range(len(LC['HF_ID'])):
        # Lens properties
        l_posbox = LC['HaloPosBox'][i, :]
        l_poslc = LC['HaloPosLC'][i, :]
        u_lenspos = np.asarray(l_poslc/np.linalg.norm(l_poslc))
        l_lenspos = np.linalg.norm(l_poslc)
        zl = LC['Halo_z'][i]
        lensdist = np.sqrt(l_poslc[0]**2 + \
                           l_poslc[1]**2 + \
                           l_poslc[2]**2)
        
        # Find starting redshift of lens
        indx = np.where(zmid > zl)[0]
        dist_sr = dist_zr[indx]
        #dist_sr = [dist_zr[idx] for idx in indx]
        dist_sr[-1] = dist_sr[-1]*0.9  # due to end-effects
        snia_number_density_sr = SNIa_number_density[indx]*scale  #[Mpc]
        distbet_p = [dist_bet[idx] for idx in indx[:-1]]
        # Calculate app. Einstein radius for point lense
        if args['hfname'] == 'Subfind':
            fov_rad = SN.Einstein_ring(LC['Vrms'][i], c, zl,
                                   None, None)
            # Calculate Volume for SNeIa distribution
            #fov_rad = 0.5*LC['Rhalfmass'][i]/cosmo.angular_diameter_distance(zl).to_value('kpc')
            V, fov_len = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)
        
        elif args['hfname'] == 'Rockstar':
            fov_rad = SN.Einstein_ring(LC['Vrms'][i], c, zl,
                                   None, None)
            # Calculate Volume for SNeIa distribution
            V, fov_len = SN.Einstein_Volume(fov_rad, dist_sr, distbet_p)
        
        #print('SNeIa will be distributed within %f[kpc] of halo which
        # has a Rhalfmass of %f[kpc]' \
        #        % (0.5*fov_Mpc[0]*1e3, LC['Rhalfmass'][i]))
        # Number of SNeIa in time and Volume
        snia_number = V*snia_number_density_sr[:-1]
        [dec, snia_number] = np.modf(snia_number)
        indx = np.where(dec > rnd.random())
        snia_number[indx] += 1
        print('max snia_number', np.max(snia_number))

        # Check whether halo has SNIa at all
        if np.count_nonzero(snia_number) == 0:
            continue

        # Position SNIa in Light-Cone
        indx = np.nonzero(snia_number)
        [sid, sred, spossky] = SN.SNIa_position(i, indx, snia_number, dist_sr,
                                                u_lenspos, l_lenspos, fov_len,
                                                S_ID, S_possky, header.unitlength)
        S_ID = np.concatenate((S_ID, sid), axis=None)
        S_z = np.concatenate((S_z, sred), axis=None)
        S_possky = np.concatenate((S_possky, spossky))

        # SNIa abs. magnitude distribution
        smag = SN.SNIa_magnitudes(snia_number, zmid)
        S_mag = np.concatenate((S_mag, smag), axis=None)

        fov_angle = (fov_rad*u.rad).to_value('arcsec')  # radians to arcsec
        assert type(fov_angle) == float, "Incorrect type of fov_angle"
        L_indx.append(i); L_fov_angle.append(fov_angle);
        L_fov.append(fov_len[0])
    
    S_ID=np.asarray(S_ID); S_z=np.asarray(S_z)
    S_possky=np.asarray(S_possky); S_mag=np.asarray(S_mag)
    R_E=np.asarray(R_E); L_indx=np.asarray(L_indx); 
    print('%d galaxies have SN Ia in background' % len(L_indx))
    L_fov_angle = [float(xx) for xx in L_fov_angle]
    L_fov = [float(xx) for xx in L_fov]
    hf = h5py.File(args['outdir']+'/LC_SN_'+args['simname']+'_%d.h5' % (args['lcnumber']), 'w')
    if args['hfname'] == 'Subfind':
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('HF_ID', data=LC['HF_ID'][L_indx])  # Halo Finder ID
        hf.create_dataset('LC_ID', data=L_indx)  # Light-Cone ID
        hf.create_dataset('Halo_z', data=LC['Halo_z'][L_indx])
        hf.create_dataset('M200', data=LC['M200'][L_indx])
        hf.create_dataset('Rhalfmass', data=LC['Rhalfmass'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        hf.create_dataset('VelDisp', data=LC['Vrms'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_angle, L_fov])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_possky)
        hf.create_dataset('SrcAbsMag', data=S_mag)
    elif args['hfname'] == 'Rockstar':
        hf.create_dataset('snapnum', data=LC['snapnum'][L_indx])
        hf.create_dataset('HF_ID', data=LC['HF_ID'][L_indx])  # Halo Finder ID
        hf.create_dataset('LC_ID', data=L_indx)  # Lightcone ID
        hf.create_dataset('Halo_z', data=LC['Halo_z'][L_indx])
        hf.create_dataset('M200', data=LC['M200'][L_indx])
        hf.create_dataset('Rvir', data=LC['Rvir'][L_indx])
        #hf.create_dataset('Rsca', data=LC['Rsca'][L_indx])
        #hf.create_dataset('Rvmax', data=LC['Rvmax'][L_indx])
        #hf.create_dataset('Vmax', data=LC['Vmax'][L_indx])
        hf.create_dataset('HaloPosBox', data=LC['HaloPosBox'][L_indx])
        hf.create_dataset('HaloPosLC', data=LC['HaloPosLC'][L_indx])
        #hf.create_dataset('HaloVel', data=LC['HaloVel'][L_indx])
        hf.create_dataset('VelDisp', data=LC['Vrms'][L_indx])
        #hf.create_dataset('Ellip', data=LC['Ellip'][L_indx])
        #hf.create_dataset('Pa', data=LC['Pa'][L_indx])
        hf.create_dataset('FOV', data=[L_fov_angle, L_fov])
        hf.create_dataset('Src_ID', data=S_ID)
        hf.create_dataset('Src_z', data=S_z)
        hf.create_dataset('SrcPosSky', data=S_possky)
        hf.create_dataset('SrcAbsMag', data=S_mag)
        #hf.create_dataset('SrcPosBox', data=S_posbos)
    hf.close()


if __name__ == '__main__':
    lc_seed_snia()
