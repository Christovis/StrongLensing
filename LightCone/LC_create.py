from __future__ import division
import os, sys, logging
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import z_at_value
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
import h5py
import CosmoDist as cd
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/lib/')  # parent directory
from lc_tools import Lightcone as LC
from lc_tools import snapshot_redshifts
from lc_tools import Dc
import lc_randomize as LCR
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')  # parent directory
import readsnap
import readlensing as rf


def merge_dicts(x, y):
    """
    For Python <3.5
    Given two dicts., merge them into a new dict as a shallow copy.
    """
    z = x.copy()
    z.update(y)
    return z


# Length Unit
def check_length_unit(filename):
    data = open(sim_dir[sim]+'arepo/output_list_new.txt', 'r')
    Settings = data.readlines()
    for k in range(len(Settings)):
        if 'UnitLength_in_cm' in Settings[k].split():
            [dum, UnitLength, dum, dum, UnitLength] = Settings[k].split()
    return UnitLength


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


################################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')

################################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/StrongLensing/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)
#hf_name = 'Rockstar'
#hf_name = 'Subfind'

# Cosmological Constants
c = const.c.to_value('km/s')
# Light Cone parameters
zmax = 1.  # highest redshift
alpha = 0.522  # apex angle
observerpos = [0, 0, 0]
coneaxis = [1, 0, 0]  # unit vector
lc_number = 1

###########################################################################
# Iterate through Simulations
for sim in range(len(sim_dir)):
    logging.info('Create a Light-cone for: %s; with: %s' %
                 (sim_name[sim], hf_name))
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    
    # Cosmological Parameters
    snap_tot_num = 45
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    cosmo = LambdaCDM(H0=header.hubble*100,
                      Om0=header.omega_m,
                      Ode0=header.omega_l)
    
    # Load Subhalo properties for z=0
    box = LC(hf_dir[sim], sim_dir[sim], snap_tot_num, hf_name)

    #Redshift Steps of snapshots; past to present
    z_lcone = snapshot_redshifts(snapfile, snap_tot_num, zmax)
    # Comoving distance between z_lcone
    CoDi = Dc(z_lcone, box.unitlength, cosmo)
    # Interpolation fct. between comoving dist. and redshift
    reddistfunc = interp1d(CoDi, z_lcone, kind='cubic')
    CoDi = CoDi[1:]
    print(CoDi)

    # Define Observer Position
    # box.boxsize not correct if subpos in comoving distance!
    box.position_box_init(0)

    # Start to fill Light-Cone with Subhalos & Initialize prop_lc
    lc = None
    lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
    
    translation_z = 0
    boxmark = 32123
    snapshot = snap_tot_num
    # Walk through comoving distance until zmax
    for i in range(len(CoDi)):
        logging.info('  Load Light-Cone %f %%', CoDi[i]/CoDi[-1])
        limit = 0
        while limit == 0:
            if CoDi[i] > np.max(box.prop['pos_b'][:, 0]):
                translation_z += box.boxsize
                box.prop['pos_b'][:, 0] += box.boxsize
                # Add randomness
                box.prop['pos_b'] = LCR.inversion_s(box.prop['pos_b'], box.boxsize)
                box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], box.boxsize)
                box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'],
                                                   box.boxsize, translation_z)
            boxmaxdist = np.max(box.prop['pos_b'][:, 0])
            if CoDi[i] >= boxmaxdist:  #---------------------------------------
                # New box does not reach end of z-range')
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
                else:
                    lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
            elif CoDi[i] == CoDi[-1]:  #---------------------------------------
                sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                if len(sub_id[0]) != 0:
                    box_end = box.box_division(box, sub_id, hf_name)
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box_end, alpha, hf_name)
                else:
                    lc = box.fill_lightcone(lc, box_end, alpha, hf_name)
                limit = 1  # End of Light Cone
            else:  #-----------------------------------------------------------
                if boxmark == boxmaxdist:
                    # Next redshift
                    box = LC(hf_dir[sim], snapfile, snapshot-i, hf_name)
                    # Add randomness
                    box.position_box_init(translation_z)
                    box.prop['pos_b'] = LCR.inversion_s(box.prop['pos_b'], box.boxsize)
                    box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], box.boxsize)
                    box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'],
                                                       box.boxsize, translation_z)
                    print('box pos y',
                            np.min(box.prop['pos_b'][:, 1]),
                            np.max(box.prop['pos_b'][:, 1]))
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z3 = box.box_division(box, sub_id, hf_name)
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
                        else:
                            lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
                else:
                    #print('# SimBox extends over 2 redshifts')
                    # Present redshift
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z1 = box.box_division(box, sub_id, hf_name)
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z1, alpha, hf_name)
                        else:
                            lc = box.fill_lightcone(lc, box_z1, alpha, hf_name)
                    # Next redshift
                    box = LC(hf_dir[sim], snapfile, snapshot-i, hf_name)
                    # Add randomness
                    #box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], box.boxsize)
                    #box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'])
                    box.position_box_init(translation_z)
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], CoDi[i],
                                                  CoDi[i+1], 0)
                    if len(sub_id[0]) != 0:
                        box_z2 = box.box_division(box, sub_id, hf_name)
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z2, alpha, hf_name)
                        else:
                            lc = box.fill_lightcone(lc, box_z2, alpha, hf_name)
                boxmark = np.max(box.prop['pos_b'][:, 0])
                limit = 1  # End of Light Cone
        #if i == 8:
        #    break
    # Find the redshift of each selected halo
    sub_dist = np.sqrt(lc['pos_lc'][:, 0]**2 + \
                       lc['pos_lc'][:, 1]**2 + \
                       lc['pos_lc'][:, 2]**2)
    redshift_lc = [reddistfunc(dist) for dist in sub_dist]
    #redshift_lc = [z_at_value(cosmo.comoving_distance, dist*u.Mpc, zmax=1) for dist in sub_dist]
    # Write data to h5 file which can be read by LightCone_read.py
    # to analyse and plot
    outdir = '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/%s/' % (hf_name)
    fname = outdir+'LC_'+sim_name[sim]+'_%d.h5' % (lc_number)
    hf = h5py.File(fname, 'w')
    if hf_name == 'Subfind':
        hf.create_dataset('Halo_z', data=redshift_lc )
        hf.create_dataset('snapnum', data=lc['snapnum_box'])
        hf.create_dataset('HF_ID', data=lc['ID_box'])  # Halo Finder ID
        hf.create_dataset('HaloPosBox', data=lc['pos_box'])  #[Mpc]
        hf.create_dataset('HaloPosLC', data=lc['pos_lc'])  #[Mpc]
        hf.create_dataset('HaloVel', data=lc['vel_lc'])  #[Mpc]
        hf.create_dataset('Mvir', data=lc['Mvir_lc'])  #[Msun/h]
        hf.create_dataset('Vmax', data=lc['velmax_lc'])  #[km/s]
        hf.create_dataset('Vrms', data=lc['veldisp_lc'])  #[km/s]
        hf.create_dataset('Rvmax', data=lc['rvmax_lc'])  #[kpc]
        hf.create_dataset('Rhalfmass', data=lc['rhalfmass_lc'])  #[kpc]
    elif hf_name == 'Rockstar':
        hf.create_dataset('HF_ID', data=lc['ID_box'])  # Halo Finder ID
        hf.create_dataset('snapnum', data=lc['snapnum_box'])
        hf.create_dataset('Halo_z', data=redshift_lc )
        hf.create_dataset('Mvir', data=lc['Mvir_lc'])  #[Msun/h]
        hf.create_dataset('M200b', data=lc['M200b_lc'])  #[Msun/h]
        hf.create_dataset('HaloPosBox', data=lc['pos_box'])  #[Mpc]
        hf.create_dataset('HaloPosLC', data=lc['pos_lc'])  #[Mpc]
        hf.create_dataset('HaloVel', data=lc['vel_lc'])  #[Mpc]
        hf.create_dataset('Vmax', data=lc['velmax_lc'])  #[km/s]
        hf.create_dataset('Vrms', data=lc['veldisp_lc'])  #[km/s]
        hf.create_dataset('Rvir', data=lc['rvir_lc'])  #[kpc]
        hf.create_dataset('Rs', data=lc['rs_lc'])  #[kpc]
        hf.create_dataset('Rvmax', data=lc['rvmax_lc'])  #[kpc]
    hf.close()
