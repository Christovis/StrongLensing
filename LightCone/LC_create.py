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
from lc_tools import Lightcone as LC
import lc_randomize as LCR
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')  # parent directory
import readsnap
import readsubf
import readlensing as rf
import imp


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
BoxSize = 62  #[Mpc]
c = const.c.to_value('km/s')
# Light Cone parameters
zmax = 1.  # highest redshift
alpha = 0.522  # apex angle
observerpos = [0, 0, 0]
coneaxis = [1, 0, 0]  # unit vector

###########################################################################
# Iterate through Simulations
for sim in range(len(sim_dir)):
    logging.info('Create a Light-cone for: %s; with: %s' %
                 (sim_name[sim], hf_name))
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    
    if hf_name == 'Subfind':
        LengthUnit = 'kpc'
    elif hf_name == 'Rockstar':
        LengthUnit = 'Mpc'
    # Cosmological Parameters
    snap_tot_num = 45
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    cosmo = LambdaCDM(H0=header.hubble*100,
                      Om0=header.omega_m,
                      Ode0=header.omega_l)
    #Redshift Steps; past to present
    z_sim = []
    for i in range( snap_tot_num, -1, -1):
        header = readsnap.snapshot_header(snapfile % (i, i))
        if header.redshift > zmax:
            break
        else:
            z_sim.append(header.redshift)
    z_lcone = [z_sim[i] + (z_sim[i+1] - z_sim[i])/2 for i in range(len(z_sim)-1)]
    z_lcone.append(zmax)
    z_lcone = [0] + z_lcone

    # Comoving distance between snapshot redshifts
    CoDi = cosmo.comoving_distance(z_lcone).to_value('Mpc')
    # Interpolation fct. between comoving dist. and redshift
    reddistfunc = interp1d(CoDi, z_lcone, kind='cubic')
    CoDi = CoDi[1:]
    # Load Subhalo properties for z=0
    #snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    box = LC(hf_dir[sim], snapfile, snap_tot_num, hf_name, LengthUnit)
    boxlength = 62  #[Mpc] box.boxlength(box.prop['pos_b'])
    # Define Observer Position
    # boxlength not correct if subpos in comoving distance!
    box.position_box_init(0)

    # Start to fill Light-Cone with Subhalos & Initialize prop_lc
    lc = None
    lc = box.fill_lightcone(lc, box.prop, alpha, hf_name)
    
    translation_z = 0
    boxmark = 32123
    snapshot = snap_tot_num
    # Walk through comoving distance until zmax
    for i in range(len(CoDi)):
        logging.info('  Comoving Distance: %f', CoDi[i])
        limit = 0
        while limit == 0:
            if CoDi[i] > np.max(box.prop['pos_b'][:, 0]):
                translation_z += boxlength
                box.prop['pos_b'][:, 0] += boxlength
                # Add randomness
                box.prop['pos_b'] = LCR.inversion_s(box.prop['pos_b'], boxlength)
                box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], boxlength)
                box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'],
                                                   boxlength, translation_z)
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
                    box = LC(hf_dir[sim], snapfile, snapshot-i,
                             hf_name, LengthUnit)
                    # Add randomness
                    box.position_box_init(translation_z)
                    box.prop['pos_b'] = LCR.inversion_s(box.prop['pos_b'], boxlength)
                    box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], boxlength)
                    box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'],
                                                       boxlength, translation_z)
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
                    box = LC(hf_dir[sim], snapfile, snapshot-i,
                             hf_name, LengthUnit)
                    boxlength = 62  #[Mpc] box.boxlength(box.prop['pos_b'])
                    # Add randomness
                    #box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], boxlength)
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
        boxlength = 62  #[Mpc] box.boxlength(box.prop['pos_b'])
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
    outdir = '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/Rockstar/'
    hf = h5py.File(outdir+'LC_'+sim_name[sim]+'_2.h5', 'w')
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
