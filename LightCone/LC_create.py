from __future__ import division
import os
import sys
import logging
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
sys.path.insert(0, '..')  # parent directory
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


################################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')

################################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)

hf_name = 'Rockstar'

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
    logging.info('Create a Light-cone for: %s', sim_name[sim])
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    
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
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    box = LC(hf_dir[sim], snapfile, snap_tot_num, hf_name, LengthUnit)
    boxlength = box.boxlength(box.prop['pos_b'])
    # Define Observer Position
    # boxlength not correct if subpos in comoving distance!
    box.position_box_init(0)

    # Start to fill Light-Cone with Subhalos & Initialize prop_lc
    lc = None
    lc = box.fill_lightcone(lc, box.prop, alpha)
    print(lc)

    translation_z = 0
    boxmark = 32123
    snapshot = snap_tot_num
    # Walk through comoving distance until zmax
    for i in range(len(CoDi)):
        logging.info('  Comoving Distance: %f', CoDi[i])
        limit = 0
        while limit == 0:
            if CoDi[i] > np.max(box.prop['pos_b'][:, 0]):
                print(' Add new box')
                translation_z += boxlength
                box.prop['pos_b'][:, 0] += boxlength
                # Add randomness
                box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], boxlength)
                box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'])
            boxmaxdist = np.max(box.prop['pos_b'][:, 0])
            if CoDi[i] >= boxmaxdist:  #---------------------------------------
                # New box does not reach end of z-range')
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box.prop, alpha)
                else:
                    lc = box.fill_lightcone(lc, box.prop, alpha)
            elif CoDi[i] == CoDi[-1]:  #---------------------------------------
                print('# End of Light Cone')
                sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                if len(sub_id[0]) != 0:
                    box_end = {'snapnum' :box.prop['snapnum'][sub_id],
                               'ID' : box.prop['ID'][sub_id],
                               'pos' : box.prop['pos'][sub_id],
                               'pos_b' : box.prop['pos_b'][sub_id],
                               'vel_b' : box.prop['vel_b'][sub_id],
                               'Mvir_b' : box.prop['Mvir_b'][sub_id],
                               'M200b_b' : box.prop['M200b_b'][sub_id],
                               'velmax_b' : box.prop['velmax_b'][sub_id],
                               'veldisp_b' : box.prop['veldisp_b'][sub_id],
                               'rvir_b' : box.prop['rvir_b'][sub_id],
                               'rs_b' : box.prop['rs_b'][sub_id],
                               'rvmax_b' : box.prop['rvmax_b'][sub_id],
                               'ellipse_b' : box.prop['ellipse_b'][sub_id],
                               'pa_b' : box.prop['pa_b'][sub_id]}
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box_end, alpha)
                else:
                    lc = box.fill_lightcone(lc, box_end, alpha)
                limit = 1  # End of Light Cone
            else:  #-----------------------------------------------------------
                print('# End of z-range reached')
                if boxmark == boxmaxdist:
                    print('# SimBox extends over more than 2 redshifts')
                    # Next redshift
                    prop_box, boxlength = prop.update_box(hf_dir[sim], snapfile,
                                                                snapshot-i, header,
                                                                hf_name, LengthUnit)
                    # Add randomness
                    #prop_box['pos_b'] = LCR.rotation_s(prop_box['pos_b'])
                    prop_box['pos_b'] = prop.position_box(prop_box['pos_b'],
                                                          boxlength,
                                                          translation_z, 0)
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z3 = {'snapnum' : box.prop['snapnum'][sub_id],
                                  'ID' : box.prop['ID'][sub_id],
                                  'pos' : box.prop['pos'][sub_id],
                                  'pos_b' : box.prop['pos_b'][sub_id],
                                  'vel_b' : box.prop['vel_b'][sub_id],
                                  'Mvir_b' : box.prop['Mvir_b'][sub_id],
                                  'M200b_b' : box.prop['M200b_b'][sub_id],
                                  'velmax_b' : box.prop['velmax_b'][sub_id],
                                  'veldisp_b' : box.prop['veldisp_b'][sub_id],
                                  'rvir_b' : box.prop['rvir_b'][sub_id],
                                  'rs_b' : box.prop['rs_b'][sub_id],
                                  'rvmax_b' : box.prop['rvmax_b'][sub_id],
                                  'ellipse_b' : box.prop['ellipse_b'][sub_id],
                                  'pa_b' : box.prop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box.prop, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box.prop, alpha)
                else:
                    print('# SimBox extends over 2 redshifts')
                    # Present redshift
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], 0, CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z1 = {'snapnum' : box.prop['snapnum'][sub_id],
                                          'ID' : box.prop['ID'][sub_id],
                                          'pos' : box.prop['pos'][sub_id],
                                          'pos_b' : box.prop['pos_b'][sub_id],
                                          'vel_b' : box.prop['vel_b'][sub_id],
                                          'Mvir_b' : box.prop['Mvir_b'][sub_id],
                                          'M200b_b' : box.prop['M200b_b'][sub_id],
                                          'velmax_b' : box.prop['velmax_b'][sub_id],
                                          'veldisp_b' : box.prop['veldisp_b'][sub_id],
                                          'rvir_b' : box.prop['rvir_b'][sub_id],
                                          'rs_b' : box.prop['rs_b'][sub_id],
                                          'rvmax_b' : box.prop['rvmax_b'][sub_id],
                                          'ellipse_b' : box.prop['ellipse_b'][sub_id],
                                          'pa_b' : box.prop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z1, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box_z1, alpha)
                    # Next redshift
                    box = LC(hf_dir[sim], snapfile, snapshot-i,
                             hf_name, LengthUnit)
                    boxlength = box.boxlength(box.prop['pos_b'])
                    # Add randomness
                    #box.prop['pos_b'] = LCR.translation_s(box.prop['pos_b'], boxlength)
                    #box.prop['pos_b'] = LCR.rotation_s(box.prop['pos_b'])
                    box.position_box_init(translation_z)
                    sub_id = box.find_sub_in_CoDi(box.prop['pos_b'], CoDi[i],
                                                  CoDi[i+1], 0)
                    if len(sub_id[0]) != 0:
                        box_z2 = {'snapnum' : box.prop['snapnum'][sub_id],
                                  'ID' : box.prop['ID'][sub_id],
                                  'pos' : box.prop['pos'][sub_id],
                                  'pos_b' : box.prop['pos_b'][sub_id],
                                  'vel_b' : box.prop['vel_b'][sub_id],
                                  'Mvir_b' : box.prop['Mvir_b'][sub_id],
                                  'M200b_b' : box.prop['M200b_b'][sub_id],
                                  'velmax_b' : box.prop['velmax_b'][sub_id],
                                  'veldisp_b' : box.prop['veldisp_b'][sub_id],
                                  'rvir_b' : box.prop['rvir_b'][sub_id],
                                  'rs_b' : box.prop['rs_b'][sub_id],
                                  'rvmax_b' : box.prop['rvmax_b'][sub_id],
                                  'ellipse_b' : box.prop['ellipse_b'][sub_id],
                                  'pa_b' : box.prop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z2, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box_z2, alpha)
                boxmark = np.max(box.prop['pos_b'][:, 0])
                limit = 1  # End of Light Cone
        boxlength = box.boxlength(box.prop['pos_b'])
    # Find the redshift of each selected halo
    sub_dist = np.sqrt(lc['pos_lc'][:, 0]**2 + \
                       lc['pos_lc'][:, 1]**2 + \
                       lc['pos_lc'][:, 2]**2)
    redshift_lc = [reddistfunc(dist) for dist in sub_dist]
    #redshift_lc = [z_at_value(cosmo.comoving_distance, dist*u.Mpc, zmax=1) for dist in sub_dist]
    #print('test 3', redshift_lc)
    #print(zs)
    # Write data to h5 file which can be read by LightCone_read.py
    # to analyse and plot
    outdir = '/cosma5/data/dp004/dc-beck3/LightCone/'
    hf = h5py.File(outdir+'XXX_'+sim_name[sim]+'.h5', 'w')
    hf.create_dataset('Halo_ID', data=lc['ID_box'])  # Rockstar ID
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
    hf.create_dataset('ellipticity', data=lc['ellipse_lc'])
    hf.create_dataset('position_angle', data=lc['pa_lc'])  #[]
    hf.close()
