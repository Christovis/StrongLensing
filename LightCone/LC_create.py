from __future__ import division
import os
import sys
import numpy as np
from astropy.cosmology import Planck15
from astropy import units as u
from astropy import constants as const
from scipy.interpolate import interp1d
import h5py
sys.path.append('/cosma5/data/dp004/dc-beck3')
import readsnap
import readsubf
import readlensing as rf
import CosmoDist as cd
from lc_tools import Lightcone as LC
import lc_randomize as LCR


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

###########################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)

# Cosmological Constants
c = const.c.to_value('km/s')
# Light Cone parameters
zmax = 1.  # highest redshift
alpha = 0.522  # apex angle
observerpos = [0, 0, 0]
coneaxis = [1, 0, 0]  # unit vector

###########################################################################
# Iterate through Simulations
for sim in range(len(sim_dir)):
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    # Load Simulation Header
    #snap_tot_num = len(open(sim_dir[sim]+'arepo/output_list_new.txt', 'r').readlines())-1
    # Length Unit
    LengthUnit = 'Mpc'
    #LengthUnit = check_length_unit(sim_dir[sim]+'arepo/param.txt')
    #snap_tot_num = 23
    snap_tot_num = 45
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    # Cosmological Parameters
    cosmo = {'omega_M_0' : header.omega_m,
            'omega_lambda_0' : header.omega_l,
            'omega_k_0' : 0.0,
            'h' : header.hubble}
    print('holliiii')
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
    CoDi = []
    for i in range(len(z_lcone)):
        CoDi.append(cd.comoving_distance(z_lcone[i], **cosmo))  #[Mpc]
    # CoDi.append(cd.comoving_distance(2., 0., **cosmo))
    CoDi = np.asarray(CoDi)
    # Interpolation fct. between comoving dist. and redshift
    reddistfunc = interp1d(CoDi, z_lcone, kind='cubic')
    CoDi = CoDi[1:]
    # Load Subhalo properties for z=0
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    box = LC(hf_dir[sim], snapfile, snap_tot_num, header, hf_name, LengthUnit)
    boxlength = box.boxlength(box.subprop['pos_b'])
    # Define Observer Position
    # boxlength not correct if subpos in comoving distance!
    box.position_box_init(0)

    # Start to fill Light-Cone with Subhalos & Initialize subprop_lc
    lc = None
    lc = box.fill_lightcone(lc, box.subprop, alpha)

    translation_z = 0
    boxmark = 32123
    snapshot = snap_tot_num
    # Walk through comoving distance until zmax
    for i in range(len(CoDi)):
        print('CoDi[i]', CoDi[i])
        limit = 0
        while limit == 0:
            if CoDi[i] > np.max(box.subprop['pos_b'][:, 0]):
                print(' Add new box')
                translation_z += boxlength
                box.subprop['pos_b'][:, 0] += boxlength
            # Add randomness
            box.subprop['pos_b'] = LCR.translation_s(box.subprop['pos_b'], boxlength)
            box.subprop['pos_b'] = LCR.rotation_s(box.subprop['pos_b'])
            boxmaxdist = np.max(box.subprop['pos_b'][:, 0])
            if CoDi[i] >= boxmaxdist:
                # New box does not reach end of z-range')
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box.subprop, alpha)
                else:
                    lc = box.fill_lightcone(lc, box.subprop, alpha)
            elif CoDi[i] == CoDi[-1]:
                # End of Light Cone
                sub_id = box.find_sub_in_CoDi(box.subprop['pos_b'], CoDi[i], 0)
                if len(sub_id[0]) != 0:
                    box_end = {'snapnum' :box.subprop['snapnum'][sub_id],
                               'ID' : box.subprop['ID'][sub_id],
                               'pos' : box.subprop['pos'][sub_id],
                               'pos_b' : box.subprop['pos_b'][sub_id],
                               'Mvir_b' : box.subprop['Mvir_b'][sub_id],
                               'M200b_b' : box.subprop['M200b_b'][sub_id],
                               'velmax_b' : box.subprop['velmax_b'][sub_id],
                               'veldisp_b' : box.subprop['veldisp_b'][sub_id],
                               'rvir_b' : box.subprop['rvir_b'][sub_id],
                               'rs_b' : box.subprop['rs_b'][sub_id],
                               'rvmax_b' : box.subprop['rvmax_b'][sub_id],
                               'ellipse_b' : box.subprop['ellipse_b'][sub_id],
                               'pa_b' : box.subprop['pa_b'][sub_id]}
                if lc == None:
                    # if lightcone is empty
                    lc = box.fill_lightcone(lc, box_end, alpha)
                else:
                    lc = box.fill_lightcone(lc, box_end, alpha)
                limit = 1  # End of Light Cone
            else:
                # End of z-range reached
                if boxmark == boxmaxdist:
                    # SimBox extends over more than 2 redshifts
                    # Next redshift
                    subprop_box, boxlength = SubProp.update_box(hf_dir[sim], snapfile,
                                                                snapshot-i, header,
                                                                hf_name, LengthUnit)
                    # Add randomness
                    subprop_box['pos_b'] = LCR.rotation_s(subprop_box['pos_b'])
                    subprop_box['pos_b'] = SubProp.position_box(subprop_box['pos_b'],
                                                                 boxlength,
                                                                 translation_z, 0)
                    sub_id = box.find_sub_in_CoDi(box.subprop['pos_b'], CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z3 = {'snapnum' : box.subprop['snapnum'][sub_id],
                                  'ID' : box.subprop['ID'][sub_id],
                                  'pos' : box.subprop['pos'][sub_id],
                                  'pos_b' : box.subprop['pos_b'][sub_id],
                                  'Mvir_b' : box.subprop['Mvir_b'][sub_id],
                                  'M200b_b' : box.subprop['M200b_b'][sub_id],
                                  'velmax_b' : box.subprop['velmax_b'][sub_id],
                                  'veldisp_b' : box.subprop['veldisp_b'][sub_id],
                                  'rvir_b' : box.subprop['rvir_b'][sub_id],
                                  'rs_b' : box.subprop['rs_b'][sub_id],
                                  'rvmax_b' : box.subprop['rvmax_b'][sub_id],
                                  'ellipse_b' : box.subprop['ellipse_b'][sub_id],
                                  'pa_b' : box.subprop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box.subprop, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box.subprop, alpha)
                else:
                    # SimBox extends over 2 redshifts
                    # Present redshift
                    sub_id = box.find_sub_in_CoDi(box.subprop['pos_b'], CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z1 = {'snapnum' : box.subprop['snapnum'][sub_id],
                                          'ID' : box.subprop['ID'][sub_id],
                                          'pos' : box.subprop['pos'][sub_id],
                                          'pos_b' : box.subprop['pos_b'][sub_id],
                                          'Mvir_b' : box.subprop['Mvir_b'][sub_id],
                                          'M200b_b' : box.subprop['M200b_b'][sub_id],
                                          'velmax_b' : box.subprop['velmax_b'][sub_id],
                                          'veldisp_b' : box.subprop['veldisp_b'][sub_id],
                                          'rvir_b' : box.subprop['rvir_b'][sub_id],
                                          'rs_b' : box.subprop['rs_b'][sub_id],
                                          'rvmax_b' : box.subprop['rvmax_b'][sub_id],
                                          'ellipse_b' : box.subprop['ellipse_b'][sub_id],
                                          'pa_b' : box.subprop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z1, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box_z1, alpha)
                    # Next redshift
                    box = LC(hf_dir[sim], snapfile, snapshot-i, header,
                             hf_name, LengthUnit)
                    boxlength = box.boxlength(box.subprop['pos_b'])
                    # Add randomness
                    box.subprop['pos_b'] = LCR.rotation_s(box.subprop['pos_b'])
                    box.position_box_init(translation_z)
                    sub_id = box.find_sub_in_CoDi(box.subprop['pos_b'], CoDi[i], 0)
                    if len(sub_id[0]) != 0:
                        box_z2 = {'snapnum' : box.subprop['snapnum'][sub_id],
                                  'ID' : box.subprop['ID'][sub_id],
                                  'pos' : box.subprop['pos'][sub_id],
                                  'pos_b' : box.subprop['pos_b'][sub_id],
                                  'Mvir_b' : box.subprop['Mvir_b'][sub_id],
                                  'M200b_b' : box.subprop['M200b_b'][sub_id],
                                  'velmax_b' : box.subprop['velmax_b'][sub_id],
                                  'veldisp_b' : box.subprop['veldisp_b'][sub_id],
                                  'rvir_b' : box.subprop['rvir_b'][sub_id],
                                  'rs_b' : box.subprop['rs_b'][sub_id],
                                  'rvmax_b' : box.subprop['rvmax_b'][sub_id],
                                  'ellipse_b' : box.subprop['ellipse_b'][sub_id],
                                  'pa_b' : box.subprop['pa_b'][sub_id]}
                        if lc == None:
                            # if lightcone is empty
                            lc = box.fill_lightcone(lc, box_z2, alpha)
                        else:
                            lc = box.fill_lightcone(lc, box_z2, alpha)
                boxmark = np.max(box.subprop['pos_b'][:, 0])
                limit = 1  # End of Light Cone
        boxlength = box.boxlength(box.subprop['pos_b'])
    # Find the redshift of each selected halo
    sub_dist = np.sqrt(lc['pos_lc'][:, 0]**2 + \
                       lc['pos_lc'][:, 1]**2 + \
                       lc['pos_lc'][:, 2]**2)
    redshift_lc = [reddistfunc(dist) for dist in sub_dist]
    # Write data to h5 file which can be read by LightCone_read.py
    # to analyse and plot
    outdir = '/cosma5/data/dp004/dc-beck3/lightcone/'
    hf = h5py.File(outdir+'TESTXXX_'+sim_name[sim]+'_Subhalos.h5', 'w')
    hf.create_dataset('ID', data=lc['ID_box'])
    hf.create_dataset('snapshot_number', data=lc['snapnum_box'])
    hf.create_dataset('redshift', data=redshift_lc )
    hf.create_dataset('Mvir', data=lc['Mvir_lc'])  #[Msun/h]
    hf.create_dataset('M200b', data=lc['M200b_lc'])  #[Msun/h]
    hf.create_dataset('position_box', data=lc['pos_box'])  #[Mpc]
    hf.create_dataset('position_lc', data=lc['pos_lc'])  #[Mpc]
    hf.create_dataset('maximum_circular_velocity', data=lc['velmax_lc'])  #[km/s]
    hf.create_dataset('velocity_dispersion', data=lc['veldisp_lc'])  #[km/s]
    hf.create_dataset('virial_radius', data=lc['rvir_lc'])  #[kpc]
    hf.create_dataset('scale_radius', data=lc['rs_lc'])  #[kpc]
    hf.create_dataset('vmax_radius', data=lc['rvmax_lc'])  #[kpc]
    hf.create_dataset('ellipticity', data=lc['ellipse_lc'])
    hf.create_dataset('position_angle', data=lc['pa_lc'])  #[]
    hf.close()
