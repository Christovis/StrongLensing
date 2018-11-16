import os, sys
import pandas as pd
import numpy as np
import CosmoDist as cd
from astropy import units as u
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import readsnap
import read_hdf5


# Disable
def blockprint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enableprint():
    sys.stdout = sys.__stdout__


def snapshot_redshifts(snapfile, snap_tot_num, zmax):
    """
    Input:
        snapfile: snapshot directory
        snap_tot_num: total number of snapshots
        zmax: maximum redshift of lightcone
    Output:
        z_lcone: mean redshift between two snapshot redshifts
    """
    z_sim = []
    for i in range(snap_tot_num, -1, -1):
        header = readsnap.snapshot_header(snapfile % (i, i))
        if header.redshift > zmax:
            break
        else:
            z_sim.append(header.redshift)
    z_lcone = [z_sim[i] + (z_sim[i+1] - z_sim[i])/2 for i in range(len(z_sim)-1)]
    z_lcone.append(zmax)
    z_lcone = [0] + z_lcone
    return z_lcone


def Dc(z, unit, cosmo):
    exp = np.floor(np.log10(np.abs(unit))).astype(int)
    if exp == 21:  # simulation in [kpc]
        CoDi = cosmo.comoving_distance(z).to_value('kpc')
    elif exp == 23:  #simulation in [Mpc]
        CoDi = cosmo.comoving_distance(z).to_value('Mpc')
    else:
        raise Exception('Dont know this unit ->', unit)
    return CoDi


class Lightcone():
    def __init__(self, hfdir, snapdir, snapnum, halo_finder):
        """
        Load Subhalo properties of snapshots into Dictionary
        Use:
            box = Lightcone(snapshot_file_name)

        Input:
            hfdir: halo-finder directory
            snap_dir: snapshot directory
            snapnum: snapshot number
            halo_finder: Subfind, Rockstar or AHF
        
        Output:
            prop_box: dictionary with subhalo properties
            boxlength: side length of simulated box in units used in simulation
                       (e.g. Mpc/h or kpc/h)
        """
        if halo_finder == 'Subfind':
            blockprint()
            s = read_hdf5.snapshot(snapnum, hfdir)
            s.group_catalog(["SubhaloIDMostbound", "SubhaloPos", "SubhaloVel",
                             "SubhaloMass", "SubhaloVmax", "SubhaloVelDisp",
                             "SubhaloVmaxRad", "SubhaloHalfmassRad"])
            self.unitlength = s.header.unitlength
            self.boxsize = s.header.boxsize
            enableprint()
            indx = np.where(s.cat["SubhaloMass"] > 10**11)[0]
            #sub_id = self.subhalo_selection(pos, data['Mvir'], data['Vrms'], 0, 0)
            prop_box = {'snapnum' : np.ones(len(s.cat['SubhaloIDMostbound'][indx])) * \
                                    snapnum,
                        'ID' : s.cat['SubhaloIDMostbound'][indx],
                        'pos' : s.cat['SubhaloPos'][indx, :],
                        'pos_b' : s.cat['SubhaloPos'][indx, :],
                        'vel_b' : s.cat['SubhaloVel'][indx, :],
                        'Mvir_b' : s.cat['SubhaloMass'][indx],
                        'velmax_b' : s.cat['SubhaloVmax'][indx],
                        'veldisp_b' : s.cat['SubhaloVelDisp'][indx],
                        'rvmax_b' : s.cat['SubhaloVmaxRad'][indx],
                        'rhalfmass_b' : s.cat['SubhaloHalfmassRad'][indx]}
            self.prop = prop_box
        elif halo_finder == 'Rockstar':
            hf_dir = hfdir + 'halos_%d.dat' % snapnum
            data = pd.read_csv(hf_dir, sep='\s+', skiprows=np.arange(1, 16))
            #if LengthUnit == 'kpc':
            #    pos = pd.concat([data['X']*1e-3, data['Y']*1e-3, data['Z']*1e-3], axis=1)
            #else:
            #    pos = pd.concat([data['X'], data['Y'], data['Z']], axis=1)
            self.boxsize = 62  #[Mpc]
            vel = pd.concat([data['VX'], data['VY'], data['VZ']], axis=1)
            sub_id = self.subhalo_selection(pos, data['Mvir'], data['Vrms'], 0, 0)
            prop_box = {'snapnum' : snapnum*np.ones(len(sub_id[0])),
                        'ID' : data['#ID'].values[sub_id][0],
                        'pos' : pos.values[sub_id][0],
                        'pos_b' : pos.values[sub_id][0],
                        'vel_b' : vel.values[sub_id][0],
                        'Mvir_b' : data['Mvir'].values[sub_id][0],
                        'M200b_b' : data['M200b'].values[sub_id][0],
                        'velmax_b' : data['Vmax'].values[sub_id][0],
                        'veldisp_b' : data['Vrms'].values[sub_id][0],
                        'rvir_b' : data['Rvir'].values[sub_id][0],
                        'rs_b' : data['Rs'].values[sub_id][0],
                        'rvmax_b' : data['Rvmax'].values[sub_id][0],
                        'Halfmass_Radius' : data['Halfmass_Radius'].values[sub_id][0]}
            self.prop= prop_box
        elif halo_finder == 'AHF':
            pass
            
        
    def update_box(self, hfdir, snap_dir, snapnum, halo_finder, LengthUnit):
        hf_dir = hfdir + 'halos_%d.dat' % snapnum
        data = pd.read_csv(hf_dir, sep='\s+', skiprows=np.arange(1, 16))
        if LengthUnit == 'kpc':
            pos = pd.concat([data['X']*1e-3, data['Y']*1e-3, data['Z']*1e-3], axis=1)
        else:
            pos = pd.concat([data['X'], data['Y'], data['Z']], axis=1)
        vel = pd.concat([data['VX'], data['VY'], data['VZ']], axis=1)
        sub_id = self.subhalo_selection(pos, data['Mvir'], data['Vrms'], 0, 0)
        prop_box = {'snapnum' : snapnum*np.ones(len(sub_id[0])),
                    'ID' : data['#ID'].values[sub_id][0],
                    'pos' : pos.values[sub_id][0],
                    'pos_b' : pos.values[sub_id][0],
                    'vel_b' : vel.values[sub_id][0],
                    'Mvir_b' : data['Mvir'].values[sub_id][0],
                    'M200b_b' : data['M200b'].values[sub_id][0],
                    'velmax_b' : data['Vmax'].values[sub_id][0],
                    'veldisp_b' : data['Vrms'].values[sub_id][0],
                    'rvir_b' : data['Rvir'].values[sub_id][0],
                    'rs_b' : data['Rs'].values[sub_id][0],
                    'rvmax_b' : data['Rvmax'].values[sub_id][0],
                    'Halfmass_Radius' : data['Halfmass_Radius'].values[sub_id][0]}
        self.prop= prop_box



    def subhalo_selection(self, pos, mass, veldisp, ellipse, pa):
        # Define subhalo selection
        sub_id1 = np.where(mass > 10**11)  # defined with Baojiu & Christian
        #sub_id2 = np.where(veldisp >= 160)  # arXiv:1507.07937
        #sub_id = set(sub_id1[0]) & set(sub_id2[0])
        #sub_id = np.array(list(sub_id))
        sub_id = np.array(sub_id1)
        return sub_id


    def update_lc(self, indx, lc, box, hfname):
        if hfname == 'Subfind':
            if lc == None:
                lc = {'snapnum_box' : box['snapnum'][indx],
                      'ID_box' : box['ID'][indx],
                      'pos_box' : box['pos'][indx],
                      'pos_lc' : box['pos_b'][indx], 
                      'vel_lc' : box['vel_b'][indx], 
                      'Mvir_lc' : box['Mvir_b'][indx], 
                      'velmax_lc' : box['velmax_b'][indx],
                      'veldisp_lc' : box['veldisp_b'][indx],
                      'rvmax_lc' : box['rvmax_b'][indx],
                      'rhalfmass_lc' : box['rhalfmass_b'][indx]}
            else:
                lc['snapnum_box'] = np.concatenate((lc['snapnum_box'],
                                            box['snapnum'][indx]),
                                            axis=0)
                lc['ID_box'] = np.concatenate((lc['ID_box'],
                                            box['ID'][indx]),
                                            axis=0)
                lc['pos_box'] = np.concatenate((lc['pos_box'],
                                                box['pos'][indx]),
                                                axis=0)
                lc['pos_lc'] = np.concatenate((lc['pos_lc'],
                                                box['pos_b'][indx]),
                                                axis=0)
                lc['vel_lc'] = np.concatenate((lc['vel_lc'],
                                               box['vel_b'][indx]),
                                               axis=0)
                lc['Mvir_lc'] = np.concatenate((lc['Mvir_lc'],
                                                box['Mvir_b'][indx]),
                                                axis=0)
                lc['velmax_lc'] = np.concatenate((lc['velmax_lc'],
                                                box['velmax_b'][indx]),
                                                axis=0)
                lc['veldisp_lc'] = np.concatenate((lc['veldisp_lc'],
                                                box['veldisp_b'][indx]),
                                                axis=0)
                lc['rvmax_lc'] = np.concatenate((lc['rvmax_lc'],
                                                box['rvmax_b'][indx]),
                                                axis=0)
                lc['rhalfmass_lc'] = np.concatenate((lc['rhalfmass_lc'],
                                                    box['rhalfmass_b'][indx]),
                                                    axis=0)
        elif hfname == 'Rockstar':
            if lc == None:
                lc = {'snapnum_box' : box['snapnum'][indx],
                      'ID_box' : box['ID'][indx],
                      'pos_box' : box['pos'][indx],
                      'pos_lc' : box['pos_b'][indx], 
                      'vel_lc' : box['vel_b'][indx], 
                      'Mvir_lc' : box['Mvir_b'][indx], 
                      'M200b_lc' : box['M200b_b'][indx], 
                      'velmax_lc' : box['velmax_b'][indx],
                      'veldisp_lc' : box['veldisp_b'][indx],
                      'rvir_lc' : box['rvir_b'][indx],
                      'rs_lc' : box['rs_b'][indx],
                      'rvmax_lc' : box['rvmax_b'][indx]}
            else:
                lc['snapnum_box'] = np.concatenate((lc['snapnum_box'],
                                            box['snapnum'][indx]),
                                            axis=0)
                lc['ID_box'] = np.concatenate((lc['ID_box'],
                                            box['ID'][indx]),
                                            axis=0)
                lc['pos_box'] = np.concatenate((lc['pos_box'],
                                                box['pos'][indx]),
                                                axis=0)
                lc['pos_lc'] = np.concatenate((lc['pos_lc'],
                                                box['pos_b'][indx]),
                                                axis=0)
                lc['vel_lc'] = np.concatenate((lc['vel_lc'],
                                               box['vel_b'][indx]),
                                               axis=0)
                lc['Mvir_lc'] = np.concatenate((lc['Mvir_lc'],
                                                box['Mvir_b'][indx]),
                                                axis=0)
                lc['M200b_lc'] = np.concatenate((lc['M200b_lc'],
                                                box['M200b_b'][indx]),
                                                axis=0)
                lc['velmax_lc'] = np.concatenate((lc['velmax_lc'],
                                                box['velmax_b'][indx]),
                                                axis=0)
                lc['veldisp_lc'] = np.concatenate((lc['veldisp_lc'],
                                                box['veldisp_b'][indx]),
                                                axis=0)
                lc['rvir_lc'] = np.concatenate((lc['rvir_lc'],
                                                box['rvir_b'][indx]),
                                                axis=0)
                lc['rs_lc'] = np.concatenate((lc['rs_lc'],
                                                box['rs_b'][indx]),
                                                axis=0)
                lc['rvmax_lc'] = np.concatenate((lc['rvmax_lc'],
                                            box['rvmax_b'][indx]),
                                            axis=0)
                #del subprop['snapnum'], ['ID']
                #del subprop['pos'], subprop['pos_b'], subprop['Mvir_b'],
                #del subprop['M200b_b']
                #del subprop['velmax_b'], subprop['veldisp_b'], subprop['rvir_b']
                #del subprop['rs_b'], subprop['ellipse_b'], subprop['pa_b']
                #print('LC length: ', np.max(lc['pos_lc'][:, 0]))
        return lc


    def box_division(self, box, sub_id, hfname):
        if hfname == 'Subfind':
            boxpart = {'snapnum' : box.prop['snapnum'][sub_id],
                      'ID' : box.prop['ID'][sub_id],
                      'pos' : box.prop['pos'][sub_id],
                      'pos_b' : box.prop['pos_b'][sub_id],
                      'vel_b' : box.prop['vel_b'][sub_id],
                      'Mvir_b' : box.prop['Mvir_b'][sub_id],
                      'velmax_b' : box.prop['velmax_b'][sub_id],
                      'veldisp_b' : box.prop['veldisp_b'][sub_id],
                      'rvmax_b' : box.prop['rvmax_b'][sub_id],
                      'rhalfmass_b' : box.prop['rhalfmass_b'][sub_id]}
        elif hfname == 'Rockstar':
            boxpart = {'snapnum' : box.prop['snapnum'][sub_id],
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
                      'rvmax_b' : box.prop['rvmax_b'][sub_id]}
        return boxpart


    def find_subhalos_in_lc(self, lc, subpos, alpha):
        """
        Find subhalos in lightcone
        Use: class-internal
        alpha [deg]
        """
        radius = np.tan(alpha*np.pi/180)*np.max(subpos[:, 0])
        dist = np.sqrt(subpos[:, 1]**2 + subpos[:, 2]**2)
        if lc == None:
            sub_id = np.where((dist <= radius) & (0 <= subpos[:, 0]))
        else:
            sub_id = np.where((dist <= radius))
        return sub_id


    def fill_lightcone(self, lc, box, alpha, hfname):
        """
        Add new subhalos to Light-Cone
        """
        sub_id = self.find_subhalos_in_lc(lc, box['pos_b'], alpha)
        if len(sub_id[0]) != 0:
            return self.update_lc(sub_id, lc, box, hfname)
        else:
            #print('No halos in Lightcone')
            return None


    def find_sub_in_CoDi(self, subpos, codi_1, codi_2, direction):
        """
        If simulation box is crossing comoving-distance/redshift threshold
        Input:
            self:
            subpos:
            comodist:
            direction:
        Output:
            sub_id: element indices
        """
        sub_dist = np.sqrt(subpos[:, 0]**2 + \
                           subpos[:, 1]**2 + \
                           subpos[:, 2]**2)
        if direction == 0:
            sub_id = np.where((sub_dist <= codi_2) & (sub_dist >= codi_1))
        else:
            sub_id = np.where(sub_dist > codi_2)
        return sub_id

    
    def boxlength(self, pos):
        return np.max(pos[:, 0]) - np.min(pos[:, 0])


    def position_box_init(self, translation_z):
        """
        Position Observer in simulated box at z=0
        """
        boxlength = np.max(self.prop['pos_b'][:, 0]) - \
                    np.min(self.prop['pos_b'][:, 0])
        obserpos = np.ones(3)*boxlength/2
        self.prop['pos_b'][:, 0] -= obserpos[0]
        self.prop['pos_b'][:, 1] -= obserpos[1]
        self.prop['pos_b'][:, 2] -= obserpos[2]
        self.prop['pos_b'][:, 0] += translation_z
