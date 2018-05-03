import CosmoDist as cd
from astropy import units as u
from astropy.cosmology import WMAP9
import Ellipticity as Ell
import readsubf
import readsnap
import numpy as np


class Lightcone():
    def __init__(self, hfdir, snap_dir, snapnum, header, halo_finder, LengthUnit):
        """
        Load Subhalo properties of snapshots into Dictionary
        Use:
            box = Lightcone(snapshot_file_name)

        Input:
            hfdir: snapshhot directory
            snap_dir: snapshot directory
            snapnum: snapshot number
            halo_finder: Subfind, Rockstar or AHF
        
        Output:
            subprop_box: dictionary with subhalo properties
            boxlength: side length of simulated box in units used in simulation
                       (e.g. Mpc/h or kpc/h)
        """
        if halo_finder == 'Subfind':
            subhalos = readsubf.subfind_catalog(hfdir, snapnum, masstab=True)
            subpos = subhalos.sub_pos
            submass = subhalos.sub_mass*1e10/header.hubble
            subveldisp = subhalos.sub_veldisp
            subrad = subhalos.sub_halfmassrad
            boxlength = np.max(subpos[:, 0]) - np.min(subpos[:, 0])
            sub_id = self.subhalo_selection(subpos, submass, subveldisp,
                                       subellipse, subpa)
            subpos = subpos[sub_id]
            submass = submass[sub_id]
            subveldisp = subveldisp[sub_id]
            # Calculate l.o.s. ellipticity and position angle	
            #parpos = readsnap.read_block(snap_dir % (snapnum, snapnum),
            #							  'POS ', parttype=0)
            #sub_e = Ell.ellipticity(subpos, submass, subrad, parpos, 1)
            #subpa = position_angle(principal_axes, los=1)
            subprop_box = {'pos_b' : subpos,
                           'mass_b' : submass,
                           'veldisp_b' : subveldisp}
                           #'ellipse_b' : sub_e}
                           #'pa_b' : subpa}
            self.subprop = subprop_box # boxlength
        elif halo_finder == 'Rockstar':
            hf_dir = hfdir + 'halos_%d.dat' % snapnum
            data = open(hf_dir, 'r')
            data = data.readlines()
            subID = []
            subpos = []
            subMvir = []
            subM200b = []
            subvmax = []
            subvrms = []
            subRvir = []
            subrs = []
            subrvmax = []
            sub_pa = []
            sub_e = []
            iki = 0
            for k in range(len(data))[16:]:
                if LengthUnit == 'kpc':
                    pos = [float(coord)*1e-3 for coord in data[k].split()[9:12]]
                else:
                    pos = [float(coord) for coord in data[k].split()[9:12]]
                av = [float(major_ax) for major_ax in data[k].split()[30:33]]
                #bv = [float(major_ax) for major_ax in data[k].split()[32:35]]
                #cv = [float(major_ax) for major_ax in data[k].split()[35:38]]
                if np.count_nonzero(av) != 0:
                    subID.append(float(data[k].split()[0]))
                    subMvir.append(float(data[k].split()[2]))
                    subvmax.append(float(data[k].split()[3]))
                    subvrms.append(float(data[k].split()[4]))
                    subRvir.append(float(data[k].split()[5]))
                    subrs.append(float(data[k].split()[6]))
                    subrvmax.append(float(data[k].split()[7]))
                    subpos.append(pos)
                    subM200b.append(float(data[k].split()[21]))
                    sub_s = float(data[k].split()[29])
                    sub_e.append(1. - sub_s)
                    PA = Ell.position_angle(av)
                    sub_pa.append(PA)
                    iki += 1
            subID = np.array(subID)
            subpos = np.array(subpos)
            subMvir = np.asarray(subMvir)
            subM200b = np.asarray(subM200b)
            subvmax = np.asarray(subvmax)
            subvrms = np.asarray(subvrms)
            subRvir = np.asarray(subRvir)
            subrs = np.asarray(subrs)
            subrvmax = np.asarray(subrvmax)
            sub_e = np.asarray(sub_e)
            sub_pa = np.asarray(sub_pa)
            boxlength = np.max(subpos[:, 0]) - np.min(subpos[:, 0])
            sub_id = self.subhalo_selection(subpos, subMvir, subvrms, sub_e,
                                            sub_pa)
            subprop_box = {'snapnum' : snapnum*np.ones(len(sub_id[0])),
                           'ID' : subID[sub_id][0],
                           'pos' : subpos[sub_id, :][0],
                           'pos_b' : subpos[sub_id, :][0],
                           'Mvir_b' : subMvir[sub_id][0],
                           'M200b_b' : subM200b[sub_id][0],
                           'velmax_b' : subvmax[sub_id][0],
                           'veldisp_b' : subvrms[sub_id][0],
                           'rvir_b' : subRvir[sub_id][0],
                           'rs_b' : subrs[sub_id][0],
                           'rvmax_b' : subrvmax[sub_id][0],
                           'ellipse_b' : sub_e[sub_id][0],
                           'pa_b' : sub_pa[sub_id][0]}
            self.subprop = subprop_box
        elif halo_finder == 'AHF':
            pass

    def subhalo_selection(self, pos, mass, veldisp, ellipse, pa):
        # Define subhalo selection
        sub_id1 = np.where(mass > 10**11)  # defined with Baojiu & Christian
        #sub_id2 = np.where(veldisp >= 160)  # arXiv:1507.07937
        #sub_id = set(sub_id1[0]) & set(sub_id2[0])
        #sub_id = np.array(list(sub_id))
        sub_id = np.array(sub_id1)
        return sub_id


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


    def update_lc(self, indx, lc, box):
        if lc == None:
            lc = {'snapnum_box' : box['snapnum'][indx],
                       'ID_box' : box['ID'][indx],
                       'pos_box' : box['pos'][indx],
                       'pos_lc' : box['pos_b'][indx], 
                       'Mvir_lc' : box['Mvir_b'][indx], 
                       'M200b_lc' : box['M200b_b'][indx], 
                       'velmax_lc' : box['velmax_b'][indx],
                       'veldisp_lc' : box['veldisp_b'][indx],
                       'rvir_lc' : box['rvir_b'][indx],
                       'rs_lc' : box['rs_b'][indx],
                       'rvmax_lc' : box['rvmax_b'][indx],
                       'ellipse_lc' : box['ellipse_b'][indx],
                       'pa_lc' : box['pa_b'][indx]}
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
            lc['ellipse_lc'] = np.concatenate((lc['ellipse_lc'],
                                               box['ellipse_b'][indx]),
                                                    axis=0)
            lc['pa_lc'] = np.concatenate((lc['pa_lc'],
                                          box['pa_b'][indx]),
                                          axis=0)
            print('lightcon test', len(lc), len(lc['Mvir_lc']), len(lc['ellipse_lc']))
            #del subprop['snapnum'], ['ID']
            #del subprop['pos'], subprop['pos_b'], subprop['Mvir_b'], subprop['M200b_b']
            #del subprop['velmax_b'], subprop['veldisp_b'], subprop['rvir_b']
            #del subprop['rs_b'], subprop['ellipse_b'], subprop['pa_b']
        return lc


    def fill_lightcone(self, lc, box, alpha):
        """
        Add new subhalos to Light-Cone
        """
        sub_id = self.find_subhalos_in_lc(lc, box['pos_b'], alpha)
        if len(sub_id[0]) != 0:
            return self.update_lc(sub_id, lc, box)
        else:
            return None


    def find_sub_in_CoDi(self, subpos, comodist, direction):
        """
        If simulation box is crossing comoving-distance/redshift threshold
        Input:
        Output:
            sub_id: element indices
        """
        sub_dist = np.sqrt(subpos[:, 0]**2 + \
                           subpos[:, 1]**2 + \
                           subpos[:, 2]**2)
        if direction == 0:
            sub_id = np.where(sub_dist <= comodist)
        else:
            sub_id = np.where(sub_dist > comodist)
        return sub_id

    
    def boxlength(self, pos):
        return np.max(pos[:, 0]) - np.min(pos[:, 0])


    def position_box_init(self, translation_z):
        """
        Position Observer in simulated box at z=0
        """
        boxlength = np.max(self.subprop['pos_b'][:, 0]) - \
                    np.min(self.subprop['pos_b'][:, 0])
        obserpos = np.ones(3)*boxlength/2
        self.subprop['pos_b'][:, 0] -= obserpos[0]
        self.subprop['pos_b'][:, 1] -= obserpos[1]
        self.subprop['pos_b'][:, 2] -= obserpos[2]
        self.subprop['pos_b'][:, 0] += translation_z
