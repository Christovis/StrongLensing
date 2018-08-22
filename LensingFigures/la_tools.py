from __future__ import division
import re, sys, glob, os.path
import time
import logging
import pandas as pd
import numpy as np
#import numba as nb
import scipy.stats as stats
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pickle
import multiprocessing as mp
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/')
import readsnap
import read_hdf5
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import lm_tools  # Why do I need to load this???


###############################################################################


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def define_sim_label(simname, simdir):
    if 'GR' in simname:
        cosmo = 'GR'
    elif 'F5' in simname:
        cosmo = 'F5'
    elif 'F6'in simname:
        cosmo = 'F6' 
    if 'full_physics' in simdir:
        phy = 'FP'
    elif 'non_radiative_hydro' in simdir:
        phy = 'NRH'
    return phy+cosmo


#@njit('int[:](float64[:], float64[:, :], float64)', fastmath=True)
def check_in_sphere(c, pos, Rth):
    r = np.sqrt((c[0]-pos[:, 0])**2 + (c[1]-pos[:, 1])**2 + (c[2]-pos[:, 2])**2)
    indx = np.where(r < Rth)
    return indx


def devide_halos(halonum, cpunum):
    """
    Input:
        halonum: number of halos acting as lense
        cpunum: number of cpu's
    Output:
        lenses_per_cpu: lens ID's for each cpu
    """
    lensnum_per_cpu = np.ones(cpunum)*int(halonum/cpunum)
    lensnum_per_cpu = [int(x) for x in lensnum_per_cpu]
    missing_lenses = halonum - np.sum(lensnum_per_cpu)
    for x in range(missing_lenses):
        lensnum_per_cpu[x] += 1
    lensnum_per_cpu = np.cumsum(lensnum_per_cpu)
    lenses_per_cpu = []
    i = 0
    for x in range(cpunum):
        lenses_per_cpu.append(np.arange(i, lensnum_per_cpu[x]))
        i = lensnum_per_cpu[x]
    return lenses_per_cpu


def lens_source_params(units, cpunum, lenses, LC, Halo_ID, scale, HQ_dir, sim,
                       sim_phy, sim_name, cosmo, results_per_cpu):
    """
    Input:
        ll: halo array indexing
        LC: Light-cone dictionary
        Halo_ID: ID of Halo
        Halo_z: redshift of Halo
        Rvir: virial radius in [Mpc]
        previous_snapnum: 
        snapnum
    Output:
    """
    # LensMaps filenames
    lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+sim_name[sim]+'/'
    lm_files = [name for name in glob.glob(lm_dir+'LM_L*')]

    first_lens = lenses[0]
    previous_snapnum = snapnum[lenses[0]]
    results = []
    # Run through lenses
    for ll in range(lenses[0], lenses[-1]):
        lm_files_match = [e for e in lm_files if 'L%d'%(Halo_ID[ll]) in e]
        if not lm_files_match:
            continue

        # Load Lens properties
        indx = np.where(LC['Halo_ID'] == Halo_ID[ll])
        measure = np.zeros(len(units))
        for jj in units:
            measure[jj] = LC[units[jj]][indx]
        
        measures.append(measure)

    #for mm in range(len(measures)):
    #    Halo_ID.append()
    #    Src_ID.append()
    if shape(measures):
        unit_dict = {units : measures}
    else:
        unit_dict = {units : measures}
    return unit_dict


def sigma_crit(zLens, zSource, cosmo):
    Ds = cosmo.angular_diameter_distance(zSource)
    Dl = cosmo.angular_diameter_distance(zLens)
    Dls = cosmo.angular_diameter_distance_z1z2(zLens, zSource)
    D = (Ds/(Dl*Dls)).to(1/u.meter)
    sig_crit = (const.c**2/(4*np.pi*const.G))*D
    return sig_crit


def mass_lensing(Rein, zl, zs, cosmo):
    """
    Estimate lensins mass
    Input:
        Rein: Einstein radii
        zl: Redshift of Lens
        zs: Redshift of Source
        cosmo: Cosmological Parameters

    Output:
        Mlens: lensing mass in solar mass
    """
    sig_crit = sigma_crit(zl, zs, cosmo)
    Mlens = (np.pi*Rein.to(u.meter)**2*sig_crit).to_value('M_sun')
    return Mlens


def mass_dynamical(Rad, PartVel, HaloPosBox, HaloVel, avec, bvec, cvec):
    """
    Estimate dynamical mass based on virial radius and
    stellar velocity dispersion
    Input:
        Rein: Einstein radii
        PartVel: Velocity of Particles
        HaloPosBox: Position of Lens
        HaloVel: Velocity of Lens

    Output:
        Mdyn: dynamical mass in solar mass
    """
    slices = np.vstack((avec/np.linalg.norm(avec),
                        bvec/np.linalg.norm(bvec),
                        cvec/np.linalg.norm(cvec)))
    sigma = cf.call_vrms_gal(PartVel[:, 0], PartVel[:, 1], PartVel[:, 2],
                             HaloVel[0], HaloVel[1], HaloVel[2], slices) * \
            (u.kilometer/u.second)
    # Virial Theorem
    Mdyn = (sigma.to('m/s')**2*Rad.to('m')/const.G.to('m3/(kg*s2)')).to_value('M_sun')
    return Mdyn


def dyn_vs_lensing_mass(cpunum, LC, lm_file, snapfile, h, scale, HQ_dir, sim,
                        sim_phy, sim_name, hfname, cosmo, results_per_cpu):
    """
    Input:
        ll: halo array indexing
        LC: Light-cone dictionary
        Halo_ID: ID of Halo
        Halo_z: redshift of Halo
        Rvir: virial radius in [Mpc]
        previous_snapnum: 
        snapnum
    Output:
    """
    LM = pickle.load(open(lm_file, 'rb'))  #, encoding="utf8")
    logging.info('Process %s started. Nr. of Halos: %s' % (mp.current_process().name, len(LM['Halo_ID'])))
    results = []
    previous_snapnum = -1
    # Run through lenses
    for ll in range(len(LM['Halo_ID'])):
        # Load Lens properties
        #HaloHFID= LM['HF_ID'][ll]
        HaloHFID= int(LM['Rockstar_ID'][ll])
        HaloPosBox = LM['HaloPosBox'][ll]
        HaloVel = LM['HaloVel'][ll]
        snapnum = LM['snapnum'][ll]
        zl = LM['zl'][ll]


        # Only load new particle data if lens is at another snapshot
        if (previous_snapnum != snapnum):
            # Load Halo Properties
            rks_file = '/cosma5/data/dp004/dc-beck3/rockstar/'+sim_phy[sim]+ \
                       sim_name[sim]+'/halos_' + str(snapnum)+'.dat'
            df = pd.read_csv(rks_file, 
                             sep='\s+', 
                             skiprows=16, 
                             usecols=[0, 4, 5, 30, 31, 32, 33, 34, 35, 36, 37, 38, 47], 
                             names=['ID', 'Vrms', 'Rvir', 'A[x]', 'A[y]', 'A[z]',
                                    'B[x]', 'B[y]', 'B[z]', 'C[x]', 'C[y]', 'C[z]',
                                    'Halfmass_Radius'])
            # Load Particle Properties
            #s = read_hdf5.snapshot(snapnum, snapfile)
            # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
            # Have to load all particles :-( -> takes too long
            #s.read(["Velocities", "Coordinates", "AGE"], parttype=-1)
            #Star_pos = s.data['Velocities']['stars']*scale
            snap = snapfile % (snapnum, snapnum)
            Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
            Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale #[Mpc]
            Star_vel = readsnap.read_block(snap, 'VEL ', parttype=4)
            Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)*1e10/h
            Star_pos = Star_pos[Star_age >= 0]
            Star_vel = Star_vel[Star_age >= 0]
            Star_mass = Star_mass[Star_age >= 0]
            del Star_age
        previous_snapnum = snapnum

        # Load Halo Properties
        indx = df['ID'][df['ID'] == HaloHFID].index[0]
        Vrms = df['Vrms'][indx]*(u.km/u.s)  #[km/s]
        Rvir = df['Rvir'][indx]*u.kpc
        Rhalfmass = df['Halfmass_Radius'][indx]*u.kpc
        epva = pd.concat([df['A[x]'], df['A[y]'], df['A[z]']], axis=1).loc[[indx]].values
        epvb = pd.concat([df['B[x]'], df['B[y]'], df['B[z]']], axis=1).loc[[indx]].values
        epvc = pd.concat([df['C[x]'], df['C[y]'], df['C[z]']], axis=1).loc[[indx]].values

        # Stellar half mass radius
        Rshm = cf.call_stellar_halfmass(Star_pos[:, 0], Star_pos[:, 1],
                                        Star_pos[:, 2], HaloPosBox[0],
                                        HaloPosBox[1], HaloPosBox[2],
                                        Star_mass, Rvir.to_value('Mpc'))*u.Mpc
        #print('Rshm', Rshm)
        Star_indx = check_in_sphere(HaloPosBox, Star_pos, Rshm.to_value('kpc'))
        if len(Star_indx[0]) > 50:
            Mdyn = mass_dynamical(Rshm, Star_vel[Star_indx], HaloPosBox,
                                  HaloVel, epva, epvb, epvc)
            print('Mdyn', Mdyn)
            if Mdyn == .0:
                continue
            # Run through sources
            for ss in range(len(LM['Sources']['Src_ID'][ll])):
                zs = LM['Sources']['zs'][ll][ss]
                Rein = LM['Sources']['Rein'][ll][ss]*u.kpc
                Mlens = mass_lensing(Rein, zl, zs, cosmo)
                #Mdyn = (Vrms.to('m/s')**2*Rvir.to('m')/ \
                #        const.G.to('m3/(kg*s2)')).to_value('M_sun')

                results.append([LM['Halo_ID'][ll], LM['Sources']['Src_ID'][ll][ss],
                                Mdyn, Mlens])
    results_per_cpu[cpunum] = results
