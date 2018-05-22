from __future__ import division
import glob
import re
import os.path
import logging
import random as rnd
import numpy as np
from pylab import *
import scipy.stats as stats
import scipy.interpolate as interpolate
from scipy import ndimage 
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy import constants as const
from astropy.constants import G
from astropy.cosmology import LambdaCDM
import matplotlib.pyplot as plt
import h5py
# Requires: Python (2.7.13), NumPy (>= 1.8.2), SciPy (>= 0.13.3)
import sklearn
from sklearn.neighbors import KDTree
import cfuncs as cf
sys.path.insert(0, '..')
import readsnap
import multiprocessing as mp
# surpress warnings from alpha_map_fourier
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())

###############################################################################


def ensure_dir(f):
    if not os.path.exists(f):
        os.makedirs(f)


def check_in_sphere(c, pos, Rth):
    r = np.sqrt((c[0]-pos[:, 0])**2 + (c[1]-pos[:, 1])**2 + (c[2]-pos[:, 2])**2)
    indx = np.where(r < Rth)
    return indx, r[indx]


def velocity_dispersion(radius, part_vel, halo_vel):
    num_of_los = 100
    los = []
    # Random Line-of-Sights ()
    for i in range(0, num_of_los):
        vec = [rnd.random(), rnd.random(), rnd.random()]
        los.append(vec/np.linalg.norm(vec))
    # substract subhalo speed from particle speed
    part_vel[:, 0] = part_vel[:, 0] - halo_vel[0]
    part_vel[:, 1] = part_vel[:, 1] - halo_vel[1]
    part_vel[:, 2] = part_vel[:, 2] - halo_vel[2]
    vlos = [part_vel*los[m] for m in range(len(los))]
    # Calculate Velocity Dispersion
    sigma = [np.std(vlos[m]) for m in range(len(vlos))]
    sigma_median = np.median(sigma)
    sigma_std = np.std(sigma)
    sigma_70perc = np.percentile(sigma, 70)
    return sigma_std


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
    logging.info('Lenses per CPU:', lensnum_per_cpu)
    return lenses_per_cpu


def sigma_crit(zLens, zSource, cosmo):
    Ds = cosmo.angular_diameter_distance(zSource)
    Dl = cosmo.angular_diameter_distance(zLens)
    Dls = cosmo.angular_diameter_distance_z1z2(zLens, zSource)
    D = (Ds/(Dl*Dls)).to(1/u.meter)
    sig_crit = (const.c**2/(4*np.pi*const.G))*D
    return sig_crit


def lensing_mass(Rein, zl, zs, cosmo):
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


def dynamical_mass(Rein, PartVel, HaloPosBox, HaloVel):
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
    sigma = velocity_dispersion(Rein.to_value('kpc'),
                                PartVel, HaloVel)*(u.kilometer/u.second)
    # Virial Theorem
    Mdyn = (sigma.to('m/s')**2*Rein.to('m')/G).to_value('M_sun')
    return Mdyn


def dyn_vs_lensing_mass(cpunum, lenses, LC, Halo_ID, snapnum, snapfile, h, scale,
                        HQ_dir, sim, sim_phy, sim_name, xi0, cosmo, results_per_cpu):
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
    logging.info('Process %s started' % mp.current_process().name)
   
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
        if 'Star_vel' not in locals():
            first_lens = ll
    
        # Load Lens properties
        indx = np.where(LC['Halo_ID'] == Halo_ID[ll])
        HaloVel = LC['HaloVel'][indx]

        # Only load new particle data if lens is at another snapshot
        if (previous_snapnum != snapnum[ll]) or (ll == first_lens):
            snap = snapfile % (snapnum[ll], snapnum[ll])
            # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH DM_pos = readsnap.read_block(snap, 'POS ', parttype=1)*scale
       #     DM_mass = readsnap.read_block(snap, 'MASS', parttype=1)*1e10/h
       #     DM_vel = readsnap.read_block(snap, 'VEL ', parttype=1)
       #     Gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*scale
       #     Gas_mass = readsnap.read_block(snap, 'MASS', parttype=0)*1e10/h
            Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale
            Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
       #     Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)
            Star_vel = readsnap.read_block(snap, 'VEL ', parttype=4)
            Star_pos = Star_pos[Star_age >= 0]
       #     Star_mass = Star_mass[Star_age >= 0]*1e10/h
            Star_vel = Star_vel[Star_age >= 0]
            del Star_age
        #    BH_pos = readsnap.read_block(snap, 'POS ', parttype=5)*scale
        #    BH_mass = readsnap.read_block(snap, 'MASS', parttype=5)*1e10/h
        previous_snapnum = snapnum[ll]
        
        # Run through lensing maps
        for ii in range(len(lm_files_match)):
            # Load LensingMap Contents
            LM = h5py.File(lm_files_match[ii])
            s = re.findall('[+-]?\d+', lm_files[ii])
            Src_ID=int(s[-2])

            Rein = LM['eqv_einstein_radius'].value*u.kpc
            zl = LM['zl'].value
            zs = LM['zs'].value
            HaloPosBox = LM['HaloPosBox'][:]
            HaloVel = LM['HaloVel'][:]

            #DM_indx, r = check_in_sphere(HaloPosBox, DM_pos, Rein.to_value('kpc'))
            #Gas_indx, r = check_in_sphere(HaloPosBox, Gas_pos, Rein.to_value('kpc'))
            Star_indx, r = check_in_sphere(HaloPosBox, Star_pos, Rein.to_value('kpc'))
            #BH_indx, r = check_in_sphere(HaloPosBox, BH_pos, Rein.to_value('kpc'))

            #Mlens = np.sum(DM_mass[DM_indx]) + np.sum(Gas_mass[Gas_indx]) + \
            #        np.sum(Star_mass[Star_indx]) + np.sum(BH_mass[BH_indx])
            Mlens = lensing_mass(Rein, zl, zs, cosmo)
            Mdyn = dynamical_mass(Rein, Star_vel[Star_indx], HaloPosBox, HaloVel)
            results.append([Halo_ID, Src_ID, Mdyn, Mlens])
    results_per_cpu[cpunum] = results
