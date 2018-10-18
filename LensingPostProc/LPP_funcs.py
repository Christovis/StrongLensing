from __future__ import division
import os, sys, logging
import numpy as np
import h5py
from astropy import units as u
from astropy import constants as const
import cfuncs as cf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import readlensing as rf
import readsnap
import read_hdf5


def check_in_sphere(c, pos, Rth):
    r = np.sqrt((c[0]-pos[:, 0])**2 + (c[1]-pos[:, 1])**2 + (c[2]-pos[:, 2])**2)
    indx = np.where(r < Rth)
    return indx


def plant_Tree():
    """ Create Tree to store data hierarchical """
    return collections.defaultdict(plant_Tree)


def lenslistinit(Nl):
    """ Create keys for tree.
        Input: Nl: number of lenses from LensingMap
    """
    global l_HFID, l_haloID, l_snapnum, l_deltat, l_mu, l_haloposbox, l_halovel, l_zs, l_zl, l_detA, l_srctheta, l_srcbeta, l_srcID, l_tancritcurves, l_einsteinradius
    
    l_HFID=np.zeros(Nl); l_haloID=np.zeros(Nl); l_snapnum=np.zeros(Nl); l_deltat=np.zeros(Nl); l_mu=np.zeros(Nl); l_haloposbox=np.zeros(Nl); l_halovel=np.zeros(Nl); l_zs=np.zeros(Nl); l_zl=np.zeros(Nl); l_detA=np.zeros(Nl); l_srctheta=np.zeros(Nl); l_srcbeta=np.zeros(Nl); l_srcID=np.zeros(Nl); l_tancritcurves=np.zeros(Nl); l_einsteinradius=np.zeros(Nl)
    return l_HFID, l_haloID, l_snapnum, l_deltat, l_mu, l_haloposbox, l_halovel, l_zs, l_zl, l_detA, l_srctheta, l_srcbeta, l_srcID, l_tancritcurves, l_einsteinradius


def srclistinit(Nl):
    """ Create keys for tree.
        Input: Nl: number of lenses from LensingMap
    """
    global s_srcID, s_deltat, s_mu, s_zs, s_alpha, s_detA, s_theta, s_beta, s_tancritcurves, s_einsteinradius
    s_srcID=np.zeros(Nl); s_deltat=np.zeros(Nl); s_mu=np.zeros(Nl); s_zs=np.zeros(Nl); s_alpha=np.zeros(Nl); s_detA=np.zeros(Nl); s_theta=np.zeros(Nl); s_beta=np.zeros(Nl); s_tancritcurves=np.zeros(Nl); s_einsteinradius=np.zeros(Nl)
    return s_srcID, s_deltat, s_mu, s_zs, s_alpha, s_detA, s_theta, s_beta, s_tancritcurves, s_einsteinradius


def mass_dynamical(Rad, PartVel, HaloPosBox, HaloVel, slices):
    """
    Estimate dynamical mass based on virial radius and
    stellar velocity dispersion
    Input:
        Rein: Einstein radii
        PartVel: Velocity of Particles
        HaloPosBox: Position of Lens
        HaloVel: Velocity of Lens
        slices: on which vrms is calculated

    Output:
        Mdyn: dynamical mass in solar mass
    """
    sigma = cf.call_vrms_gal(PartVel[:, 0], PartVel[:, 1], PartVel[:, 2],
                             HaloVel[0], HaloVel[1], HaloVel[2], slices) * \
            (u.kilometer/u.second)
    
    # Virial Theorem
    Mdyn = (sigma.to('m/s')**2*Rad.to('m')/const.G.to('m3/(kg*s2)')).to_value('M_sun')
    return Mdyn


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




