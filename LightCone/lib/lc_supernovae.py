import sys
import constants as cc
import CosmoDist as cd
import numpy as np
import numba as nb
import random as rnd
from scipy.integrate import quad
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import readsubf
import readsnap
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/lib/')
import CosmoDist as cd


def redshift_division(zmax, cosmo, unit, cosmosim):
    exp = np.floor(np.log10(np.abs(unit))).astype(int)

    # Redshift bins for SNeIa
    zr = np.linspace(0, zmax, 1000)
    # middle redshift & distnace
    zmid = np.linspace((zr[1]-zr[0])/2, zr[-2] + (zr[-1]-zr[-2])/2, 999)
    if exp == 21:  # simulation in [kpc]
        dist_zr = (cosmo.comoving_distance(zr)).to_value('kpc')
        # Comoving distance between redshifts
        dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim)*1e3 for j in range(len(zr)-1)]
    elif exp == 23:  # simulation in [Mpc]
        dist_zr = (cosmo.comoving_distance(zr)).to_value('Mpc')
        # Comoving distance between redshifts
        dist_bet = [cd.comoving_distance(zr[j+1],zr[j],**cosmosim) for j in range(len(zr)-1)]
    else:
        raise Exception('Dont know this unit ->', exp)
    return zmid, dist_zr, dist_bet


#class SN_Type_Ia():
def update_SNeIa(indx, z, pos, **SNeIa):
    SNeIa['ID'] = np.concatenate((SNeIa['ID'], indx), axis=0)
    SNeIa['redshift'] = np.concatenate((SNeIa['redshift'], z), axis=0)
    SNeIa['position'] = np.concatenate((SNeIa['position'], pos), axis=0)
    return SNeIa

#@nb.njit(fastmath=True, parallel=False)
def Einstein_ring(veldisp, c, z, Dls=None, Ds=None):
    """
    Equation (2) from: https://arxiv.org/pdf/1708.00003.pdf
    Calculate Einstein angle, assume that the source is at inf. distance
    (Dls/Ds = 1)
    Assumes Point mass
    Input:
    - veldisp: velocity dispersion in [km/s]
    - c: speed of light in [km/s]
    - Dls: comoving distance between lense and source in [Mpc]
    - Ds: comoving distance between observer and source in [Mpc]
    Output:
    - theta_E: Einstein anlge in [rad]
    - R_E: Einstein radius in [kpc]
    Info:
        numba does not work with: astropy.units, -.constants
    """
    # Einstein angle
    if (Dls is None or Ds is None):
        # For Source at Inf.
        A_E = 4*np.pi*(veldisp/c)**2 #*u.rad  # [rad]
    else:
        A_E = 4*np.pi*(veldisp/c)**2*(Dls/Ds) #*u.rad  # [rad]
    return A_E #.to_value('rad')


#@nb.njit(fastmath=True, parallel=False)
def angle(v1, v2):
    """
    Returns the angle in radians
    """
    angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return angle


def perpendicular_vector(v, unit=0):
    pv = [-v[1], v[0], 0]
    if unit == 0:
        return np.asarray(pv)
    else:
        perp_unit_v = pv/np.linalg.norm(pv) 
        return np.asarray(perp_unit_v)


#@nb.njit(fastmath=True, parallel=False)
def round_axis(u, v, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians
    """ 
    u = np.asarray(u)
    u = u/np.sqrt(np.dot(u, u))
    a = np.cos(theta/2.)
    b, c, d = -u*np.sin(theta/2.)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    R = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    v_rot = np.dot(R, v)
    return v_rot


@nb.njit(fastmath=True, parallel=False)
def Einstein_Volume(fov, l_dist, z_dist):
    """
    Input:
        fov : estimate of Einstein Radii
              (assuming SIE and point source)
        l_dist :
        z_dist :
    Output:
        rfov : edge length of field-of-view [Mpc]
        V : Volume [Mpc]
    """
    # radius of f.o.v. (see Goldstein & Oguri)
    rfov = np.zeros(len(l_dist), dtype=np.float64)
    for j in range(len(l_dist)):
        rfov[j] = 4*fov*l_dist[j]  # 2, 4
    # Volume Element [in unit of simulation kpc, Mpc, ...]
    V = np.zeros(len(z_dist), dtype=np.float64)
    for j in range(len(z_dist)):
        V[j] = np.pi*rfov[j]**2*z_dist[j]  # for Subfind
    return V, rfov


def delaytime_power_law(tau):
    #Totani et al. (2008)
    return tau**(-1.08)


def rho_snia(z):
    # Hopkins & Beacom 2006
    _star_rhodot = (0.0118 + 0.08*z)*0.6779/(1 + (z/3.3)**5.2)
    return _star_rhodot


def snia_formation_rate(tau, t_upper_limit, redfunc):
    # Hopkins & Beacom 2006
    age = t_upper_limit  - tau  # [Gyr]
    if age > 0.18:
        zt = redfunc(age*cc.Gyr_s)
    else:
        #print('age: %f < 0.18 for tau: %f' % (age, tau))
        zt = 20.0
    star_rhodot = rho_snia(zt)*delaytime_power_law(tau)
    return star_rhodot


def integration(_sfr, a, b, N, redfunc):
    """
    Input:
        _sfr : the function to be integrated
        a, b : limits of integration
        N : integration steps 
    """
    x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
    fx = np.zeros(len(x))
    for ss in range(len(x)):
        fx[ss] = _sfr(x[ss], b, redfunc)
    area = np.sum(fx)*(b-a)/N
    return area


def integrand(a, b, redfunc):
    area = quad(snia_formation_rate, a, b, args=(b, redfunc))[0]
    return area


#@nb.njit(fastmath=True, parallel=False)
def snia_number_density(yr, z, agefunc, redfunc, **cosmosim):
    """
    Calculate Supernovae Type Ia distribution at given redshift

    Input:
    - zmid: redshift at the centre of volume
    - SFRmodel: 
    - cosmo: Dictionary of cosmological parameters
    Output:
    - SNeIa number density in [1/Mpc**3]
    """
    # Strigari et al. 2005
    SNeIa_delaytime = 2.7  # [Gyr] 1-3
    SNeIa_eff = 0.04  # 1%-5%
    norm = 4.87559
    
    sniafr = np.zeros(len(z))
    for ss in range(len(z)):
        t_upper_limit = agefunc(z[ss])/cc.Gyr_s  #[Gyr]
        #TODO

        sniafr[ss] = 3e-2*yr*SNeIa_eff/norm * \
                     integrand(0.1, t_upper_limit, redfunc)
    return sniafr


#@nb.njit  #(fastmath=True, parallel=False)
def SNIa_position(i, indx, SNIa_num, dist_sr, u_lenspos, l_lenspos, fov,
                  S_ID, S_possky, unit):
    exp = np.floor(np.log10(np.abs(unit))).astype(int)
    if exp == 21:  # simulation in [kpc]
        unit = u.kpc
    elif exp == 23:  #simulation in [Mpc]
        unit = u.Mpc

    sid = np.zeros(int(np.sum(SNIa_num)))
    sred = np.zeros(int(np.sum(SNIa_num)))
    sx = np.zeros(int(np.sum(SNIa_num)))
    sy = np.zeros(int(np.sum(SNIa_num)))
    sz = np.zeros(int(np.sum(SNIa_num)))
    indxtot = 0
    # Iterate over Volume
    for y in indx[0]:
        [dist_min, dist_max] = [dist_sr[y], dist_sr[y+1]]
        # Iterate over Supernovae
        for x in range(int(SNIa_num[y])):
            # Generate random SNeIa location within a cone
            # with apex-angle thetaE_inf along the l.o.s.
            radial_rnd = u_lenspos*(l_lenspos + \
                         rnd.random()*(dist_max - l_lenspos))
            sposx = np.sqrt(radial_rnd[0]**2 + radial_rnd[1]**2 + radial_rnd[2]**2)
            zs = z_at_value(cosmo.comoving_distance, sposx*unit, zmax=2)
            # max. distance equa to re
            charge = 1 if rnd.random() < 0.5 else -1
            sposy = charge*rnd.random()*fov[y]*0.5
            charge = 1 if rnd.random() < 0.5 else -1
            sposz = charge*rnd.random()*fov[y]*0.5
            # Write glafic file
            sid[indxtot] = i
            sred[indxtot] = zs
            sx[indxtot] = sposx
            sy[indxtot] = sposy
            sz[indxtot] = sposz
            indxtot = indxtot + 1;
    spossky = np.stack((sx, sy, sz)).transpose()
    return sid, sred, spossky


#@nb.njit  # (fastmath=True, parallel=False)
def SNIa_magnitudes(SNIa_num, zmid):
    """ https://arxiv.org/pdf/1001.2037.pdf
    Input:
        SNIa_num: num. of SNIa per redshift&Volume
        zmid: redshift at centre of volume
    Output:
        SNIa_M: magnitude of each SNIa
    """
    sigma_SNIa = 0.56  # magnitude dispersion around mean
    mu_SNIa = -19.06  # meant magnitude
    M_SNIa = np.zeros((1))
    for ii in range(len(SNIa_num)):
        mag = 1/(1 + zmid[ii]) * np.random.normal(mu_SNIa, sigma_SNIa, int(SNIa_num[ii]))
        M_SNIa = np.concatenate((M_SNIa, mag), axis=None)
    M_SNIa = np.delete(M_SNIa, 0)
    return M_SNIa


def select_halos(Halos, hfname):
    if hfname == 'Subfind':
        indx = np.where(Halos['M200'] > 1e11)[0]
        Halos = {'snapnum' : Halos['snapnum'][indx],
                'HF_ID' : Halos['HF_ID'][indx],
                'redshift' : Halos['Halo_z'][indx],
                'M200' : Halos['M200'][indx],
                'Rhalfmass' : Halos['Rhalfmass'][indx],
                'Rvmax' : Halos['Rvmax'][indx],
                'Vmax' : Halos['Vmax'][indx],
                'HaloPosBox' : Halos['HaloPosBox'][indx],
                'HaloPosLC' : Halos['HaloPosLC'][indx],
                'HaloVel' : Halos['HaloVel'][indx],
                'VelDisp' : Halos['Vrms'][indx]}
    elif hfname == 'Rockstar':
        indx = np.where(Halos['M200'] > 1e11)[0]
        Halos = {'snapnum' : Halos['snapnum'][indx],
                'HF_ID' : Halos['HF_ID'][indx],
                'redshift' : Halos['Halo_z'][indx],
                'M200' : Halos['M200'][indx],
                'Rvir' : Halos['Rvir'][indx],
                'Rsca' : Halos['Rsca'][indx],
                'Rvmax' : Halos['Rvmax'][indx],
                'Vmax' : Halos['Vmax'][indx],
                'HaloPosBox' : Halos['HaloPosBox'][indx],
                'HaloPosLC' : Halos['HaloPosLC'][indx],
                'HaloVel' : Halos['HaloVel'][indx],
                'VelDisp' : Halos['Vrms'][indx]}
    return Halos
