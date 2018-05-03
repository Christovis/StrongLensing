import Ellipticity as Ell
import constants as cc
import CosmoDist as cd
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15
import readsubf
import readsnap

#class SN_Type_Ia():
def update_SNeIa(indx, z, pos, **SNeIa):
    SNeIa['ID'] = np.concatenate((SNeIa['ID'], indx), axis=0)
    SNeIa['redshift'] = np.concatenate((SNeIa['redshift'], z), axis=0)
    SNeIa['position'] = np.concatenate((SNeIa['position'], pos), axis=0)
    return SNeIa


def Einstein_ring(veldisp, c, z, Dls, Ds, dim):
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
    """
    # Einstein angle
    if (Dls is None or Ds is None):
        # For Source at Inf.
        theta_E = 4*np.pi*(veldisp/c)**2  # [rad]
    else:
        theta_E = 4*np.pi*(veldisp/c)**2*(Dls/Ds)  # [rad]
    if dim == 'kpc':
        # convert radians to kpc
        R_E = theta_E*(180*3600/(np.pi*60)) * \
        Planck15.kpc_proper_per_arcmin(z)*(u.arcmin/u.kpc)
        return theta_E, R_E
    elif dim == 'rad':
        return theta_E


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


def SNeIa_distr_fct(z, SFRmodel, agefunc, redfunc, **cosmosim):
    """
    Calculate Supernovae Type Ia distribution at given redshift

    Input:
    - z: redshift
    - SFRmodel: 
    - cosmo: Dictionary of cosmological parameters
    Output:
    - SNeIa density in [1/yr/Mpc**3]
    """
    # Hopkins & Beacom 2006
    SNeIa_delaytime = 2.7  # [Gyr]
    # Strigari et al. 2005
    SNeIa_eff = 0.05  # 1%-5%

    if SFRmodel == 0:
        # Hopkins & Beacom 2006
        # convert redshift to conformal time
        age = agefunc(z)/cc.Gyr_s  # [Gyr]
        SNeIa_age = age - SNeIa_delaytime  # [Gyr]
        # convert conformal time to redshift
        z = redfunc(SNeIa_age*cc.Gyr_s)
    else:
        # Strolger et al. 2004
        # convert redshift to conformal time
        age = cd.age(z, **cosmosim)  # [sec]
        age /= cc.Gyr_s  # [Gyr]

    # Hopkins & Beacom 2006
    # star_rhodot(time - SNeIa_delaytime)
    star_rhodot = (0.017 + 0.13*z)*0.7/(1 + (z/3.3)**5.3)

    # Strolger et al. 2004
    # t0 = age of universe at z=0 
    # [a, b, c, d] = [0.182, 1.26, 1.865, 0.071]
    # star_rhodot = a*(t**b*np.exp(-t/c) + d*np.exp(d*(t-t0)/c))

    # Salpter A:
    f_SNeIa = 0.028*SNeIa_eff
    # BG IMFs: f_SNeIa = 0.032*SNeIa_eff
    SNeIa_rhodot = star_rhodot*f_SNeIa  # [1/yr/Mpc**3]
    return SNeIa_rhodot


def SNeIa_distr(yr, V, zmid, agefunc, redfunc, **cosmosim):
    SNIa_num = np.zeros(len(V))
    for j in range(len(V)):
        SNIa_num[j] = yr*V[j]*SNeIa_distr_fct(zmid[j], 0, agefunc, redfunc, **cosmosim)
    return SNIa_num


def Einstein_Volume(A_E, distmid_p, distbet_p):
    # see Goldstein & Oguri
    radiusE_inf = [8*A_E*dist for dist in distmid_p]  # [Mpc]
    # Volume Element  [Mpc**3]
    V = [2*np.pi*radiusE_inf[j]**2*distbet_p[j] for j in range(len(distbet_p))]
    return V, radiusE_inf 


def select_halos(Halos):
    indx = np.where(Halos['M200'] > 1e13)
    Halos = {'snapnum' : Halos['snapnum'][indx],
            'Halo_ID' : Halos['Halo_ID'][indx],
            'redshift' : Halos['redshift'][indx],
            'M200' : Halos['M200'][indx],
            'Rvir' : Halos['Rvir'][indx],
            'Rsca' : Halos['Rsca'][indx],
            'Rvmax' : Halos['Rvmax'][indx],
            'Vmax' : Halos['Vmax'][indx],
            'HaloPosBox' : Halos['HaloPosBox'][indx],
            'HaloPosLC' : Halos['HaloPosLC'][indx],
            'VelDisp' : Halos['VelDisp'][indx],
            'Ellip' : Halos['Ellip'][indx],
            'Pa' : Halos['Pa'][indx]}
    return Halos
