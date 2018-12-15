from __future__ import division
import os, sys, logging
import math
import numpy as np
import h5py
from astropy import units as u
from astropy import constants as const
import lpp_cfuncs as lcf
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
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


def select_particles():
    _dist = np.sqrt(_pos[:, 0]**2 +
                    _pos[:, 1]**2 +
                    _pos[:, 2]**2)
    indx = np.where(_dist <= _radius)[0]
    return indx

def vrms_at_radius(particles, lvel, radius, radius_max):
    """
    Parameters:
    -----------
    ppos : ndarrau
        Particle Positions
    radius : float
        Radius at which to evaluate velocity dispersion
    Returns:
    --------
    """
    particles['Vel'] -= lvel
    _rmin = 0
    _rmax = radius_max

    _dist = np.sqrt(particles['Pos'][:, 0]**2 +
                    particles['Pos'][:, 1]**2 +
                    particles['Pos'][:, 2]**2)
    print('dist test', np.min(_dist), np.max(_dist))
    _indx = np.where((_dist >= _rmin) & (_dist <= _rmax))[0]
    rmedian = np.median(_dist[_indx])

    counter = 0
    while np.abs(rmedian-radius)/radius > 0.1 or counter < 20:
        
        if rmedian > radius:
            _rmax -= radius*0.1
            if _rmax < radius:
                _rmax += (radius-_rmax) + radius*0.01

        else:
            _rmin += radius*0.1
            if _rmin > radius:
                _rmin -= (_rmin - radius) - radius*0.01
        
        _indx = np.where((_dist >= _rmin) & (_dist <= _rmax))[0]
        rmedian = np.median(_dist[_indx])
        print('rmedian/radius', np.abs(rmedian-radius)/radius, len(_indx), counter)
        counter += 1
    
    vrms = velocity_dispersion(particles['Vel'])
    return vrms

def mass_dynamical(sigma, radius):
    """
    Estimate dynamical mass based on virial radius and
    stellar velocity dispersion
    Parameters:
    -----------
    Rein : float
        Einstein radii [kpc]
    vrms : float
        velocity dispersion [km/s]
    PartVel : ndarray
        Velocity of Particles
    HaloPosBox : ndarray
        Position of Lens
    HaloVel : ndarray
        Velocity of Lens
    slices  : ndarray
        on which vrms is calculated

    Returns:
    --------
    Mdyn : float
        dynamical mass in solar mass
    """
    #TODO: remove from this function and make it stand alone
    #sigma = cf.call_vrms_gal(PartVel[:, 0], PartVel[:, 1], PartVel[:, 2],
    #                         HaloVel[0], HaloVel[1], HaloVel[2], slices) * \
    #        (u.kilometer/u.second)
    
    # Virial Theorem
    #Mdyn = (sigma.to('m/s')**2*Rad.to('m')/const.G.to('m3/(kg*s2)')).to_value('M_sun')
    Mdyn = (sigma.to('km/s')**2*radius.to('m')/const.G.to('m3/(kg*s2)')).to_value('M_sun')
    return Mdyn


def sigma_crit(zLens, zSource, cosmo):
    """
    Parameters:
    -----------
    Returns:
    --------
    sig_crit : float
        Critical surface density [kg/m^2]
    """
    Ds = cosmo.angular_diameter_distance(zSource)
    Dl = cosmo.angular_diameter_distance(zLens)
    Dls = cosmo.angular_diameter_distance_z1z2(zLens, zSource)
    D = (Ds/(Dl*Dls)).to(1/u.meter)
    sig_crit = (const.c**2/(4*np.pi*const.G))*D
    return sig_crit.to_value('M_sun/kpc^2')


def mass_lensing(Rein, zl, zs, cosmo):
    """
    Estimate lensins mass
    Input:
        Rein: Einstein radii [kpc]
        zl: Redshift of Lens
        zs: Redshift of Source
        cosmo: Cosmological Parameters

    Output:
        Mlens: lensing mass in solar mass
    """
    sig_crit = sigma_crit(zl, zs, cosmo)
    Mlens = (np.pi*Rein**2*sig_crit) #.to_value('M_sun')
    return Mlens


def ellipticity_and_prolateness(_pos, dimensions):
    """
    Parameters:
    -----------
    pos : ndarray
        particle positions
    dimensions : int
        Dimensions
    Returns:
    --------
    ellipticity : float
    prolateness : float
    """
    if dimensions == 2:
        _centre = [_pos[:, 0].min() + (_pos[:, 0].max() - _pos[:, 0].min())/2,
                   _pos[:, 1].min() + (_pos[:, 1].max() - _pos[:, 1].min())/2,
                   _pos[:, 2].min() + (_pos[:, 2].max() - _pos[:, 2].min())/2]
        # Distance to parent halo
        _distance =  _pos - _centre 
        # Distance weighted Intertia Tensor / Reduced Inertia Tensor
        _I = np.dot(_distance.transpose(), _distance)
        _I /= np.sum(_distance**2)

        _eigenvalues, _eigenvectors = np.linalg.eig(_I)
        if ((_eigenvalues < 0).sum() > 0) or (np.sum(_eigenvalues) == 0):
            print('eigenvalue problem')
            ellipticity = 0
            prolateness = 0
        else:
            _eigenvalues = np.sqrt(_eigenvalues)
            _c, _b, _a = np.sort(_eigenvalues)
            _tau = _a + _b + _c
            ellipticity = (_a - _b) / (2*_tau)
            prolateness = (_a - 2*_b + _c) / (2*_tau)
            
    elif dimensions == 3:
        _centre = [_pos[:, 0].min() + (_pos[:, 0].max() - _pos[:, 0].min())/2,
                   _pos[:, 1].min() + (_pos[:, 1].max() - _pos[:, 1].min())/2,
                   _pos[:, 2].min() + (_pos[:, 2].max() - _pos[:, 2].min())/2]
        # Distance to parent halo
        _distance =  _pos - _centre 
        # Distance weighted Intertia Tensor / Reduced Inertia Tensor
        _I = np.dot(_distance.transpose(), _distance)
        _I /= np.sum(_distance**2)

        _eigenvalues, _eigenvectors = np.linalg.eig(_I)
        if ((_eigenvalues < 0).sum() > 0) or (np.sum(_eigenvalues) == 0):
            print('eigenvalue problem')
            ellipticity = 0
            prolateness = 0
        else:
            _eigenvalues = np.sqrt(_eigenvalues)
            _c, _b, _a = np.sort(_eigenvalues)
            _tau = _a + _b + _c
            ellipticity = (_a - _b) / (2*_tau)
            prolateness = (_a - 2*_b + _c) / (2*_tau)
            
    return ellipticity, prolateness


def projection(vec, proj):
    """
    Parameters:
    -----------
    Returns:
    --------
    """
    vec = vec*proj
    return vec


def velocity_dispersion(_velocity):
    """
    Calculate velocity dispersion
    Parameters:
    -----------
    Returns:
    --------
        _velocity[np.ndarray] : particle velocities
    """
    #TODO: along line-of-sight
    vel_los = np.sqrt(_velocity[:, 0]**2 + \
                      _velocity[:, 1]**2 + \
                      _velocity[:, 2]**2)
    vrms = np.std(vel_los)
    return vrms


def linear_radii_bin_factors(rmin, rmax, NCL):
    """ Create factors f,a to sort particles into the linear-bins """
    dx = (rmax-rmin)/NCL       # bin size
    a = -rmin*NCL/(rmax-rmin)  # fraction
    f = NCL/(rmax-rmin)        # bin size fraction
    return f, a, dx


def linear_profile(radpos, mass, quantity, param, partype,
                   rmin, rmax, NCL=100):
    # nshell -> number of particles in shell
    # dshell -> 
    dshell0 = 0.0
    nshell0 = 0.0
    nshell = np.zeros(NCL, np.float)
    dshell = np.zeros(NCL, np.float)
    f, a, dx = linear_radii_bin_factors(rmin, rmax, NCL)
    xiden = radpos*f + a
    iden = np.int32(xiden+0.0000001)
    id = np.where(radpos < rmax)
    i = 0
    for j in range(len(id[0])):
        i = id[0][j]
        if (param < 'density'):
            if (iden[i] >= 0) & (iden[i] < NCL):
                dshell[iden[i]] += m[i]
                nshell[iden[i]] += 1
            elif (iden[i] < 0):
                dshell0 += m[i]
                nshell0 += 1
        else:
            if (iden[i] >= 0) & (iden[i] < NCL):
                dshell[iden[i]] += m[i]*t[i]
                nshell[iden[i]] += m[i]
            elif (iden[i] < 0):
                dshell0 += m[i]*t[i]
                nshell0 += m[i]

    # output data
    rcoord = []         # coordinate
    diffprof = []       # differential profile

    xr = rmin-dx
    if (xr < 0.0):
        xrold = 0.0
    else:
        xrold = xr
    for i in range(0, NCL):
        xr = rmin+dx*i
        vshell = 4.0*math.pi*xr*xr*(xr-xrold)
        vol = 4.0*math.pi*xr*xr*xr/3.0
        if (dshell[i] > 0.0):
            if (param < 'density'):
                diffprof.append(dshell[i]/vshell)
            else:
                diffprof.append(dshell[i]/nshell[i])
            rcoord.append(xr+dx/2.0)
        xrold = xr
    return rcoord, diffprof


def log_radii_bin_factors(rmin, rmax, NCL):
    """ Create factors f,a to sort particles into the log-bins """
    dx = (rmax-rmin)/NCL       # bin size
    a = -rmin*NCL/(rmax-rmin)  # fraction
    f = NCL/(rmax-rmin)        # bin size fraction
    return f, a, dx


def logarithmic_profile(radpos, mass, quantity, param, partype,
                        rmin, rmax, sim, NCL=100):
    """
    Input:
        radpos[np.array]: radial particle position
        mass[np.array]: particle mass
        quantity[np.array]: e.g. temperature, entropy, etc. of particles
        param[str]: what quantity shall be analyzed
        partype[int]: 0:gas, 1:dm, 4:stars, 5:bh
        rmin, rmax[float]: boundaries of profile
        sim[dict]: simulation header containing constants and parameters
        NCL[int]: number of radial bins
    Output:
        rcoord[np.array]: radial coordinates
        diffprof[np.array]: average of measured quantity in radial bins
    """
    rmin = np.log10(rmin)
    rmax = np.log10(rmax)
    radpos = np.log10(radpos)
    
    # nshell -> number of particles in shell
    # dshell -> 
    dshell0=0.0; nshell0=0
    nshell = np.zeros(NCL, np.int32)
    dshell = np.zeros(NCL, np.float)
    f, a, dx = log_radii_bin_factors(rmin, rmax, NCL) 
    # xiden(radpos==min)=0; xiden(radpos==max)=NCL
    xiden = radpos*f + a
    iden = np.int32(xiden+1e-6)
    indx = np.where(radpos < rmax)

    # output data
    rcoord = np.zeros(NCL, np.float)    # coordinate
    diffprof = np.zeros(NCL, np.float)  # differential profile
   
    if param != 'vrms':
        # Run through particles
        for j in range(len(indx[0])):
            
            i = indx[0][j]  # particle index
            if (param in ['density', 'mass', 'circular_velocity']):
                if (iden[i] >= 0) & (iden[i] < NCL):
                    dshell[iden[i]] += mass[i]
                    nshell[iden[i]] += 1
                elif (iden[i] < 0):
                    # first shell
                    dshell0 += mass[i]
                    nshell0 += 1
            elif (param == 'velocity'):
                if (iden[i] >= 0) & (iden[i] < NCL):
                    dshell[iden[i]] += mass[i]*quantity[i]
                    nshell[iden[i]] += mass[i]
                elif (iden[i] < 0):
                    dshell0 += mass[i]*quantity[i]
                    nshell0 += mass[i]
            else:
                if (iden[i] >= 0) & (iden[i] < NCL):
                    dshell[iden[i]] += mass[i]*quantity[i]
                    nshell[iden[i]] += mass[i]
                elif (iden[i] < 0):
                    dshell0 += mass[i]*quantity[i]
        
        xr = rmin - dx
        xrold = 10.0**xr
        # Run through radii bins
        for i in range(0, NCL):
            xr = rmin + dx*i
            xr = 10.0**xr                          # linear-radius
            vshell = 4.0*math.pi*(xr-xrold)*xr**2  # differential Volume
            vol = 4.0/3.0*math.pi*xr**3            # Volume
            if (dshell[i] > 0.0):
                rcoord[i] = xr + (10.0**dx)/2.0
                if param == 'density':
                    diffprof[i] = dshell[i]/vshell
                elif param in ['mass', 'circular_velocity']:
                    diffprof[i] = dshell[i]   #[Msol/h]  Arepo units
                elif param == 'velocity':
                    diffprof[i] = dshell[i]/nshell[i]
            xrold = xr
        if param == 'mass':
            diffprof = np.cumsum(diffprof)
        if param == 'circular_velocity':
            #diffprof = diffprof*1e10*1.989*1e30/sim.header.hubble  # Msol/h to kg
            #rcoord *= 3.086e+16  #kiloparsec to meter
            diffprof = np.cumsum(diffprof)
            diffprof = np.asarray([np.sqrt(sim.const.G*1e9*diffprof[ii]/(rcoord[ii])) for ii in range(NCL)])
            diffprof *= 3.086e+16  # kpc to km
    elif param == 'vrms':
        vrms = np.zeros(NCL, np.float)
        radii = np.linspace(np.log10(rmin), np.log10(rmax), NCL)
        # Run through radii bins
        for rr in range(0, NCL):
            indx = np.where(radpos < radii[rr])
            rcoord[rr] = radii[rr]
            diffprof[rr] = velocity_dispersion(quantity[indx[0]])

    return rcoord, diffprof


def profiles(position, mass, quantity, param, partype, sim, rmin, rmax, NCL=100,
             ilog=1, itemp=1, iism=0):
    """
    Input:
        position: particle positions
        quantity: quantity for which to compute the profile
        param: What quantity it is (e.g. Mass, Temp, Density, ...)
        partype: particle type
        sim: simulation dictionary containing units, constants, boxsize...
        rmin,rmax: radii range
        NCL: number of bins
    Output:
        density: [Msun/kpc^3]
    """
    # needed to calculate the temperature
    GRAVITY = 6.67408*1e-11*3.24078e-20  #[m3/kg/s2], sim.const.G[Mpc3/Msolar/s2]
    #sim.const.G = GRAVITY
    BOLTZMANN = sim.const.kB  # erg/K
    PROTONMASS = sim.const.mproton*1e3  #[grams]
    Xh = sim.const.f
    HubbleParam = sim.header.hubble
    rhocr = sim.const.rho_crit/1.0e9  #[Msol/kpc^3] critical density

    # Radial Particle Positions
    radpos = np.sqrt((position[:, 0])**2 + \
                     (position[:, 1])**2 + \
                     (position[:, 2])**2)
    if param == 'velocity':
        quantity = np.asarray([np.linalg.norm(vec) for vec in quantity])
    
    if (ilog == False):
        rcoord, diffprof = linear_profile(
                radpos, mass, quantity, param, partype, rmin, rmax,
                sim, NCL=100)
    elif (ilog == True):
        rcoord, diffprof = logarithmic_profile(
                radpos, mass, quantity, param, partype, rmin, rmax,
                sim, NCL=100)

    return rcoord, diffprof


