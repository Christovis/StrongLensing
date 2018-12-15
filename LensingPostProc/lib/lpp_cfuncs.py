import numpy as np
import ctypes as ct
import os

lib_path = "/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/lib/"
#---------------------------------------------------------------------------------
f_vrms = ct.CDLL(lib_path+"lib_so_vrms/libvrms.so")
f_vrms.cal_vrms_gal.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               ct.c_float, ct.c_float, ct.c_float, \
                               np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               np.ctypeslib.ndpointer(dtype = ct.c_float, ndim=1), \
                               ct.c_int, ct.c_int, ct.c_float]
f_vrms.cal_vrms_gal.restype  = ct.c_float

def call_vrms_gal(particle_velocity, halo_velocity, slices):
    """
    Parameters:
    -----------
    particle_velocity : ndarray 2D
        particle velocities
    halo_velocity : ndarray 1D
        host halo velocity
    slices : ndarray 1D or 2D
        planes on which to evaluate velocity dispersion

    Returns:
    --------
    sigma : float
        velocity dispersion
    """
    Np = len(particle_velocity)
    Np = ct.c_int(Np)
    Ns = len(slices)
    Ns = ct.c_int(Ns)
    sv1 = np.array(particle_velocity[:, 0], dtype=ct.c_float)
    sv2 = np.array(particle_velocity[:, 1], dtype=ct.c_float)
    sv3 = np.array(particle_velocity[:, 2], dtype=ct.c_float)
    gv1 = ct.c_float(halo_velocity[0])  #np.array(gv1, dtype=ct.c_float)
    gv2 = ct.c_float(halo_velocity[1])  #np.array(gv2, dtype=ct.c_float)
    gv3 = ct.c_float(halo_velocity[2])  #np.array(gv3, dtype=ct.c_float)
    if len(slices.shape) == 1:  # if only one slice (1D)
        slices = slices.reshape((1, 3))
        slices = slices.T
        slices1 = slices[0]; slices2=slices[1]; slices3=slices[2]
    elif len(slices.shape) == 2:  # if only one slice (2D)
        slices = slices.T
        slices1 = slices[0]; slices2=slices[1]; slices3=slices[2]
    slices1 = np.array(slices1, dtype=ct.c_float)
    slices2 = np.array(slices2, dtype=ct.c_float)
    slices3 = np.array(slices3, dtype=ct.c_float)
    sigma = ct.c_float(0.0)

    sigma = f_vrms.cal_vrms_gal(sv1, sv2, sv3, gv1, gv2, gv3, slices1, slices2,
                                slices3, Np, Ns, sigma);
    return sigma

#-------------------------------------------------------------------------------
f_shmr = ct.CDLL(lib_path+"lib_so_shmr/libshmr.so")
f_shmr.cal_shmr_gal.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                 np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                 np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                 ct.c_float, ct.c_float, ct.c_float, \
                                 np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                 ct.c_int, ct.c_float]
f_shmr.cal_shmr_gal.restype  = ct.c_float

def call_stellar_halfmass(px1, px2, px3, cx1, cx2, cx3, pmass, rad):
    Np = len(px1)
    Np = ct.c_int(Np)
    px1 = np.array(px1, dtype=ct.c_float)
    px2 = np.array(px2, dtype=ct.c_float)
    px3 = np.array(px3, dtype=ct.c_float)
    cx1 = ct.c_float(cx1)
    cx2 = ct.c_float(cx2)
    cx3 = ct.c_float(cx3)
    pmass = np.array(pmass, dtype=ct.c_float)
    rad = ct.c_float(rad)

    R_shm = f_shmr.cal_shmr_gal(px1, px2, px3, cx1, cx2, cx3, pmass, Np, rad);
    return R_shm
