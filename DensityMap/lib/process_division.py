import os, sys
import numpy as np
import pandas as pd
import h5py
from sklearn.neighbors.kde import KernelDensity
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

def histedges_equalN(x, nbin):
    npt = len(x)
    bin_edges = np.interp(np.linspace(0, npt, nbin+1), np.arange(npt), np.sort(x))
    bin_edges[0] = 0
    bin_edges[-1] *= 1.1
    return bin_edges


def cube_of_cuboids(he, wi, de):
    """ Find optimal way to break cube into cuboids. """
    from fractions import gcd
    # gcd to find sides
    side = gcd(he, gcd(wi, de))


def cluster_subhalos(id_in, vrms_in, x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        if b == 0:
            id_out = id_in[binds]
            vrms_out = vrms_in[binds]
            x_out = x_in[binds]
            y_out = y_in[binds]
            z_out = z_in[binds]
            split_size_1d[b] = int(len(binds[0]))
        else:
            id_out = np.hstack((id_out, id_in[binds]))
            vrms_out = np.hstack((vrms_out, vrms_in[binds]))
            x_out = np.hstack((x_out, x_in[binds]))
            y_out = np.hstack((y_out, y_in[binds]))
            z_out = np.hstack((z_out, z_in[binds]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    SH = {'ID' : id_out,
          'Vrms' : vrms_out,
          'X' : x_out,
          'Y' : y_out,
          'Z' : z_out,
          'split_size_1d' : split_size_1d,
          'split_disp_1d' : split_disp_1d}
    return SH


def cluster_particles(mass_in, x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.

    Output:
        x_out, y_out, z_out : reordered coordinates
        split_size_1d, split_disp_1d : info. where to split array over cores
    """
    #_inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        #binds = np.where(_inds == b+1)
        binds = np.where((_boundary[b]*0.9 < x_in) & (x_in < _boundary[b+1]*1.1))
        if b == 0:
            mass_out = mass_in[binds]
            x_out = x_in[binds]
            y_out = y_in[binds]
            z_out = z_in[binds]
            split_size_1d[b] = int(len(binds[0]))
        else:
            mass_out = np.hstack((mass_out, mass_in[binds]))
            x_out = np.hstack((x_out, x_in[binds]))
            y_out = np.hstack((y_out, y_in[binds]))
            z_out = np.hstack((z_out, z_in[binds]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    return mass_out, x_out, y_out, z_out, split_size_1d, split_disp_1d

