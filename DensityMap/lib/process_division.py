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
from mpi4py import MPI

def histedges_equalN(x, nbin):
    npt = len(x)
    bin_edges = np.interp(np.linspace(0, npt, nbin+1), np.arange(npt), np.sort(x))
    bin_edges[0] = 0
    # some padding to make sure that all partciles are included
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


def cluster_subhalos_lc(hfid_in, id_in, red_in, snap_in, vrms_in, fov_in,
                        x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
        if b == 0:
            hfid_out = hfid_in[binds]
            id_out = id_in[binds]
            red_out = red_in[binds]
            snap_out = snap_in[binds]
            vrms_out = vrms_in[binds]
            fov_out = fov_in[binds]
            x_out = x_in[binds]
            y_out = y_in[binds]
            z_out = z_in[binds]
            split_size_1d[b] = int(len(binds[0]))
        else:
            hfid_out = np.hstack((hfid_out, id_in[binds]))
            id_out = np.hstack((id_out, id_in[binds]))
            red_out = np.hstack((red_out, id_in[binds]))
            snap_out = np.hstack((snap_out, id_in[binds]))
            vrms_out = np.hstack((vrms_out, vrms_in[binds]))
            fov_out = np.hstack((fov_out, vrms_in[binds]))
            x_out = np.hstack((x_out, x_in[binds]))
            y_out = np.hstack((y_out, y_in[binds]))
            z_out = np.hstack((z_out, z_in[binds]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    SH = {'HF_ID' : hfid_out,
          'ID' : id_out,
          'redshift' : red_out,
          'snapshot' : snap_out,
          'Vrms' : vrms_out,
          'fov_Mpc' : fov_out,
          'X' : x_out,
          'Y' : y_out,
          'Z' : z_out,
          'split_size_1d' : split_size_1d,
          'split_disp_1d' : split_disp_1d}
    return SH


def cluster_particles(Pa, _boundary, comm_size):
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
        binds = np.where((_boundary[b]*0.9 < Pa['Pos'][:, 0]) & \
                         (Pa['Pos'][:, 0] < _boundary[b+1]*1.1))
        if b == 0:
            mass_out = Pa['Mass'][binds]
            x_out = Pa['Pos'][binds, 0]
            y_out = Pa['Pos'][binds, 1]
            z_out = Pa['Pos'][binds, 2]
            split_size_1d[b] = int(len(binds[0]))
        else:
            mass_out = np.hstack((mass_out, Pa['Mass'][binds]))
            x_out = np.hstack((x_out, Pa['Pos'][binds, 0]))
            y_out = np.hstack((y_out, Pa['Pos'][binds, 1]))
            z_out = np.hstack((z_out, Pa['Pos'][binds, 2]))
            split_size_1d[b] = int(len(binds[0]))
    split_disp_1d = np.insert(np.cumsum(split_size_1d), 0, 0)[0:-1].astype(int)
    Particle = {'Mass' : mass_out,
                'X' : x_out,
                'Y' : y_out,
                'Z' : z_out,
                'split_size_1d' : split_size_1d,
                'split_disp_1d' : split_disp_1d}
    return Particle


def scatter_subhalos(SH, split_size_1d,
                     comrank, comm, root_proc=0):
    # Initiliaze variables for each processor
    sh_id_local = np.zeros((int(split_size_1d[comrank])))
    sh_vrms_local = np.zeros((int(split_size_1d[comrank])))
    sh_x_local = np.zeros((int(split_size_1d[comrank])))
    sh_y_local = np.zeros((int(split_size_1d[comrank])))
    sh_z_local = np.zeros((int(split_size_1d[comrank])))

    # Scatter
    comm.Scatterv([SH['ID'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_id_local, root=root_proc)
    comm.Scatterv([SH['Vrms'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_vrms_local, root=root_proc)
    comm.Scatterv([SH['X'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_x_local,root=root_proc)
    comm.Scatterv([SH['Y'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_y_local,root=root_proc)
    comm.Scatterv([SH['Z'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_z_local,root=root_proc)
    
    # Collect
    SH_out = {"ID"   : sh_id_local,
              "Vrms" : sh_vrms_local,
              "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    return SH_out


def scatter_subhalos_lc(SH, split_size_1d,
                        comrank, comm, root_proc=0):
    # Initiliaze variables for each processor
    sh_hfid_local = np.zeros((int(split_size_1d[comrank])))
    sh_id_local = np.zeros((int(split_size_1d[comrank])))
    sh_red_local = np.zeros((int(split_size_1d[comrank])))
    sh_snap_local = np.zeros((int(split_size_1d[comrank])))
    sh_vrms_local = np.zeros((int(split_size_1d[comrank])))
    sh_fov_local = np.zeros((int(split_size_1d[comrank])))
    sh_x_local = np.zeros((int(split_size_1d[comrank])))
    sh_y_local = np.zeros((int(split_size_1d[comrank])))
    sh_z_local = np.zeros((int(split_size_1d[comrank])))

    # Scatter
    comm.Scatterv([SH['HF_ID'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_hfid_local, root=root_proc)
    comm.Scatterv([SH['ID'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_id_local, root=root_proc)
    comm.Scatterv([SH['redshift'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_red_local, root=root_proc)
    comm.Scatterv([SH['snapshot'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_snap_local, root=root_proc)
    comm.Scatterv([SH['Vrms'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_vrms_local, root=root_proc)
    comm.Scatterv([SH['fov_Mpc'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_fov_local, root=root_proc)
    comm.Scatterv([SH['X'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_x_local,root=root_proc)
    comm.Scatterv([SH['Y'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_y_local,root=root_proc)
    comm.Scatterv([SH['Z'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_z_local,root=root_proc)
    
    # Collect
    SH_out = {"HF_ID"   : sh_hfid_local,
              "ID" : sh_id_local,
              "redshift" : sh_red_local,
              "snapshot" : sh_snap_local,
              "Vrms" : sh_vrms_local,
              "fov_Mpc" : sh_fov_local,
              "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    return SH_out


def scatter_particles(Pa, split_size_1d,
                      comrank, comm, root_proc=0):
    # Initiliaze variables for each processor
    mass_local = np.zeros((int(split_size_1d[comrank])))
    x_local = np.zeros((int(split_size_1d[comrank])))
    y_local = np.zeros((int(split_size_1d[comrank])))
    z_local = np.zeros((int(split_size_1d[comrank])))

    # Scatter
    comm.Scatterv([Pa['X'], Pa['split_size_1d'],
                   Pa['split_disp_1d'], MPI.DOUBLE],
                  x_local, root=root_proc) 
    comm.Scatterv([Pa['Y'], Pa['split_size_1d'],
                   Pa['split_disp_1d'], MPI.DOUBLE],
                  y_local, root=root_proc)
    comm.Scatterv([Pa['Z'], Pa['split_size_1d'],
                   Pa['split_disp_1d'], MPI.DOUBLE],
                  z_local, root=root_proc)
    comm.Scatterv([Pa['Mass'], Pa['split_size_1d'],
                   Pa['split_disp_1d'], MPI.DOUBLE],
                  mass_local,root=0)
    
    # Collect
    Pa_out = {"Mass"  : mass_local,
              "Pos" : np.transpose([x_local, y_local, z_local])}
    return Pa_out
