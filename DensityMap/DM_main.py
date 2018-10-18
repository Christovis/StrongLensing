# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging
from glob import glob
import subprocess
import numpy as np
from sklearn.neighbors import KDTree
import h5py
from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import DM_funcs as DMf
#import readlensing as rf

# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


def histedges_equalN(x, nbin):
    npt = len(x)
    bin_edges = np.interp(np.linspace(0, npt, nbin+1), np.arange(npt), np.sort(x))
    bin_edges[0] = 0
    bin_edges[-1] *= 1.1
    return bin_edges


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
    return id_out, vrms_out, x_out, y_out, z_out, split_size_1d, split_disp_1d


def cluster_particles(mass_in, x_in, y_in, z_in, _boundary, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _inds = np.digitize(x_in[:], _boundary)
    split_size_1d = np.zeros(comm_size)
    for b in range(comm_size):
        binds = np.where(_inds == b+1)
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


def adaptively_smoothed_maps(pos, h, mass, Lbox, centre, ncells, smooth_fac):
    """
    Gaussien smoothing kernel for 'neighbour_no' nearest neighbours
    Input:
        h: distance to furthest particle for each particle
    """
    hbins = int(np.log2(h.max()/h.min()))+2
    hbin_edges = 0.8*h.min()*2**np.arange(hbins)
    hbin_mids = np.sqrt(hbin_edges[1:]*hbin_edges[:-1])
    hmask = np.digitize(h, hbin_edges)-1  # returns bin for each h
    sigmaS = np.zeros((len(hbin_mids), ncells, ncells))
    for i in np.arange(len(hbin_mids)):
        maskpos = pos[hmask==i]  #X
        maskm = mass[hmask==i]
        maskSigma, xedges, yedges = np.histogram2d(maskpos[:, 0], maskpos[:, 1],
                                                   bins=[ncells, ncells],
                                                   weights=maskm)
        pixelsmooth = smooth_fac*hbin_mids[i]/(xedges[1]-xedges[0])
        sigmaS[i] = gaussian_filter(maskSigma, pixelsmooth, truncate=3)
    return np.sum(sigmaS, axis=0), xedges, yedges


def projected_surface_density_smooth_old(pos, mass, centre, fov, ncells,
                                     smooth_fac, neighbour_no, ptype):
    """
    Input:
        pos: particle positions
        mass: particle masses
        centre: centre of sub-&halo
        fov: field-of-view
        ncells: number of grid cells
    """
    pos = pos - centre
    
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*fov,
                           np.abs(pos[:, 1]) < 0.5*fov)
    pos = pos[_indx, :]
    mass = mass[_indx]
    
    if (ptype == 'dm' and len(mass) <= 32):
        return np.zeros((ncells, ncells))
    elif (ptype == 'gas' and len(mass) <= 16):
        return np.zeros((ncells, ncells))
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return np.zeros((ncells, ncells))
    else:
        # Find 'smoothing lengths'
        kdt = KDTree(pos, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(pos, k=neighbour_no, return_distance=True)
        h = np.max(dist, axis=1)  # furthest particle for each particle
        centre = np.array([0, 0, 0])
        mass_in_cells, xedges, yedges = adaptively_smoothed_maps(pos, h,
                                                                 mass,
                                                                 fov,
                                                                 centre,
                                                                 ncells,
                                                                 smooth_fac)
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
        return sigma


def projected_surface_density_smooth_old(pos, mass, centre, fov, ncells,
                                     smooth_fac, neighbour_no, ptype):
    """
    Input:
        pos: particle positions
        mass: particle masses
        centre: centre of sub-&halo
        fov: field-of-view
        ncells: number of grid cells
    """
    pos = pos - centre
    
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*fov,
                           np.abs(pos[:, 1]) < 0.5*fov)
    pos = pos[_indx, :]
    mass = mass[_indx]
    
    if (ptype == 'dm' and len(mass) <= 32):
        return np.zeros((ncells, ncells))
    elif (ptype == 'gas' and len(mass) <= 16):
        return np.zeros((ncells, ncells))
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return np.zeros((ncells, ncells))
    else:
        # Find 'smoothing lengths'
        kdt = KDTree(pos, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(pos, k=neighbour_no, return_distance=True)
        h = np.max(dist, axis=1)  # furthest particle for each particle
        centre = np.array([0, 0, 0])
        mass_in_cells, xedges, yedges = adaptively_smoothed_maps(pos, h,
                                                                 mass,
                                                                 fov,
                                                                 centre,
                                                                 ncells,
                                                                 smooth_fac)
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
        return sigma


def projected_surface_density(pos, mass, centre, Lbox, ncells):
    """
    """
    # centre particles around subhalo
    pos = pos - centre
    # 2D histogram map
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*Lbox,
                           np.abs(pos[:, 1]) < 0.5*Lbox)
    pos = pos[_indx, :]
    mass = mass[_indx]
    if len(mass) == 0:
        return np.zeros((ncells, ncells))
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return np.zeros((ncells, ncells))
    else:
        mass_in_cells, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                                       bins=[ncells, ncells],
                                                       weights=mass)
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        sigma = mass_in_cells/(dx*dy)  #[Msun/Mpc^2]
        #pixelsmooth = 0.3/(xedges[1]-xedges[0])
        #sigma = gaussian_filter(sigma, pixelsmooth, truncate=3)
        return sigma


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["snapnum"]      = int(sys.argv[3])
        args["ncells"]       = int(sys.argv[4])
        args["outbase"]      = sys.argv[5]
        args["nfileout"]     = sys.argv[6]
    args = comm.bcast(args)
    label = args["simdir"].split('/')[-2].split('_')[2]
   
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Sort Sub-&Halos over Processes
        df = pd.read_csv(args["hfdir"]+'halos_%d.dat' % args["snapnum"],
                         sep='\s+', skiprows=16,
                         usecols=[0, 2, 4, 9, 10, 11],
                         names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        df = df[df['Mvir'] > 5e11]
        sh_id = df['ID'].values.astype('float64')
        sh_vrms = df['Vrms'].values.astype('float64')
        sh_x = df['X'].values.astype('float64')
        sh_y = df['Y'].values.astype('float64')
        sh_z = df['Z'].values.astype('float64')
        del df
        hist_edges =  histedges_equalN(sh_x, comm_size)
        sh_id, sh_vrms, sh_x, sh_y, sh_z, sh_split_size_1d, sh_split_disp_1d = cluster_subhalos(sh_id, sh_vrms, sh_x, sh_y, sh_z, hist_edges, comm_size)
        
        # Sort Particles over Processes
        s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
        s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
               parttype=[0, 1, 4])
        scale = 1e-3*s.header.hubble
        ## Dark Matter
        dm_mass = (s.data['Masses']['dm']).astype('float64')
        dm_x = (s.data['Coordinates']['dm'][:, 0]*scale).astype('float64')
        dm_y = (s.data['Coordinates']['dm'][:, 1]*scale).astype('float64')
        dm_z = (s.data['Coordinates']['dm'][:, 2]*scale).astype('float64')
        dm_mass, dm_x, dm_y, dm_z, dm_split_size_1d, dm_split_disp_1d = cluster_particles(
                dm_mass, dm_x, dm_y, dm_z, hist_edges, comm_size)
        ## Gas
        gas_mass = (s.data['Masses']['gas']).astype('float64')
        gas_x = (s.data['Coordinates']['gas'][:, 0]*scale).astype('float64')
        gas_y = (s.data['Coordinates']['gas'][:, 1]*scale).astype('float64')
        gas_z = (s.data['Coordinates']['gas'][:, 2]*scale).astype('float64')
        gas_mass, gas_x, gas_y, gas_z, gas_split_size_1d, gas_split_disp_1d = cluster_particles(gas_mass, gas_x, gas_y, gas_z, hist_edges, comm_size)
        ## Stars
        star_mass = (s.data['Masses']['stars']).astype('float64')
        star_x = (s.data['Coordinates']['stars'][:, 0]*scale).astype('float64')
        star_y = (s.data['Coordinates']['stars'][:, 1]*scale).astype('float64')
        star_z = (s.data['Coordinates']['stars'][:, 2]*scale).astype('float64')
        star_age = s.data['GFM_StellarFormationTime']['stars']
        star_x = star_x[star_age >= 0]  #[Mpc]
        star_y = star_y[star_age >= 0]  #[Mpc]
        star_z = star_z[star_age >= 0]  #[Mpc]
        star_mass = star_mass[star_age >= 0]
        del star_age
        star_mass, star_x, star_y, star_z, star_split_size_1d, star_split_disp_1d = cluster_particles(star_mass, star_x, star_y, star_z, hist_edges, comm_size)

        # Define Cosmology
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
        cosmosim = {'omega_M_0' : s.header.omega_m,
                    'omega_lambda_0' : s.header.omega_l,
                    'omega_k_0' : 0.0,
                    'h' : s.header.hubble}
        redshift = s.header.redshift
    else:
        sh_id=None; sh_vrms=None; sh_x=None; sh_y=None; sh_z=None
        sh_split_size_1d=None; sh_split_disp_1d=None
        dm_mass=None; dm_x=None; dm_y=None; dm_z=None
        dm_split_size_1d=None; dm_split_disp_1d=None
        gas_mass=None; gas_x=None; gas_y=None; gas_z=None
        gas_split_size_1d=None; gas_split_disp_1d=None
        star_mass=None; star_x=None; star_y=None; star_z=None
        star_split_size_1d=None; star_split_disp_1d=None
        cosmosim=None; cosmo=None; redshift=None
      
    # Broadcast variables over all processors
    sh_split_size_1d = comm.bcast(sh_split_size_1d, root=0)
    sh_split_disp_1d = comm.bcast(sh_split_disp_1d, root=0)
    dm_split_size_1d = comm.bcast(dm_split_size_1d, root=0)
    dm_split_disp_1d = comm.bcast(dm_split_disp_1d, root=0)
    gas_split_size_1d = comm.bcast(gas_split_size_1d, root=0)
    gas_split_disp_1d = comm.bcast(gas_split_disp_1d, root=0)
    star_split_size_1d = comm.bcast(star_split_size_1d, root=0)
    star_split_disp_1d = comm.bcast(star_split_disp_1d, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    redshift = comm.bcast(redshift, root=0)

    # Initiliaze variables for each processor
    sh_id_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_vrms_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_x_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_y_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    sh_z_local = np.zeros((int(sh_split_size_1d[comm_rank])))
    dm_mass_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_x_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_y_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    dm_z_local = np.zeros((int(dm_split_size_1d[comm_rank])))
    gas_mass_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_x_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_y_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    gas_z_local = np.zeros((int(gas_split_size_1d[comm_rank])))
    star_mass_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_x_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_y_local = np.zeros((int(star_split_size_1d[comm_rank])))
    star_z_local = np.zeros((int(star_split_size_1d[comm_rank])))
    
    # Devide Data over Processes
    comm.Scatterv([sh_id, sh_split_size_1d, sh_split_disp_1d, MPI.DOUBLE],
                  sh_id_local, root=0)
    comm.Scatterv([sh_vrms, sh_split_size_1d, sh_split_disp_1d, MPI.DOUBLE],
                  sh_vrms_local, root=0)
    comm.Scatterv([sh_x, sh_split_size_1d, sh_split_disp_1d, MPI.DOUBLE],
                  sh_x_local,root=0) 
    comm.Scatterv([sh_y, sh_split_size_1d, sh_split_disp_1d, MPI.DOUBLE],
                  sh_y_local,root=0) 
    comm.Scatterv([sh_z, sh_split_size_1d, sh_split_disp_1d, MPI.DOUBLE],
                  sh_z_local,root=0) 
    
    comm.Scatterv([dm_x, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_x_local, root=0) 
    comm.Scatterv([dm_y, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_y_local, root=0) 
    comm.Scatterv([dm_z, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_z_local, root=0) 
    comm.Scatterv([dm_mass, dm_split_size_1d, dm_split_disp_1d, MPI.DOUBLE],
                  dm_mass_local,root=0) 

    comm.Scatterv([gas_x, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_x_local, root=0) 
    comm.Scatterv([gas_y, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_y_local, root=0) 
    comm.Scatterv([gas_z, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_z_local, root=0) 
    comm.Scatterv([gas_mass, gas_split_size_1d, gas_split_disp_1d, MPI.DOUBLE],
                  gas_mass_local,root=0) 

    comm.Scatterv([star_x, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_x_local, root=0)
    comm.Scatterv([star_y, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_y_local, root=0)
    comm.Scatterv([star_z, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_z_local, root=0)
    comm.Scatterv([star_mass, star_split_size_1d, star_split_disp_1d, MPI.DOUBLE],
                  star_mass_local,root=0)
    print(': Redshift: %f' % redshift)
    print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))

    comm.Barrier()

    SH = {"ID"   : sh_id_local,
          "Vrms" : sh_vrms_local,
          "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    DM = {"Mass"  : np.unique(dm_mass_local),
          "Pos" : np.transpose([dm_x_local, dm_y_local, dm_z_local])}
    Gas = {"Mass"  : gas_mass_local,
           "Pos" : np.transpose([gas_x_local, gas_y_local, gas_z_local])}
    Star = {"Mass"  : star_mass_local,
            "Pos" : np.transpose([star_x_local, star_y_local, star_z_local])}
    
    sigma_tot=[]; subhalo_id=[]; FOV=[]
    ## Run over Sub-&Halos
    for ll in range(len(SH['ID'])):
        # Define field-of-view
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(SH['Vrms'][ll]/c)**2
        #TODO: for z=0 sh_dist=0!!!
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        alpha = 6  # multiplied by 4 because of Oguri&Marshall
        fov_Mpc = alpha*fov_rad*sh_dist  # is it the diameter?
        
        dm_sigma, h = DMf.projected_surface_density_smooth(
                DM['Pos'],   #[Mpc]
                SH['Pos'][ll],
                fov_Mpc,  #[Mpc]
                args["ncells"])
        dm_sigma *= DM['Mass']
        gas_sigma = projected_surface_density(
                Gas['Pos'], #*a/h,
                Gas['Mass'],
                SH['Pos'][ll], #*a/h,
                fov_Mpc,
                args["ncells"])
        gas_sigma = gaussian_filter(gas_sigma, sigma=h)
        star_sigma = projected_surface_density(
                Star['Pos'], #*a/h,
                Star['Mass'],
                SH['Pos'][ll], #*a/h,
                fov_Mpc,
                args["ncells"])
        star_sigma = gaussian_filter(star_sigma, sigma=h)

        # Check if all density maps are empty
        if ((np.count_nonzero(dm_sigma) == args["ncells"]**2) and
                (np.count_nonzero(gas_sigma) == (args["ncells"])**2) and
                (np.count_nonzero(star_sigma) == (args["ncells"])**2)):
            continue
        sigmatotal = dm_sigma+gas_sigma+star_sigma
        
        sigma_tot.append(sigmatotal)
        subhalo_id.append(int(SH['ID'][ll]))
        FOV.append(fov_Mpc)
    
    fname = args["outbase"]+'z_'+str(args["snapnum"])+'/'+'DM_'+label+'_'+str(comm_rank)+'.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('density_map', data=sigma_tot)
    hf.create_dataset('subhalo_id', data=np.asarray(subhalo_id))
    hf.create_dataset('fov_width', data=np.asarray(FOV))
    #RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
    hf.close()


if __name__ == "__main__":
    create_density_maps()

