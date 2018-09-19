# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging
from glob import glob
import subprocess
import numpy as np
import h5py
#from scipy.ndimage.filters import gaussian_filter
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pandas as pd
#import dm_funcs as DM
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
#import readlensing as rf

# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#import parallel_sort as ps
from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


def cluster_subhalos(id_in, vrms_in, x_in, y_in, z_in, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _boundary = [np.min(x_in[:]), np.max(x_in[:])*1.01]
    _boundary = np.linspace(_boundary[0], _boundary[1], comm_size+1)
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


def cluster_particles(mass_in, x_in, y_in, z_in, comm_size):
    """
    Devide simulation box into a number of equal sized parts 
    as there are processes along one axis.
    """
    _boundary = [np.min(x_in[:]), np.max(x_in[:])*1.01]
    _boundary = np.linspace(_boundary[0], _boundary[1], comm_size+1)
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


def projected_surface_density(pos, mass, centre, fov, bins=512, smooth=True,
                              smooth_fac=None, neighbour_no=None):
    """
    Fit ellipsoid to 3D distribution of points and return eigenvectors
    and eigenvalues of the result.
    The same as 'projected_surface_density_adaptive', but assumes that
    density map is not created and particle data loaded, and you can
    choose between smoothed and closest particle density map.
    Should be able to combine all three project_surface_density func's

    Args:
        pos: particle position (physical)
        mass: particle mass
        centre: lense centre [x, y, z]
        fov:
        bins:
        smooth:
        smooth_fac:
        neighbour_no:

    Returns:
        Sigma: surface density [Msun/Mpc^2]
        x: 
        y:
    """
    Lbox = fov  #[Mpc]
    Ncells = bins

    ################ Shift particle coordinates to centre ################
    pos = pos - centre
    ####################### DM 2D histogram map #######################
    _indx = np.logical_and(np.abs(pos[:, 0]) < 0.5*Lbox,
                           np.abs(pos[:, 1]) < 0.5*Lbox)
    pos = pos[_indx, :]
    mass = mass[_indx]
    if len(mass) == 0:
        return None, None, None
    elif ((np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])) > 0.1) or 
            (np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])) > 0.1)):
        #TODO: plot this falty situation
        return None, None, None
    else:
        mass_in_cells, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1],
                                                       bins=[Ncells, Ncells],
                                                       #range=[[-0.5*Lbox, 0.5*Lbox],
                                                       #       [-0.5*Lbox, 0.5*Lbox]],
                                                       weights=mass)
        ###################### Projected surface density ######################
        dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]  #[Mpc]
        Sigma = mass_in_cells/(dx*dy)      #[Msun/Mpc^2]
        xs = 0.5*(xedges[1:]+xedges[:-1])  #[Mpc]
        ys = 0.5*(yedges[1:]+yedges[:-1])  #[Mpc]
        return Sigma, xs, ys


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["snapnum"]      = sys.argv[3]
        args["ncells"]       = sys.argv[4]
        args["outbase"]      = sys.argv[5]
        args["nfileout"]     = sys.argv[6]
    args = comm.bcast(args)
   
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Sort Sub-&Halos over Processes
        df = pd.read_csv(args["hfdir"]+'halos_%s.dat' % args["snapnum"],
                         sep='\s+', skiprows=16,
                         usecols=[0, 2, 4, 9, 10, 11],
                         names=['ID', 'Mvir', 'Vrms', 'X', 'Y', 'Z'])
        df = df[df['Mvir'] > 1e11]
        sh_id = df['ID'].values.astype('float64')
        sh_vrms = df['Vrms'].values.astype('float64')
        sh_x = df['X'].values.astype('float64')
        sh_y = df['Y'].values.astype('float64')
        sh_z = df['Z'].values.astype('float64')
        del df
        sh_id, sh_vrms, sh_x, sh_y, sh_z, sh_split_size_1d, sh_split_disp_1d = cluster_subhalos(
                sh_id, sh_vrms, sh_x, sh_y, sh_z, comm_size)
        
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
                dm_mass, dm_x, dm_y, dm_z, comm_size)
        ## Gas
        gas_mass = (s.data['Masses']['gas']).astype('float64')
        gas_x = (s.data['Coordinates']['gas'][:, 0]*scale).astype('float64')
        gas_y = (s.data['Coordinates']['gas'][:, 1]*scale).astype('float64')
        gas_z = (s.data['Coordinates']['gas'][:, 2]*scale).astype('float64')
        gas_mass, gas_x, gas_y, gas_z, gas_split_size_1d, gas_split_disp_1d = cluster_particles(
                gas_mass, gas_x, gas_y, gas_z, comm_size)
        ## Stars
        star_mass = (s.data['Masses']['stars']).astype('float64')
        star_x = (s.data['Coordinates']['stars'][:, 0]*scale).astype('float64')
        star_y = (s.data['Coordinates']['stars'][:, 1]*scale).astype('float64')
        star_z = (s.data['Coordinates']['stars'][:, 2]*scale).astype('float64')
        star_age = s.data['GFM_StellarFormationTime']['stars']
        star_x = star_x[star_age >= 0]*scale  #[Mpc]
        star_y = star_y[star_age >= 0]*scale  #[Mpc]
        star_z = star_z[star_age >= 0]*scale  #[Mpc]
        star_mass = star_mass[star_age >= 0]
        del star_age
        star_mass, star_x, star_y, star_z, star_split_size_1d, star_split_disp_1d = cluster_particles(
                star_mass, star_x, star_y, star_z, comm_size)

        # Define Cosmology
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
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
        cosmo=None; redshift=None
      
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

    print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))
    
    SH = {"ID"   : sh_id_local,
          "Vrms" : sh_vrms_local,
          "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    DM = {"Mass"  : dm_mass_local,
          "Pos" : np.transpose([dm_x_local, dm_y_local, dm_z_local])}
    Gas = {"Mass"  : gas_mass_local,
           "Pos" : np.transpose([gas_x_local, gas_y_local, gas_z_local])}
    Star = {"Mass"  : star_mass_local,
            "Pos" : np.transpose([star_x_local, star_y_local, star_z_local])}
    
    sigma_tot=[]; subhalo_id=[]
    ## Run over Sub-&Halos
    for ll in range(len(SH['ID'])):
        print('Proc. %d analysing subhalo nr. %d' % (comm_rank, ll))
        # Define field-of-view
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(SH['Vrms'][ll]/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        fov_Mpc = 4*fov_rad*sh_dist  # multiplied by 4 because of Oguri&Marshall

        dm_sigma, xs, ys = projected_surface_density(DM['Pos'],   #[Mpc]
                                                     DM['Mass'],
                                                     SH['Pos'][ll],
                                                     fov=fov_Mpc,  #[Mpc]
                                                     bins=int(args["ncells"]),
                                                     smooth=False,
                                                     smooth_fac=0.5,
                                                     neighbour_no=32)
        if dm_sigma is None: continue
        gas_sigma, xs, ys = projected_surface_density(Gas['Pos'], #*a/h,
                                                      Gas['Mass'],
                                                      SH['Pos'][ll], #*a/h,
                                                      fov=fov_Mpc,
                                                      bins=int(args["ncells"]),
                                                      smooth=False,
                                                      smooth_fac=0.5,
                                                      neighbour_no=32)
        if gas_sigma is None: continue
        star_sigma, xs, ys = projected_surface_density(Star['Pos'], #*a/h,
                                                       Star['Mass'],
                                                       SH['Pos'][ll], #*a/h,
                                                       fov=fov_Mpc,
                                                       bins=int(args["ncells"]),
                                                       smooth=False,
                                                       smooth_fac=0.5,
                                                       neighbour_no=8)
        if star_sigma is None: continue

        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        #sigma_tot.append(dm_sigma+gas_sigma+star_sigma)
        sigma_tot.append(gaussian_filter(dm_sigma+gas_sigma+star_sigma, sigma=3))
        subhalo_id.append(SH['ID'][ll])

        #if (ll % 1000 == 0) and (ll != 0):
        #    hf = h5py.File('./'+sim_phy[sim]+sim_name[sim]+'/DM_'+sim_name[sim]+'_' +str(ll)+'.h5', 'w')
        #    hf.create_dataset('density_map', data=np.asarray(sigma_tot))
        #    hf.create_dataset('subhalo_id', data=np.asarray(subhalo_id))
        #    hf.close()
        #    sigma_tot=[]; subhalo_id=[]
    label = sim_name[sim].split('_')[2]
    fname = './'+sim_phy[sim]+sim_name[sim]+'/DM_'+label+'_'+str(comm_rank)+'.h5'
    hf = h5py.File(fname, 'w')
    hf.create_dataset('density_map', data=np.asarray(sigma_tot))
    hf.create_dataset('subhalo_id', data=np.asarray(subhalo_id))
    hf.close()


if __name__ == "__main__":
    create_density_maps()

