# Python implementation using the multiprocessing module
#
from __future__ import division
import os, sys, logging, time
from glob import glob
import numpy as np
import h5py
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
import pandas as pd
#import testdensitymap as tmap
#import density_maps as dmap
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/lib/')
import read_hdf5
import pmesh
sys.path.insert(0, '../lib/')
import density_maps as dmap
import process_division as procdiv
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/')
import cfuncs as cf


# MPI initialisation
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

from mpi_errchk import mpi_errchk

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)
os.system("taskset -p 0xff %d" % os.getpid())


def sigma_crit(zLens, zSource, cosmo):
    Ds = cosmo.angular_diameter_distance(zSource)
    Dl = cosmo.angular_diameter_distance(zLens)
    Dls = cosmo.angular_diameter_distance_z1z2(zLens, zSource)
    sig_crit = (const.c**2/(4*np.pi*const.G))*Ds/(Dl*Dls)
    return sig_crit


def cal_lensing_signals(kap, bzz, ncc):
    dsx_arc = bzz/ncc
    # deflection maps
    alpha1, alpha2 = cf.call_cal_alphas(kap, bzz, ncc)

    # shear maps
    npad = 5
    al11 = 1 - np.gradient(alpha1, dsx_arc, axis=0)
    al12 = - np.gradient(alpha1, dsx_arc, axis=1)
    al21 = - np.gradient(alpha2, dsx_arc, axis=0)
    al22 = 1 - np.gradient(alpha2, dsx_arc, axis=1)
    detA = al11*al22 - al12*al21

    kappa0 = 1 - 0.5*(al11 + al22)
    shear1 = 0.5*(al11 - al22)
    shear2 = 0.5*(al21 + al12)
    shear0 = (shear1**2 + shear2**2)**0.5
    
    # magnification maps
    mu = 1.0/((1.0-kap)**2.0-shear1*shear1-shear2*shear2)
    lambda_t = 1 - kappa0 - shear0  # tangential eigenvalue, page 115
    
    # lensing potential
    phi = cf.call_cal_phi(kap, bzz, ncc)

    return alpha1, alpha2, mu, phi, detA, lambda_t


def einstein_radii(xs, ys, detA, lambda_t, zl, cosmo, ax, method, ll):
    curve_crit = ax.contour(xs, ys, detA,
                            levels=(0,), colors='r',
                            linewidths=1.5, zorder=200)
    Ncrit = len(curve_crit.allsegs[0])
    curve_crit = curve_crit.allsegs[0]
    curve_crit_tan = ax.contour(xs, ys,
                                lambda_t, levels=(0,), colors='r',
                                linewidths=1.5, zorder=200)
    Ncrit_tan = len(curve_crit_tan.allsegs[0])
    if Ncrit_tan > 0:
        len_tan_crit = np.zeros(Ncrit_tan)
        for i in range(Ncrit_tan):
            len_tan_crit[i] = len(curve_crit_tan.allsegs[0][i])
        curve_crit_tan = curve_crit_tan.allsegs[0][len_tan_crit.argmax()]
        if method == 'eqv':
            Rein = np.sqrt(np.abs(area(curve_crit_tan))/np.pi)  #[arcsec]
        if method == 'med':
            dist = np.sqrt(curve_crit_tan[:, 0]**2 + curve_crit_tan[:, 1]**2)
            Rein = np.median(dist)  #[arcsec]
    else:
        curve_crit_tan= np.array([])
        Rein = 0
    return Rein


@mpi_errchk
def create_density_maps():
    # Get command line arguments
    args = {}
    if comm_rank == 0:
        print(':Registered %d processes' % comm_size)
        args["simdir"]       = sys.argv[1]
        args["hfdir"]        = sys.argv[2]
        args["snapnum"]      = int(sys.argv[3])
        args["zs"]           = float(sys.argv[4])/10
    args = comm.bcast(args)
    label = args["simdir"].split('/')[-2].split('_')[2]
   
    # Organize devision of Sub-&Halos over Processes on Proc. 0
    if comm_rank == 0:
        # Load simulation
        s = read_hdf5.snapshot(args["snapnum"], args["simdir"])
        s.read(["Coordinates", "Masses", "GFM_StellarFormationTime"],
               parttype=[0,1,4])  #[0,1,4]
        scale = 1e-3*s.header.hubble
        
        # Define Cosmology
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
        cosmosim = {'omega_M_0' : s.header.omega_m,
                    'omega_lambda_0' : s.header.omega_l,
                    'omega_k_0' : 0.0,
                    'h' : s.header.hubble}
        redshift = s.header.redshift
        print(': Redshift: %f' % redshift)
        
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
        hist_edges =  procdiv.histedges_equalN(sh_x, comm_size)
        SH = procdiv.cluster_subhalos(sh_id, sh_vrms, sh_x, sh_y, sh_z, hist_edges, comm_size)
      
        # Calculate overlap for particle cuboids
        c = (const.c).to_value('km/s')
        fov_rad = 4*np.pi*(np.percentile(SH['Vrms'], 90)/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        alpha = 6  # multiplied by 4 because of Oguri&Marshall
        overlap = 0.5*alpha*fov_rad*sh_dist  #[Mpc] half of field-of-view
        print('Cuboids overlap is: %f [Mpc]' % overlap)

	# Sort Particles over Processes
        ## Dark Matter
        DM = {'Mass' : (s.data['Masses']['dm']).astype('float64'),
              'Pos' : (s.data['Coordinates']['dm']*scale).astype('float64')}
        DM = procdiv.cluster_particles(DM, hist_edges, comm_size)
        ## Gas
        Gas = {'Mass' : (s.data['Masses']['gas']).astype('float64'),
               'Pos' : (s.data['Coordinates']['gas']*scale).astype('float64')}
        Gas = procdiv.cluster_particles(Gas, hist_edges, comm_size)
        ## Stars
        age = (s.data['GFM_StellarFormationTime']['stars']).astype('float64')
        Star = {'Mass' : (s.data['Masses']['stars'][age >= 0]).astype('float64'),
                'Pos' : (s.data['Coordinates']['stars'][age >= 0, :]*scale).astype('float64')}
        del age
        Star = procdiv.cluster_particles(Star, hist_edges, comm_size)

    else:
        c=None; alpha=None; overlap=None
        cosmosim=None; cosmo=None; redshift=None; hist_edges=None;
        SH = {'ID':None, 'Vrms':None, 'X':None, 'Y':None, 'Z':None,
              'split_size_1d':None, 'split_disp_1d':None}
        DM = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
              'split_size_1d':None, 'split_disp_1d':None}
        Gas = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
               'split_size_1d':None, 'split_disp_1d':None}
        Star = {'Mass':None, 'X':None, 'Y':None, 'Z':None,
                'split_size_1d':None, 'split_disp_1d':None}
      
    # Broadcast variables over all processors
    sh_split_size_1d = comm.bcast(SH['split_size_1d'], root=0)
    dm_split_size_1d = comm.bcast(DM['split_size_1d'], root=0)
    gas_split_size_1d = comm.bcast(Gas['split_size_1d'], root=0)
    star_split_size_1d = comm.bcast(Star['split_size_1d'], root=0)
    c = comm.bcast(c, root=0)
    alpha = comm.bcast(alpha, root=0)
    overlap = comm.bcast(overlap, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    redshift = comm.bcast(redshift, root=0)
    hist_edges = comm.bcast(hist_edges, root=0)

    SH = procdiv.scatter_subhalos(SH, sh_split_size_1d,
                                  comm_rank, comm, root_proc=0)
    DM = procdiv.scatter_particles(DM, dm_split_size_1d,
                                   comm_rank, comm, root_proc=0)
    Gas = procdiv.scatter_particles(Gas, gas_split_size_1d,
                                    comm_rank, comm, root_proc=0)
    Star = procdiv.scatter_particles(Star, star_split_size_1d,
                                     comm_rank, comm, root_proc=0)
    print(': Proc. %d got: \n\t %d Sub-&Halos \n\t %d dark matter \n\t %d gas \n\t %d stars \n' % (comm_rank, int(sh_split_size_1d[comm_rank]), int(dm_split_size_1d[comm_rank]), int(gas_split_size_1d[comm_rank]), int(star_split_size_1d[comm_rank])))
   
    ## Run over Sub-&Halos
    zl = redshift
    zs = args["zs"]
    ncells = [512, 256, 128]
    nparts = [1, 2, 4, 8]
    M200 = np.ones(len(SH['ID']))
    ID = np.ones(len(SH['ID']))
    Rein = np.ones((len(SH['ID']), len(ncells), len(nparts)))
    for ll in range(len(SH['ID'])):
        # Define field-of-view
        fov_rad = 4*np.pi*(SH['Vrms'][ll]/c)**2
        sh_dist = (cosmo.comoving_distance(redshift)).to_value('Mpc')
        fov_Mpc = alpha*fov_rad*sh_dist  #[Mpc] is it the diameter?
        fov_arc = (fov_Mpc/cf.Da(zl, cosmo)*u.rad).to_value('arcsec')
        sigma_cr = sigma_crit(zl, zs, cosmo).to_value('Msun Mpc-2')

        # Check cuboid boundary condition,
        # that all surface densities are filled with particles
        if ((SH['Pos'][ll,0]-hist_edges[comm_rank] < overlap) or
                (hist_edges[comm_rank+1]-overlap < \
                 SH['Pos'][ll,0]-hist_edges[comm_rank])):
            if fov_Mpc*0.45 > overlap:
                print("FOV is bigger than cuboids overlap: %f > %f" % \
                        (fov_Mpc*0.45, overlap))
                continue

        ## Run over different Ncells
        for cc in range(len(ncells)):
            dsx_arc = fov_arc/ncells[cc]  #[arcsec] pixel size
            
            ## Run over particle reductions
            for mm in range(len(nparts)):
                smlpixel = 20  # maximum smoothing pixel length
                pos, indx = dmap.select_particles(
                        Gas['Pos'], SH['Pos'][ll], #*a/h,
                        fov_Mpc, 'box')
                gas_sigma = dmap.projected_density_pmesh_adaptive(
                        pos[::nparts[mm],:],
                        Gas['Mass'][indx][::nparts[mm]],
                        fov_Mpc,
                        ncells[cc],
                        hmax=smlpixel)
                pos, indx = dmap.select_particles(
                        Star['Pos'], SH['Pos'][ll], #*a/h,
                        fov_Mpc, 'box')
                star_sigma = dmap.projected_density_pmesh_adaptive(
                        pos[::nparts[mm],:],
                        Star['Mass'][indx][::nparts[mm]],
                        fov_Mpc,
                        ncells[cc],
                        hmax=smlpixel)
                pos, indx = dmap.select_particles(
                        DM['Pos'], SH['Pos'][ll], #*a/h,
                        fov_Mpc, 'box')
                dm_sigma = dmap.projected_density_pmesh_adaptive(
                        pos[::nparts[mm],:],
                        DM['Mass'][indx][::nparts[mm]],
                        fov_Mpc,  #[Mpc]
                        ncells[cc],
                        hmax=smlpixel)
                tot_sigma = dm_sigma+gas_sigma+star_sigma
       
                # Make sure that density-map is filled
                while 0.0 in tot_sigma:
                    smlpixel += 5
                    dm_sigma = dmap.projected_density_pmesh_adaptive(
                            pos[::nparts[mm],:],
                            DM['Mass'][indx][::nparts[mm]],
                            fov_Mpc,  #[Mpc]
                            ncells[cc],
                            hmax=smlpixel)
                    tot_sigma = dm_sigma+gas_sigma+star_sigma
                #tmap.plotting(tot_sigma, ncells[cc], fov_Mpc, zl)

                # initialize the coordinates of grids (light rays on lens plan)
                lpv = np.linspace(-(fov_arc-dsx_arc)/2,
                                  (fov_arc-dsx_arc)/2, ncells[cc])
                lp1, lp2 = np.meshgrid(lpv, lpv)  #[arcsec]

                fig = plt.figure()
                ax = fig.add_subplot(111)
                # Calculate convergence map
                kappa = tot_sigma/sigma_cr
                
                # Calculate Deflection Maps
                alpha1,alpha2,mu_map,phi,detA,lambda_t = cal_lensing_signals(kappa,
                                                                             fov_arc,
                                                                             ncells[cc]) 
                # Calculate Einstein Radii
                Rein[ll, cc, mm] = einstein_radii(lp1, lp2, detA, lambda_t,
                                                  zl, cosmo, ax, 'med', ll)
                #print('Rein = %f' % Rein[ll, cc, mm])
                ID[ll] = SH['ID'][ll]
                #plt.close(fig)
    output = {}
    for cc in range(len(ncells)):
        for mm in range(len(nparts)):
            output[(str(ncells[cc]), str(nparts[mm]))] = Rein[:,cc,mm]
    df = pd.DataFrame.from_dict(output)
    df['ID'] = ID
    #self.df = pd.concat([self.df, dfp], axis=1)
    fname = 'DMConvTest_'+label+'_'+str(comm_rank)+'_zs150.h5'
    df.to_hdf(fname, key='Rein', mode='w')
    plt.close(fig)

if __name__ == "__main__":
    create_density_maps()

