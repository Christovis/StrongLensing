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
        dm_mass = (s.data['Masses']['dm']).astype('float64')
        dm_x = (s.data['Coordinates']['dm'][:, 0]*scale).astype('float64')
        dm_y = (s.data['Coordinates']['dm'][:, 1]*scale).astype('float64')
        dm_z = (s.data['Coordinates']['dm'][:, 2]*scale).astype('float64')
        dm_mass, dm_x, dm_y, dm_z, dm_split_size_1d, dm_split_disp_1d = procdiv.cluster_particles(
                dm_mass, dm_x, dm_y, dm_z, hist_edges, comm_size)
        ### Gas
        gas_mass = (s.data['Masses']['gas']).astype('float64')
        gas_x = (s.data['Coordinates']['gas'][:, 0]*scale).astype('float64')
        gas_y = (s.data['Coordinates']['gas'][:, 1]*scale).astype('float64')
        gas_z = (s.data['Coordinates']['gas'][:, 2]*scale).astype('float64')
        gas_mass, gas_x, gas_y, gas_z, gas_split_size_1d, gas_split_disp_1d = procdiv.cluster_particles(gas_mass, gas_x, gas_y, gas_z, hist_edges, comm_size)
        ### Stars
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
        star_mass, star_x, star_y, star_z, star_split_size_1d, star_split_disp_1d = procdiv.cluster_particles(star_mass, star_x, star_y, star_z, hist_edges, comm_size)

    else:
        c=None; alpha=None; overlap=None
        cosmosim=None; cosmo=None; redshift=None; hist_edges=None;
        #sh_id=None; sh_vrms=None; sh_x=None; sh_y=None; sh_z=None
        #sh_split_size_1d=None; sh_split_disp_1d=None
        SH = {'split_size_1d':None, 'split_disp_1d':None,
              'ID':None, 'Vrms':None, 'X':None, 'Y':None, 'Z':None}
        dm_mass=None; dm_x=None; dm_y=None; dm_z=None
        dm_split_size_1d=None; dm_split_disp_1d=None
        gas_mass=None; gas_x=None; gas_y=None; gas_z=None
        gas_split_size_1d=None; gas_split_disp_1d=None
        star_mass=None; star_x=None; star_y=None; star_z=None
        star_split_size_1d=None; star_split_disp_1d=None
      
    # Broadcast variables over all processors
    sh_split_size_1d = comm.bcast(SH['split_size_1d'], root=0)
    sh_split_disp_1d = comm.bcast(SH['split_disp_1d'], root=0)
    dm_split_size_1d = comm.bcast(dm_split_size_1d, root=0)
    dm_split_disp_1d = comm.bcast(dm_split_disp_1d, root=0)
    gas_split_size_1d = comm.bcast(gas_split_size_1d, root=0)
    gas_split_disp_1d = comm.bcast(gas_split_disp_1d, root=0)
    star_split_size_1d = comm.bcast(star_split_size_1d, root=0)
    star_split_disp_1d = comm.bcast(star_split_disp_1d, root=0)
    c = comm.bcast(c, root=0)
    alpha = comm.bcast(alpha, root=0)
    overlap = comm.bcast(overlap, root=0)
    cosmo = comm.bcast(cosmo, root=0)
    redshift = comm.bcast(redshift, root=0)
    hist_edges = comm.bcast(hist_edges, root=0)

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
    comm.Scatterv([SH['ID'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_id_local, root=0)
    comm.Scatterv([SH['Vrms'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_vrms_local, root=0)
    comm.Scatterv([SH['X'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_x_local,root=0) 
    comm.Scatterv([SH['Y'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
                  sh_y_local,root=0) 
    comm.Scatterv([SH['Z'], SH['split_size_1d'], SH['split_disp_1d'], MPI.DOUBLE],
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

    comm.Barrier()

    SH = {"ID"   : sh_id_local,
          "Vrms" : sh_vrms_local,
          "Pos"  : np.transpose([sh_x_local, sh_y_local, sh_z_local])}
    DM = {"Mass"  : dm_mass_local,
          "Pos" : np.transpose([dm_x_local, dm_y_local, dm_z_local])}
    Gas = {"Mass"  : gas_mass_local,
           "Pos" : np.transpose([gas_x_local, gas_y_local, gas_z_local])}
    Star = {"Mass"  : star_mass_local,
            "Pos" : np.transpose([star_x_local, star_y_local, star_z_local])}
   
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
                print('oarticle ratio', len(DM['Mass'][:])/len(DM['Mass'][::nparts[mm]]))
                smlpixel = 20  # maximum smoothing pixel length
                indx = dmap.select_particles(Gas['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
                gas_sigma = dmap.projected_density_pmesh_adaptive(
                        Gas['Pos'][indx,:][::nparts[mm],:],
                        Gas['Mass'][indx][::nparts[mm]],
                        SH['Pos'][ll], #*a/h,
                        fov_Mpc,
                        ncells[cc],
                        hmax=smlpixel,
                        particle_type=0)
                indx = dmap.select_particles(Star['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
                star_sigma = dmap.projected_density_pmesh_adaptive(
                        Star['Pos'][indx,:][::nparts[mm],:],
                        Star['Mass'][indx][::nparts[mm]],
                        SH['Pos'][ll], #*a/h,
                        fov_Mpc,
                        ncells[cc],
                        hmax=smlpixel,
                        particle_type=4)
                indx = dmap.select_particles(DM['Pos'], SH['Pos'][ll], fov_Mpc, 'box')
                dm_sigma = dmap.projected_density_pmesh_adaptive(
                        DM['Pos'][indx,:][::nparts[mm],:],
                        DM['Mass'][indx][::nparts[mm]],
                        SH['Pos'][ll],
                        fov_Mpc,  #[Mpc]
                        ncells[cc],
                        hmax=smlpixel,
                        particle_type=1)
                tot_sigma = dm_sigma+gas_sigma+star_sigma
       
                # Make sure that density-map is filled
                while 0.0 in tot_sigma:
                    smlpixel += 5
                    dm_sigma = dmap.projected_density_pmesh_adaptive(
                            DM['Pos'][indx,:][::nparts[mm],:],
                            DM['Mass'][indx][::nparts[mm]],
                            SH['Pos'][ll],
                            fov_Mpc,  #[Mpc]
                            ncells[cc],
                            hmax=smlpixel,
                            particle_type=1)
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

