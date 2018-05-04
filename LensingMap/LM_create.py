from __future__ import division
import numpy as np
import os
import sys
import scipy
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
import matplotlib.pyplot as plt
import h5py
import time
import lm_tools as LI
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import readsnap
import readlensing as rf

# Works only with Python 2.7.~
print("Python version:", sys.version)
print("Numpy version:", np.version.version)
print("Scipy version:", scipy.version.version)


############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, dd, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

###########################################################################
# Define Lensing Map Parameters
# lensing map parameters
Ncells = 1024  # devide density map into cells
Lrays = 2.0*u.Mpc  # Length of, to calculate alpha from kappa
Nrays = 1024  # Number of, to calculate alpha from kappa
save_maps = True

###########################################################################
# Run through simulations
for sim in range(len(sim_dir))[:1]:
    print('Create lensing map for: ', sim_name[sim])
    # File for lens & source properties
    lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    # Simulation Snapshots
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    
    # Units of Simulation
    scale = rf.simulation_units(sim_dir[sim])

    # Cosmological Parameters
    snap_tot_num = 45  # read output_list_new.txt
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    cosmo = {'omega_M_0' : header.omega_m,
             'omega_lambda_0' : header.omega_l,
             'omega_k_0' : 0.0,
             'h' : header.hubble}
    h = header.hubble
    a = 1/(1 + header.redshift)

    # Load LightCone Contents
    LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')
    Src_ID = LC['Src_ID']
    Src_z = LC['Src_z']

    # Sort Lenses according to Snapshot Number (snapnum)
    indx = np.argsort(LC['snapnum'])
    Halo_ID= LC['Halo_ID'][indx]
    Halo_z = LC['Halo_z'][indx]
    M200 = LC['M200'][indx]
    Rvir = LC['Rvir'][indx]
    snapnum = LC['snapnum'][indx]
    HaloPosBox = LC['HaloPosBox'][indx]

    first_lens = 201
    previous_snapnum = snapnum[first_lens]
    # Run through lenses
    for ll in range(len(Halo_ID))[first_lens:202]:
        print('Lens System nr.: ', ll, 'and ID: ', Halo_ID[ll])
        zs = LI.source_selection(LC['Src_ID'], LC['Src_z'], Halo_ID[ll]) 
        zl = Halo_z[ll]
        Lbox = Rvir[ll]*0.3*u.Mpc
        FOV = Lbox.to_value('Mpc')  #[Mpc]

        # Only load new particle data if lens is at another snapshot
        if (previous_snapnum != snapnum[ll]) or (ll == first_lens):
            print('Load Particle Data', snapnum[ll])
            snap = snapfile % (snapnum[ll], snapnum[ll])
            # 0 Gas, 1 DM, 4 Star[Star=+time & Wind=-time], 5 BH
            DM_pos = readsnap.read_block(snap, 'POS ', parttype=1)*scale
            DM_mass = readsnap.read_block(snap, 'MASS', parttype=1)*1e10/h
            Gas_pos = readsnap.read_block(snap, 'POS ', parttype=0)*scale
            Gas_mass = readsnap.read_block(snap, 'MASS', parttype=0)*1e10/h
            Star_pos = readsnap.read_block(snap, 'POS ', parttype=4)*scale
            Star_age = readsnap.read_block(snap, 'AGE ', parttype=4)
            Star_mass = readsnap.read_block(snap, 'MASS', parttype=4)
            Star_pos = Star_pos[Star_age >= 0]
            Star_mass = Star_mass[Star_age >= 0]*1e10/h
            del Star_age
            BH_pos = readsnap.read_block(snap, 'POS ', parttype=5)*scale
            BH_mass = readsnap.read_block(snap, 'MASS', parttype=5)*1e10/h
        previous_snapnum = snapnum[ll] 

        #start_time = time.time()
        DM_sigma, xs, ys = LI.projected_surface_density(DM_pos,
                                                        DM_mass,
                                                        HaloPosBox[ll],
                                                        fov=FOV,
                                                        bins=Ncells,
                                                        smooth=False,
                                                        smooth_fac=0.5,
                                                        neighbour_no=32)
        Gas_sigma, xs, ys = LI.projected_surface_density(Gas_pos, #*a/h,
                                                         Gas_mass,
                                                         HaloPosBox[ll], #*a/h,
                                                         fov=FOV,
                                                         bins=Ncells,
                                                         smooth=False,
                                                         smooth_fac=0.5,
                                                         neighbour_no=32)
        Star_sigma, xs, ys = LI.projected_surface_density(Star_pos, #*a/h,
                                                          Star_mass,
                                                          HaloPosBox[ll], #*a/h,
                                                          fov=FOV,
                                                          bins=Ncells,
                                                          smooth=False,
                                                          smooth_fac=0.5,
                                                          neighbour_no=8)
        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        tot_sigma = DM_sigma + Gas_sigma + Star_sigma

        # Calculate critical surface density
        sigma_cr = LI.sigma_crit(zl, zs).to_value('Msun Mpc-2')
        kappa = tot_sigma/sigma_cr

        # Calculate deflection angle
        xi0 = 0.001  # Mpc
        alphax, alphay, detA, xrays, yrays, lambda_t, lambda_r = LI.alpha_from_kappa(kappa, xs, ys,
                                                                           xi0, Nrays,
                                                                           Lrays.to_value('Mpc'))
        xraysgrid, yraysgrid = np.meshgrid(xrays,yrays,indexing='ij')
        # Mapping light rays from image plane to source plan
        xrayssource, yrayssource = xraysgrid - alphax, yraysgrid - alphay

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        critical_curves = ax.contour(xraysgrid*xi0, yraysgrid*xi0, detA,
                                    levels=(0,), colors='r',
                                    linewidths=1.5, zorder=200)
        Ncrit = len(critical_curves.allsegs[0])
        crit_curves = critical_curves.allsegs[0]
        tangential_critical_curves = ax.contour(xraysgrid*xi0, yraysgrid*xi0, lambda_t,
                                               levels=(0,), colors='r',
                                               linewidths=1.5, zorder=200)
        Ncrit_tan = len(tangential_critical_curves.allsegs[0])
        if Ncrit_tan > 0:
           len_tan_crit = np.zeros(Ncrit_tan)
           for i in range(Ncrit_tan):
              len_tan_crit[i] = len(tangential_critical_curves.allsegs[0][i])
           tangential_critical_curve = tangential_critical_curves.allsegs[0][len_tan_crit.argmax()]
           eqv_einstein_radius = ((np.sqrt(np.abs(LI.area(tangential_critical_curve))/ \
                                           np.pi)*u.Mpc/ \
                                   cosmo.angular_diameter_distance(zl))* \
                                  u.rad).to_value('arcsec')
        else:
           tangential_critical_curve = np.array([])
           eqv_einstein_radius = 0
        print('Einstein Radius', eqv_einstein_radius)

        ########## Save to File ########
        # xs, ys in Mpc in lens plane, kappa measured on that grid
        # xrays, yrays, alphax, alphay in dimensionless coordinates
        if save_maps == True:
            lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
            LI.ensure_dir(lm_dir)
            filename = lm_dir+'LM_'+str(Halo_ID[ll])+'.h5'
            
            LensPlane = [xs, ys]
            RaysPos = [xrays, yrays]
            alpha = [alphax, alphay]

            hf = h5py.File(filename, 'w')
            hf.create_dataset('HaloPosBox', data=HaloPosBox)
            hf.create_dataset('zs', data=zs)
            hf.create_dataset('zl', data=zl)
            hf.create_dataset('Grid', data=LensPlane)
            hf.create_dataset('RaysPos', data=RaysPos)
            hf.create_dataset('DM_sigma', data=DM_sigma)
            hf.create_dataset('Gas_sigma', data=Gas_sigma)
            hf.create_dataset('Star_sigma', data=Star_sigma)
            hf.create_dataset('kappa', data=kappa)
            hf.create_dataset('alpha', data=alpha)
            hf.create_dataset('detA', data=detA)
            hf.create_dataset('Ncrit', data=Ncrit)
            try:
                hf.create_dataset('crit_curves', data=crit_curves)
            except:
                cc = hf.create_group('crit_curve')
                for k, v in enumerate(crit_curves):
                    cc.create_dataset(str(k), data=v)
            hf.create_dataset('tangential_critical_curves',
                              data=tangential_critical_curve)
            hf.create_dataset('eqv_einstein_radius', data=eqv_einstein_radius)
            hf.close()
        plt.close(fig)
