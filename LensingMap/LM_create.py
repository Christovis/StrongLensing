#!/usr/bin/env python
from __future__ import division
import numpy as np
import os
import sys
import scipy
from scipy.ndimage.filters import gaussian_filter
from astropy import units as u
from astropy.cosmology import Planck15
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
num_of_sim = 1
data = open('/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt', 'r')
Settings = data.readlines()
sim_dir = []
sim_phy = []
sim_name = []
sim_col = []
sim_units = []
lc_dir = []
HQ_dir = []
glafic_dir = []
for k in range(len(Settings)):
    if 'Python' in Settings[k].split():
        HQdir = Settings[k+1].split()[0]
        HQ_dir.append(HQdir)
    if 'Simulation' in Settings[k].split():
        [simphy, simname] = Settings[k+1].split()
        sim_phy.append(simphy)
        sim_name.append(simname)
        [simdir, simcol, simunit] = Settings[k+2].split()
        sim_dir.append(simdir)
        sim_col.append(simcol)
        sim_units.append(simunit)
        lcdir = Settings[k+4].split()[0]
        lc_dir.append(lcdir)
        glaficdir = Settings[k+5].split()[0]
        glafic_dir.append(glaficdir)

###########################################################################
# Define Lensing Map Parameters
'''
# source parameters
Lsourcephys = 0.1*u.Mpc # in lens plane
Nsource = 32
refinements = 2
rsourcephys = 2.0*u.kpc
'''
# lensing map parameters
Ncells = 1024  # devide density map into cells
Lrays = 2.0*u.Mpc  # Length of, to calculate alpha from kappa
Nrays = 1024  # Number of, to calculate alpha from kappa
save_maps = True
###########################################################################
# Run through simulations
for sim in range(len(sim_dir))[:1]:
    # File for lens & source properties
    LensPropFile = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    print(LensPropFile)
    # To get header info 
    snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
    # Folder for all lensing properties of lenses per simulation
    LensingPath = HQ_dir[0]+'lensing/'+sim_phy[sim]+'/'+sim_name[sim]
    LI.ensure_dir(LensingPath)
    # Folder for density surfaces
    SigmaPath = LensingPath + '/sigma/'
    LI.ensure_dir(SigmaPath)
    # Folder for critical density
    KappaPath = LensingPath + '/kappa/'
    LI.ensure_dir(KappaPath)
    # Folder for deflection angle
    AlphaPath = LensingPath + '/alpha/'
    LI.ensure_dir(AlphaPath)
    # Folder for lens properties
    LensPropertiesPath = LensingPath + '/lens_properties/'
    LI.ensure_dir(LensPropertiesPath)
    # Units of Simulation
    if sim_units[sim] is 'kpc':
        print('yes')
        scale = 1e-3
    elif sim_units[sim] is 'Mpc':
        print('no')
        scale = 1
    scale = 1e-3

    # Snapshot numbers
    snap_tot_num = 45
    snapnums = np.arange(snap_tot_num+1)
    
    # Cosmological Constants
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
    cosmo = {'omega_M_0' : header.omega_m,
             'omega_lambda_0' : header.omega_l,
             'omega_k_0' : 0.0,
             'h' : header.hubble}
    h = header.hubble
    a = 1/(1 + header.redshift)

    # Load Lenses
    #LensID, Lensz, LensMvir, LensRvir, LensPos, \
    #RockstarID, SnapNM = lens_files(LensPropFile[0])
    LC = rf.LightCone_with_SN_lens(LensPropFile, 'dictionary')
    # Load Sources
    #SrcID, Srcz, SrcPos = src_files(SrcPropFile[0])
    SrcID = LC['Src_ID']
    Srcz = LC['Src_z']

    # Sort Lenses according to Snapshot Number (SnapNM)
    indx = np.argsort(LC['snapnum'])
    LensID = LC['Halo_ID'][indx]
    Lensz = LC['Halo_z'][indx]
    LensMvir = LC['M200'][indx]
    LensRvir = LC['Rvir'][indx]
    SnapNM = LC['snapnum'][indx]
    LensPos = LC['HaloPosBox'][indx]

    first_lens = 201
    previous_SnapNM = SnapNM[first_lens]
    # Run through lenses
    for ll in range(len(LensID))[first_lens:202]:
        print('ll:', ll, 'Lens z:', Lensz[ll], 'Lens Rvir:', LensRvir[ll], LensMvir[ll]*1e-10)
        # Choose Sources at highest redshift for each Lens
        Src_indx = np.where(SrcID == LensID[ll])
        indx = np.argmax(Srcz[Src_indx[0]])
        zSource = Srcz[Src_indx[0][indx]]
        zLens = Lensz[ll]
        #Lbox = RE_kpc[Src_indx[0][indx]]*1.5*u.Mpc
        Lbox = LensRvir[ll]*0.3*u.Mpc
        FOV = Lbox.to_value('Mpc')  #[Mpc]
        print('zl zs', zLens, zSource)
        print('CoP', LensPos[ll])

        # Only load new particle data if lens is at another snapshot
        if (previous_SnapNM != SnapNM[ll]) or (ll == first_lens):
            print('Load Particle Data', SnapNM[ll])
            snap = snapfile % (SnapNM[ll], SnapNM[ll])
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
        previous_SnapNM = SnapNM[ll] 

        #start_time = time.time()
        DM_sigma, xs, ys = LI.projected_surface_density(DM_pos,
                                                        DM_mass,
                                                        LensPos[ll],
                                                        fov=FOV,
                                                        bins=Ncells,
                                                        smooth=False,
                                                        smooth_fac=0.5,
                                                        neighbour_no=32)
        Gas_sigma, xs, ys = LI.projected_surface_density(Gas_pos, #*a/h,
                                                         Gas_mass,
                                                         LensPos[ll], #*a/h,
                                                         fov=FOV,
                                                         bins=Ncells,
                                                         smooth=False,
                                                         smooth_fac=0.5,
                                                         neighbour_no=32)
        Star_sigma, xs, ys = LI.projected_surface_density(Star_pos, #*a/h,
                                                          Star_mass,
                                                          LensPos[ll], #*a/h,
                                                          fov=FOV,
                                                          bins=Ncells,
                                                          smooth=False,
                                                          smooth_fac=0.5,
                                                          neighbour_no=8)
        BH_sigma, xs, ys = LI.projected_surface_density(BH_pos, #*a/h, Bahama version
                                                        BH_mass,
                                                        LensPos[ll], #*a/h,
                                                        fov=FOV,
                                                        bins=Ncells,
                                                        smooth=False,
                                                        smooth_fac=None,
                                                        neighbour_no=None)
        # point sources need to be smoothed by > 1 pixel to avoid artefacts
        tot_sigma = DM_sigma + Gas_sigma + Star_sigma + \
                    gaussian_filter(BH_sigma, 1.5, truncate=3)

        # location of most massive BH
        if len(BH_mass > 0):
            SMBH_pos = BH_pos[BH_mass.argmax()]
        else:
            SMBH_pos = np.array([-999999,-999999,-999999])
        if save_maps == False:
            LensPlane = [xs, ys]
            FilePath = SigmaPath + 'Lens_' + str(ll)  # .zfill(4)
            hf = h5py.File(FilePath+'.h5', 'w')
            hf.create_dataset('lens_plane_pos', data=LensPlane)
            hf.create_dataset('DM_sigma', data=DM_sigma)
            hf.create_dataset('Gas_sigma', data=Gas_sigma)
            hf.create_dataset('Star_sigma', data=Star_sigma)
            hf.create_dataset('BH_sigma', data=BH_sigma)
            hf.create_dataset('tot_sigma', data=tot_sigma)
            hf.create_dataset('LensPos', data=LensPos)
            hf.create_dataset('SMBH_pos', data=SMBH_pos)
            hf.close()

        # Calculate critical surface density
        sigma_cr = LI.sigma_crit(zLens, zSource).to_value('Msun Mpc-2')
        kappa = tot_sigma/sigma_cr

        # Calculate deflection angle
        xi0 = 0.001  # Mpc
        alphax, alphay, detA, xrays, yrays, lambda_t, lambda_r = LI.alpha_from_kappa(kappa, xs, ys,
                                                                           xi0, Nrays,
                                                                           Lrays.to_value('Mpc'))
        xraysgrid, yraysgrid = np.meshgrid(xrays,yrays,indexing='ij')
        # Mapping light rays from image plane to source plan
        xrayssource, yrayssource = xraysgrid - alphax, yraysgrid - alphay

        ########## Save to File ########
        ## xs, ys in Mpc in lens plane, kappa measured on that grid
        LensPlane = [xs, ys]
        FilePath = KappaPath+'Lens_'+str(ll)  # .zfill(5)
        hf = h5py.File(FilePath+'.h5', 'w')
        hf.create_dataset('lens_plane_pos', data=LensPlane)
        hf.create_dataset('kappa', data=kappa)
        hf.create_dataset('Source_z', data=Srcz)
        hf.create_dataset('Lens_z', data=Lensz)
        hf.close()
        # xrays, yrays, alphax, alphay in dimensionless coordinates
        RaysPos = [xrays, yrays]
        alpha = [alphax, alphay]
        FilePath = AlphaPath+'Lens_'+str(ll)  # .zfill(5)
        hf = h5py.File(FilePath+'.h5', 'w')
        hf.create_dataset('xi0', data=xi0)
        hf.create_dataset('RaysPos', data=RaysPos)
        hf.create_dataset('alpha', data=alpha)
        hf.create_dataset('detA', data=detA)
        hf.create_dataset('Source_z', data=Srcz)
        hf.create_dataset('Lens_z', data=Lensz)
        hf.close()

        ########### Plot ########## 
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
        #print('tang curve', tangential_critical_curves.allsegs[0])
        # Einstein radius is found by setting circle area equal to area inside
        # critical curve
        Ncrit_tan = len(tangential_critical_curves.allsegs[0])
        if Ncrit_tan > 0:
           print('Ncrit_tan > 0')
           len_tan_crit = np.zeros(Ncrit_tan)
           for i in range(Ncrit_tan):
              len_tan_crit[i] = len(tangential_critical_curves.allsegs[0][i])
           tangential_critical_curve = tangential_critical_curves.allsegs[0][len_tan_crit.argmax()]
           eqv_einstein_radius = ((np.sqrt(np.abs(LI.area(tangential_critical_curve))/ \
                                           np.pi)*u.Mpc/ \
                                   Planck15.angular_diameter_distance(zLens))* \
                                  u.rad).to_value('arcsec')
        else:
           print('Ncrit_tan <= 0')
           tangential_critical_curve = np.array([])
           eqv_einstein_radius = 0
        print('Einstein Radius', eqv_einstein_radius)

        if save_maps == True:
            ######### Save to File ########
            # xs, ys in Mpc in lens plane, crit_curves in same units
            LensPlane = [xs, ys]
            Filen = LensPropertiesPath+'Lens_'+str(ll)  # .zfill(5)
            hf = h5py.File(Filen+'.h5', 'w')
            hf.create_dataset('lens_plane_pos', data=LensPlane)
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
            hf.create_dataset('zSource', data=zSource)
            hf.close()
        plt.close(fig)
