from __future__ import division
import sys
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import numpy as np
import readsnap
import SLSNreadfile as rf
import LensInst as LI
import cfuncs as cf
import astropy
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt, rc
from astropy.cosmology import Planck15
from astropy import cosmology
from astropy import units as u
import h5py

rc('figure', figsize=(8,6))
rc('font', size=18)
rc('lines', linewidth=3)
rc('axes', linewidth=2)
rc('xtick.major', width=2)
rc('ytick.major', width=2)


def list_to_array(lis, element_len):
    array = np.empty([len(element_len), np.max(element_len)])
    array[:] = np.nan
    for i,j in enumerate(lis):
        if len(j) > 1:
            print(np.shape(array), i, len(j))
        array[i][0:len(j)] = j
    return array


def halo_mass(rr, delta_crit, rho_crit, rrsc):
    rho_NFW = (rho_crit*delta_crit)/((rr/rrsc)*(1 + rr/rrsc)**2)
    volume = 4/3*np.pi*rr**3
    return rho_NFW*volume  # [Msun]


def rEhistogram(my_radius_E):
    plt.hist(my_radius_E, 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel(r'$r_{E} \quad [kpc]$')
    plt.ylabel('Occurances')
    plt.savefig('EinsteinRadius_hist.png', bbox_inches='tight')
    plt.clf()
    print('histogram done')


def cal_lensing_signals(kap, bzz, ncc):
    # deflection maps
    alpha1, alpha2 = cf.call_cal_alphas(kap, bzz, ncc)
    # shear maps
    npad = 5
    al11, al12, al21, al22 = cf.call_lanczos_derivative(alpha1, alpha2, bzz, ncc)
    shear1 = 0.5*(al11-al22)
    shear2 = 0.5*(al21+al12)
    shear1[:npad, :]=0.0; shear1[-npad:,:]=0.0
    shear1[:, :npad]=0.0; shear1[:,-npad:]=0.0
    shear2[:npad, :]=0.0; shear2[-npad:,:]=0.0
    shear2[:, :npad]=0.0; shear2[:,-npad:]=0.0;
    # magnification maps
    mu = 1.0/((1.0-kap)**2.0-shear1*shear1-shear2*shear2)
    # lensing potential
    phi = cf.call_cal_phi(kap, bzz, ncc)
    return alpha1, alpha2, mu, phi


def einstein_radius(ai1, ai2, ds, lp1, lp2):
    """
    Input:
        ai1, ai2: Deflection angles
        lp1, lp2: Grid coordinates on image plane
        ds: Grid-cell side length
    Output:
        Einstein Radius [arcsec]
    """
    # calculate Einstein radius of lense
    A11 = 1 - np.gradient(ai1, ds, axis=0)
    A12 = - np.gradient(ai1, ds, axis=1)
    A21 = - np.gradient(ai2, ds, axis=0)
    A22 = 1 - np.gradient(ai2, ds, axis=1)
    ka = 1 - 0.5*(A11 + A22)
    ga1 = 0.5*(A22 - A11)
    ga2 = -0.5*(A12 + A21)
    ga = (ga1**2 + ga2**2)**0.5
    lambda_t = 1 - ka - ga
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    tangential_critical_curves = ax.contour(lp1, lp2, lambda_t, levels=(0,),
                                            colors='r',linewidths=1.5, zorder=200)
    Ncrit_tan = len(tangential_critical_curves.allsegs[0])

    if Ncrit_tan > 0:
        len_tan_crit = np.zeros(Ncrit_tan)
        for i in range(Ncrit_tan):
            len_tan_crit[i] = len(tangential_critical_curves.allsegs[0][i])
        tangential_critical_curve = tangential_critical_curves.allsegs[0][len_tan_crit.argmax()]
        eqv_einstein_radius = np.sqrt(np.abs(LI.area(tangential_critical_curve))/np.pi)
    else:
        tangential_critical_curve = np.array([])
        eqv_einstein_radius = 0
    plt.close(fig)
    return eqv_einstein_radius 

#####################################################################
# Load halo lensing properties
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, hf_dir, lc_dir, glafic_dir, HQ_dir = rf.Simulation_Specs(LCSettings)

# set parameters of the images simulation
Ncells = 1024   # number of pixels per side
###########################################################################
# Run through simulations
for sim in range(len(sim_dir))[1:]:
    print(sim_dir[sim])
    snapfile = sim_dir[sim]+'/snapdir_%03d/snap_%03d'
    LensPropFile = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
    LensingPath = HQ_dir[0]+'lensing/'+sim_phy[sim]+sim_name[sim]
    LensPropPath = LensingPath + '/lens_properties/'
    KappaPath = LensingPath + '/kappa/'
    LC = rf.LightCone_with_SN_lens(LensPropFile, 'dictionary')
    LensID = LC['Halo_ID']
    # Find snapshot redshifts
    snap_tot_num = 45
    header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))

    # Initialize Lists to save
    R_E_list=[]; delta_t_list=[]; mu_list=[]; n_img_list=[]
    
    txt_file = open('LensProp_'+sim_name[sim]+'.txt', 'w')  # w=write_new a=add_on
    txt_file.write('ID, time_delay[days], magnification[]\n')
    # Loop through Lenses 
    for ll in range(len(LensID))[0:]:
        #if ll in F6[6, 30, 102]: GR[67, 69]
        #    continue
        print('Lense Nr.:', ll)
        zl = LC['Halo_z'][ll]
        Src_indx = np.where(LC['Src_ID'] == LensID[ll])
        indx = np.argmax(LC['Src_z'][Src_indx[0]])
        zs = LC['Src_z'][Src_indx[0][indx]]
        SrcPosSky = LC['SrcPosSky'][Src_indx[0][indx]]  #[arcsec]
        #print('zs', zs, 'zl', zl)

        KappaFile = h5py.File(KappaPath+'Lens_'+str(ll)+'.h5')
        kappa_plane_pos = KappaFile['lens_plane_pos'].value
        kappa = KappaFile['kappa'].value
        LensPropFile = h5py.File(LensPropPath+'Lens_'+str(ll)+'.h5')
        einstein_angle = LensPropFile['eqv_einstein_radius'].value  #[arcsec]
        FOV = LC['Rvir'][ll]*0.1  #[Mpc]
        #FOV = einstein_angle*3
        #FOV_arc = LC['FOV'][ll]  #[arcsec]???

        #converting box size and pixels size from co-moving distance to arcsec
        FOV_arc = FOV/cf.Dc(zl)*cf.apr     #[arcsec] box size
        dsx_arc = FOV_arc/Ncells           #[arcsec] pixel size
        # initialize the coordinates of grids (light rays on lens plan)
        lp1, lp2 = cf.make_r_coor(FOV_arc, Ncells)
        ds = FOV_arc/Ncells  # grid-cell edge length
        #print('SrcPosSky', SrcPosSky, 'FOV_arc', FOV_arc)
        # Calculate the maps of deflection angles, magnifications, and lensing potential
        kappa = gaussian_filter(kappa,sigma=3)
        ai1, ai2, mua, phia = cal_lensing_signals(kappa, FOV_arc, Ncells)
        # Mapping light rays from image plane to source plan
        sp1=lp1-ai1; sp2=lp2-ai2  #[arcsec]
        # Source position [arcsec]
        beta1=SrcPosSky[0]; beta2=SrcPosSky[1]
        # Looking for the postions of lensed Supernovae
        theta1, theta2 = cf.call_mapping_triangles([beta1, beta2], lp1, lp2, sp1, sp2)
        # calculate magnifications of lensed Supernovae
        mu = cf.call_inverse_cic_single(mua, 0.0, 0.0, theta1, theta2, dsx_arc)
        mu_list.append(mu)
        n_img_list.append(len(mu)) # number of lensed images
        #print 'number of lensed images', n_img_list
        # calculate time delays of lensed Supernovae in Days
        prts = cf.call_inverse_cic_single(phia,0.0,0.0,theta1,theta2,dsx_arc)
        Kc = (1.0+zl)/cf.vc*(cf.Da(zl)*cf.Da(zs)/cf.Da2(zl,zs)) * \
              cf.Mpc_h/(1e3*cf.Day*cf.apr*cf.apr)
        delta_t = Kc*(0.5*((theta1-beta1)**2.0+(theta2-beta2)**2.0)-prts)
        delta_t_list.append(delta_t)

        for jj in range(len(mu)):
            txt_file.write('%d %f %f \n' % (ll, delta_t[jj], mu[jj]))
        #print 'Time Delays: ', delta_t_list, 'in  days'

        #R_E_list.append(einstein_radius(ai1, ai2, ds, lp1, lp2))  #[arcsec]
        #print(einstein_angle, R_E)
        #print('*****************************************')
    
    txt_file.close()
    # Change Lists to Arrays
    #n_img = np.asarray(n_img_list)
    #R_E = list_to_array(R_E_list, n_img)
    #delta_t = list_to_array(delta_t_list, n_img)
    #mu = list_to_array(mu_list, n_img)

    #print('write file')
    #hf = h5py.File('LensProp_'+sim_name[sim]+'.h5', 'w')
    #hf.create_dataset('Einstein_radius', data=R_E)  #[arcsec]
    #hf.create_dataset('time_delay', data=delta_t)  #[Days]
    #hf.create_dataset('magnification', data=mu)  #[]
    #hf.create_dataset('num_lensed_imgs', data=n_img)  #[]
    #hf.close()
    print('written')
    break
