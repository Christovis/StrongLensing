from __future__ import division
import sys
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3')
import numpy as np
import readsnap
import SLSNreadfile as rf
import astropy
from scipy.integrate import quad
from matplotlib import pyplot as plt, rc
from astropy.cosmology import WMAP9
from astropy import cosmology
from astropy import units as u
import h5py

rc('figure', figsize=(8,6))
rc('font', size=18)
rc('lines', linewidth=3)
rc('axes', linewidth=2)
rc('xtick.major', width=2)
rc('ytick.major', width=2)


#####################################################################
# Load halo lensing properties
data = open('/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt', 'r')
Settings = data.readlines()
sim_dir = []
sim_phy = []
sim_name = []
sim_col = []
hf_dir = []
lc_dir = []
glafic_dir = []
HQ_dir = []
for k in range(len(Settings)):
    if 'Python' in Settings[k].split():
        HQdir = Settings[k+1].split()[0]
        HQ_dir.append(HQdir)
    if 'Simulation' in Settings[k].split():
        [simphy, simname] = Settings[k+1].split()
        sim_phy.append(simphy)
        sim_name.append(simname)
        [simdir, simcol] = Settings[k+2].split()
        sim_dir.append(simdir)
        sim_col.append(simcol)
        [hfdir, hf_name] = Settings[k+3].split()
        hf_dir.append(hfdir)
        lcdir = Settings[k+4].split()[0]
        lc_dir.append(lcdir)
        glaficdir = Settings[k+5].split()[0]
        glafic_dir.append(glaficdir)

analytic = False  # NFW-profile
profile_fitting = False  # Particle Profile
count_particles_in_Rth = True
plot = False

G = 6.67408e-20  # [km^3/kg^1/s^2]
kg_to_Msun = 1.989e30  # [kg]
km_to_Mpc = 3.086e19
###########################################################################
# Run through simulations
for sim in range(len(sim_dir)):
    outfile = glafic_dir[sim] + 'outlens_' + sim_name[sim] + '.dat'
    [id_lens, my_radius_E, theta_E, dd, redshift, dd, Mlens, dd, dd, subpos, dd, dd, dd, dd, dd] = rf.Glafic_lens(outfile)

    # Run through lenses
    if analytic == True:
        M_error = []
        for l in range(len(R_E)):
            xlims = [1e-3*R_E[l], 1e1*R_E[l]]  # [Mpc]
            rad = np.linspace(xlims[0], xlims[1], 1000)  # [Mpc]
            rho_crit = (3*H0**2)/(8*np.pi*G)*(km_to_Mpc/kg_to_Msun)  # [Msun/Mpc^3]
            # Chr. Arnold et al. 2016 (arxiv:1604:06095)
            delta_crit = 7.213*2*(Vmax[l]/(H0*Rvmax[l]))**2
            rho_NFW = (rho_crit*delta_crit)/((rad/Rsc[l])*(1 + rad/Rsc[l])**2)
            [M_est, error] = quad(halo_mass, xlims[0], xlims[1],
                                  args=(delta_crit, rho_crit, Rsc[l]))
            M_error.append(M_est/Mlensmax[l])
            if plot:
                plt.loglog()
                plt.xlim(xlims)
                plt.plot(rad, rho_NFW, c='b')
                plt.plot([R_E[l], R_E[l]], [np.min(rho_NFW), np.max(rho_NFW)], c='k')
                plt.xlabel(r'$R \quad [kpc]$')
                plt.ylabel(r'$\rho \quad [M_{\odot}/Mpc^{3}]$')
                plt.savefig('NFW_profile_'+str(l)+'.png', bbox_inches='tight')
                plt.clf()
    if profile_fitting:
        for l in range(len(R_E)):
            xlims = [1e-1, Rvir[l]]  # [Mpc]
            snapnum = find_nearest(z_sim, redshiftmax[l])
            snap = snapfile % (snapnum, snapnum)
            dmpos = readsnap.read_block(snap, 'POS ', parttype=0)
            dmmass = readsnap.read_block(snap, 'MASS', parttype=0)
            centre = subposmax[l]
            print('Start fitting')
            (sim_r, sim_measure, dum, dum, ndum, ndum, dum) = c.densprof(0,
                                                                         xlims[0],
                                                                         xlims[1],
                                                                         snapfile=snap,
                                                                         xc=centre[0],
                                                                         yc=centre[1],
                                                                         zc=centre[2])
            if plot:
                plt.loglog()
                plt.xlim(xlims)
                plt.plot(sim_r, sim_measure, c='b')
                plt.xlabel('R [kpc]')
                plt.ylabel(r'$\rho$ [M_{\odot}/Mpc^{3}]')
                plt.savefig('NFW_profile'+str(l)+'.png', bbox_inches='tight')
                plt.clf()
    if count_particles_in_Rth:
        # Loop through Lenses 
        print('number of lenses', len(R_E))
        #data = open('./shell_script/RE_eqv_area'+sim_name[sim]+'_1.txt', 'r')
        #lines = data.readlines()
        #einstein_angle = np.zeros(len(lines))
        #for k in range(len(lines)):
        #    einstein_angle[k] = float(lines[k].split()[0])
        #print(einstein_angle)
        for l in range(len(R_E))[:15]:
            R_E[l] = 5
            #einstein_angle = np.asarray(einstein_angle)
            #print((einstein_angle*u.arcsec).to_value('rad'))
            #R_E_1 = ((einstein_angle*u.arcsec).to_value('rad') * \
            #        WMAP9.angular_diameter_distance(Lensz[:21])).to_value('Mpc')
            #print(R_E_1)
            #plt.hist(R_E, 50, normed=1, facecolor='green', alpha=0.75, label='PointMass')
            #plt.hist(R_E_1, 50, normed=1, facecolor='red', alpha=0.75, label='Area')
            #plt.legend(loc=0)
            #plt.savefig('Test_'+sim_name[sim]+'.png', bbox_inches='tight')
            #plt.clf()
            #print('plotted')
            snapnum = find_nearest(z_sim, redshiftmax[l])
            snap = snapfile % (snapnum, snapnum)
            centre = subposmax[l]
            gpos = readsnap.read_block(snap, 'POS ', parttype=0)*1e-3   #[kpc] Gas
            dmpos = readsnap.read_block(snap, 'POS ', parttype=1)*1e-3  #[kpc] DM
            spos = readsnap.read_block(snap, 'POS ', parttype=4)*1e-3   #[kpc] Stars
            bhpos = readsnap.read_block(snap, 'POS ', parttype=5)*1e-3  #[kpc] BH
            
            #print('%.3f < X [kpc/h] < %.3f' % (np.min(dmpos[:,0]),np.max(dmpos[:,0])))
            #print('%.3f < Y [kpc/h] < %.3f' % (np.min(dmpos[:,1]),np.max(dmpos[:,1])))
            #print('%.3f < Z [kpc/h] < %.3f\n' % (np.min(dmpos[:,2]),np.max(dmpos[:,2])))
            
            gnum, grmin[l] = check_in_sphere(centre, gpos, R_E[l])
            dmnum, dmrmin[l] = check_in_sphere(centre, dmpos, R_E[l])
            snum, srmin[l] = check_in_sphere(centre, spos, R_E[l])
            bhnum, bhrmin[l] = check_in_sphere(centre, bhpos, R_E[l])
            totindx = gnum + dmnum + snum + bhnum
            print('Gas', gnum, 'Stars', snum, 'DM', dmnum, 'tot', totindx)
        plt.ylim(0, 0.1)
        plt.plot([0, 1], [0, 1], c='orange')
        plt.scatter(grmin, R_E, c='r', label='Gas')
        plt.scatter(dmrmin, R_E, c='g', label='Dark Matter')
        plt.scatter(srmin, R_E, c='b', label='Stars')
        plt.scatter(bhrmin, R_E, c='k', label='Black Holes')
        plt.legend(loc=0)
        plt.ylabel(r'$D_{nearest} \quad [kpc/h]$')
        plt.xlabel(r'$R_{E} \quad [kpc/h]$')
        plt.savefig('Rmin_'+sim_name[sim]+'.png', bbox_inches='tight')
        plt.clf()

    #plt.loglog()
    #plt.scatter(R_E, M_error)
    #plt.xlabel(r'$R_{E} \quad [kpc]$')
    #plt.ylabel(r'$M_{NFW}/M_{glafic}$')
    #plt.savefig('Merror_'+sim_name[sim]+'.png', bbox_inches='tight')
    #plt.clf()
