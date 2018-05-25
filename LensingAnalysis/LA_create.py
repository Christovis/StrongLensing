from __future__ import division
import os
import re
import sys
import glob
import logging
import scipy
from scipy import stats
import numpy as np
from astropy import units as u
from astropy.cosmology import LambdaCDM
import multiprocessing
from multiprocessing import Process
import h5py
from matplotlib import pyplot as plt, rc
sys.path.insert(0, '..')
import readsnap
import readlensing as rf
import la_tools as la


############################################################################



def histogram(unit, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir):
    # Run through simulations
    for sim in range(len(sim_dir)):
        #logging.info('Create lensing map for: %s', sim_name[sim])
        # File for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        # LensMaps filenames
        lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+sim_name[sim]+'/'
        lm_files = [name for name in glob.glob(lm_dir+'LM_L*')]

        # Units of Simulation
        scale = rf.simulation_units(sim_dir[sim])
        
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        # Sort Lenses according to Snapshot Number (snapnum)
        indx = np.argsort(LC['snapnum'])
        Halo_ID= LC['Halo_ID'][indx]
        snapnum = LC['snapnum'][indx]

        HaloUnit = []
        # Run through lenses
        for ll in range(0, len(Halo_ID)):
            lm_files_match = [e for e in lm_files if 'L%d'%(Halo_ID[ll]) in e]
            if not lm_files_match:
                continue

            # Load Lens properties
            indx = np.where(LC['Halo_ID'] == Halo_ID[ll])
            HaloUnit.append(LC[unit][indx][0])

        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        if unit == 'M200':
            plt.hist(np.log10(HaloUnit), 20, alpha=0.75, label=sim_label)
        elif unit == 'Halo_z':
            print(HaloUnit)
            plt.hist(HaloUnit, 10, alpha=0.75, label=sim_label)
    if unit == 'M200':
        plt.xlabel(r'$log(M_{\odot}/h)$')
        plt.legend(loc=1)
        plt.savefig('./images/Hstl_M200.png', bbox_inches='tight')
    elif unit == 'Halo_z':
        plt.xlabel(r'$z$')
        plt.legend(loc=1)
        plt.savefig('./images/Hstl_redshift.png', bbox_inches='tight')
    plt.clf()


def dyn_vs_lensing_mass(CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir):
    # protect the 'entry point' for Windows OS
    if __name__ == '__main__':
        # after importing numpy, reset the CPU affinity of the parent process so
        # that it will use all cores
        os.system("taskset -p 0xff %d" % os.getpid())

        # Run through simulations
        for sim in range(len(sim_dir)):
            #logging.info('Create lensing map for: %s', sim_name[sim])
            # File for lens & source properties
            lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
            # Simulation Snapshots
            snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
            
            # Units of Simulation
            scale = rf.simulation_units(sim_dir[sim])
            # scale = 1e-3
            
            # Cosmological Parameters
            snap_tot_num = 45
            header = readsnap.snapshot_header(snapfile % (snap_tot_num, snap_tot_num))
            cosmo = LambdaCDM(H0=header.hubble*100,
                              Om0=header.omega_m,
                              Ode0=header.omega_l)
            h = header.hubble
            a = 1/(1 + header.redshift)

            # Load LightCone Contents
            LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

            # Sort Lenses according to Snapshot Number (snapnum)
            indx = np.argsort(LC['snapnum'])
            Halo_ID= LC['Halo_ID'][indx]
            snapnum = LC['snapnum'][indx]

            # Devide Halos over CPUs
            lenses_per_cpu = la.devide_halos(len(Halo_ID), CPUs)
            # Prepatre Processes to be run in parallel
            jobs = []
            manager = multiprocessing.Manager()
            results_per_cpu = manager.dict()
            for cpu in range(CPUs)[:]:
                p = Process(target=la.dyn_vs_lensing_mass, name='Proc_%d'%cpu,
                              args=(cpu, lenses_per_cpu[cpu], LC, Halo_ID,
                                    snapnum, snapfile, h, scale,
                                    HQ_dir, sim, sim_phy, sim_name,
                                    xi0, cosmo, results_per_cpu))
                jobs.append(p)
                p.start()
            print('started all jobs')
            # Run Processes in parallel
            # Wait until every job is completed
            for p in jobs:
                p.join()
           
            # Save Date
            Halo_ID = []
            Src_ID = []
            Mdyn = []
            Mlens = []
            for cpu in range(CPUs):
                results = results_per_cpu.values()[cpu]
                for src in range(len(results)):
                    Halo_ID.append(results[src][0])
                    Src_ID.append(results[src][1])
                    Mdyn.append(results[src][2])
                    Mlens.append(results[src][3])

            la_dir = HQ_dir+'LensingAnalysis/'+sim_phy[sim]
            hf = h5py.File(la_dir+'DynLensMass_'+sim_name[sim]+'.h5', 'w')
            hf.create_dataset('Halo_ID', data=Halo_ID)
            hf.create_dataset('Src_ID', data=Src_ID)
            hf.create_dataset('Mdyn', data=Mdyn)
            hf.create_dataset('Mlens', data=Mlens)
            hf.close()


def M200_Rein(sim_dir, sim_phy, sim_name, lc_dir, HQ_dir):
    for sim in range(len(sim_dir)):
        # LightCone file for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'.h5'
        # LensingMap files
        lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
        
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        # LensMaps filenames
        lm_files = [name for name in glob.glob(lm_dir+'LM_L*')]
        
        SnapNM = LC['snapnum']
        A_E = np.zeros(len(lm_files))
        M200 = np.zeros(len(lm_files))
        SNdist = np.zeros(len(lm_files))
        first_lens = 0
        previous_SnapNM = SnapNM[first_lens]
        
        delta_t = []
        delta_mu = []
        for ll in range(len(lm_files)):
            # Load LensingMap Contents
            s = re.findall('[+-]?\d+', lm_files[ll])
            Halo_ID=s[-3]; Src_ID=s[-2]
            indx = np.where(LC['Halo_ID'][:] == int(Halo_ID))[0]
            
            LM = h5py.File(lm_files[ll])
            #centre = LC['HaloPosBox'][ll]
            #zl = LC['Halo_z'][ll]
            M200[ll] = LC['M200'][indx]  #Mvir[ll]
            if LC['M200'][indx] > 1e-50:
                M200[ll] = LC['M200'][indx]  #Mvir[ll]
                A_E[ll] = LM['eqv_einstein_radius'].value  #[arcsec]
                
                t = LM['delta_t'].value
                indx_max = np.argmax(t)
                t -= t[indx_max]
                t = np.absolute(t[t != 0])
                for tt in t:
                    delta_t.append(tt)
                
                mu = LM['mu'].value
                mu -= mu[indx_max]
                mag = np.absolute(mu[mu != 0])
                for mm in mag:
                    delta_mu.append(mm)
                
        delta_t = np.asarray(delta_t)
        delta_mu = np.asarray(delta_mu)
        
        # Averageing the Scatter
        #plt.scatter(data[0]*1e14/h, data[1], c='k', label='Wiesner et al. 2012', s=20)
        binmean, binedg, binnum = stats.binned_statistic(np.log10(M200),
                                                         A_E,
                                                         statistic='median',
                                                         bins=10)
        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        plt.plot(binedg[:-1], binmean, label=sim_label)
        plt.scatter(np.log10(M200), A_E, s=5,
                    alpha=0.3, edgecolors='none')
    plt.xlabel(r'$M_{vir} \quad [M_\odot/h$]')
    plt.ylabel(r'$\theta_{E} \quad [arcsec]$')
    plt.legend(loc=2)
    plt.savefig('./images/M200_Rein.png', bbox_inches='tight')
    plt.clf()
