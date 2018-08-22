from __future__ import division
import os, re, sys
import scipy
from scipy import stats
import numpy as np
from astropy import units as u
from astropy.cosmology import LambdaCDM
import multiprocessing
from multiprocessing import Process
import h5py
import pickle
from matplotlib import pyplot as plt, rc
import la_tools as la
sys.path.insert(0, '/cosma5/data/dp004/dc-beck3/')
import read_hdf5
import readlensing as rf


############################################################################

def histogram_lens(unit, CPUs, sim_dir, sim_phy, sim_name, hfname, lc_dir, HQ_dir):
    # Run through simulations
    for sim in range(len(sim_dir)):
        # File for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'_rndseed.h5'
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        # LensMaps filenames
        lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+hfname+'/'+sim_name[sim]+'/'
        # Units of Simulation
        scale = rf.simulation_units(sim_dir[sim])
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        HaloUnit = []
        # Run through LensingMap output files
        for cpu in range(CPUs):
            lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
            # Load LensingMap Contents
            filed = open(lm_file, 'rb', encoding='utf8')
            LM = pickle.load(open(lm_file, 'rb'))

            indx = np.nonzero(np.in1d(LC['Halo_ID'], LM['Halo_ID']))[0]
            HaloUnit.append(LC[unit][indx])

        HaloUnit = np.concatenate(HaloUnit, axis=0)
        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        if unit == 'M200':
            plt.hist(np.log10(HaloUnit), 20, alpha=0.75,
                     label=sim_label+': '+str(len(HaloUnit)))
        elif unit == 'Halo_z':
            plt.hist(HaloUnit, 20, alpha=0.75,
                     label=sim_label+': '+str(len(HaloUnit)))
    if unit == 'M200':
        plt.xlabel(r'$log(M_{\odot}/h)$')
        plt.legend(loc=1)
        plt.savefig('./images/Hstl_M200.png', bbox_inches='tight')
    elif unit == 'Halo_z':
        plt.xlabel(r'$z$')
        plt.legend(loc=1)
        plt.savefig('./images/Hstl_redshift.png', bbox_inches='tight')
    plt.clf()


def histogram_src(unit, CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir):
    # Run through simulations
    for sim in range(len(sim_dir)):
        # File for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'_rndseed.h5'
        # Simulation Snapshots
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        # LensMaps filenames
        lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+sim_name[sim]+'/'
        # Units of Simulation
        scale = rf.simulation_units(sim_dir[sim])
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        SrcUnit = []
        # Run through LensingMap output files
        for cpu in range(CPUs):
            lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
            # Load LensingMap Contents
            filed = open(lm_file, 'rb')
            LM = pickle.load(filed)
            for ll in range(len(LM['Halo_ID'])):
                for ss in range(len(LM['Sources']['Src_ID'][ll])):
                    if unit == 'delta_t':
                        t = LM['Sources']['delta_t'][ll][ss]
                        indx_max = np.argmax(t)
                        t -= t[indx_max]
                        t = np.absolute(t[t != 0])
                        for ii in range(len(t)):
                            SrcUnit.append(t[ii])

        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        if unit == 'mu':
            plt.hist(SrcUnit, 20, alpha=0.75,
                     label=sim_label+': '+str(len(SrcUnit)))
        elif unit == 'delta_t':
            plt.hist(SrcUnit, 20, alpha=0.75,
                     label=sim_label+': '+str(len(SrcUnit)))
    if unit == 'mu':
        plt.xlabel(r'$log(M_{\odot}/h)$')
        plt.legend(loc=1)
        plt.savefig('./images/Hsts_mu.png', bbox_inches='tight')
    elif unit == 'delta_t':
        plt.xlabel(r'$\Delta t$')
        plt.legend(loc=1)
        plt.savefig('./images/Hsts_delta_t.png', bbox_inches='tight')
    plt.clf()


def dyn_vs_lensing_mass(CPUs, sim_dir, sim_phy, sim_name, hfname, lc_dir, HQ_dir):
    # protect the 'entry point' for Windows OS
    # if __name__ == '__main__':
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    os.system("taskset -p 0xff %d" % os.getpid())

    # Run through simulations
    for sim in range(len(sim_dir)):
        # File for lens & source properties
        lc_file = lc_dir[sim]+hfname+'/LC_SN_'+sim_name[sim]+'_rndseed1.h5'
        # File for lensing-maps
        lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+hfname+'/'+sim_name[sim]+'/'
        # Simulation Snapshots
        snapfile = sim_dir[sim]
        
        # Units of Simulation
        scale = rf.simulation_units(sim_dir[sim])
        
        # Cosmological Parameters
        s = read_hdf5.snapshot(45, snapfile)
        cosmo = LambdaCDM(H0=s.header.hubble*100,
                          Om0=s.header.omega_m,
                          Ode0=s.header.omega_l)
        h = s.header.hubble
        a = 1/(1 + s.header.redshift)

        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, hfname)

        # Sort Lenses according to Snapshot Number (snapnum)
        indx = np.argsort(LC['snapnum'])
        Halo_ID= LC['Halo_ID'][indx]

        # Prepatre Processes to be run in parallel
        jobs = []
        manager = multiprocessing.Manager()
        results_per_cpu = manager.dict()
        snapfile = sim_dir[sim]+'snapdir_%03d/snap_%03d'
        for cpu in range(CPUs):
            # Load LensingMaps Contents
            lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
            p = Process(target=la.dyn_vs_lensing_mass, name='Proc_%d'%cpu,
                          args=(cpu, LC, lm_file, snapfile, h, scale,
                                HQ_dir, sim, sim_phy, sim_name, hfname,
                                cosmo, results_per_cpu))
            p.start()
            jobs.append(p)
        # Run Processes in parallel
        # Wait until every job is completed
        for p in jobs:
            p.join()
       
        # Save Data
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

        la_dir = HQ_dir+'/LensingAnalysis/'+sim_phy[sim]
        print('save data', la_dir, sim_name[sim])
        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        hf = h5py.File(la_dir+'DLMass_pa_shmr_svrms'+sim_label+'.h5', 'w')
        hf.create_dataset('Halo_ID', data=Halo_ID)
        hf.create_dataset('Src_ID', data=Src_ID)
        hf.create_dataset('Mdyn', data=Mdyn)
        hf.create_dataset('Mlens', data=Mlens)
        hf.close()


def M200_Rein(CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir):
    for sim in range(len(sim_dir)):
        # LightCone file for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'_rndseed.h5'
        # LensingMap files
        #lm_dir = HQ_dir+'LensingMap/'+sim_phy[sim]+'/'+sim_name[sim]+'/'
        lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+sim_name[sim]+'/'
        
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        M200 = []
        Rein = []
        for cpu in range(CPUs):
            lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
            # Load LensingMap Contents
            filed = open(lm_file, 'rb')
            LM = pickle.load(filed)

            #indx = np.nonzero(np.in1d(LC['Halo_ID'], LM['Halo_ID']))[0]
            for ii in range(len(LM['Sources']['Rein'])):
                for jj in range(len(LM['Sources']['Rein'][ii])):
                    indx = np.where(LC['Halo_ID'] == LM['Halo_ID'][ii])[0]
                    M200.append(LC['M200'][indx][0])
                    Rein.append(LM['Sources']['Rein'][ii][jj])
        M200 = np.asarray(M200)
        Rein = np.asarray(Rein)

        #plt.scatter(data[0]*1e14/h, data[1], c='k', label='Wiesner et al. 2012', s=20)
        print('length of M200', len(M200))
        binmean, binedg, binnum = stats.binned_statistic(np.log10(M200),
                                                         Rein,
                                                         statistic='median',
                                                         bins=15)
        sim_label = la.define_sim_label(sim_name[sim], sim_dir[sim])
        plt.plot(binedg[:-1], binmean, label=sim_label)
        plt.scatter(np.log10(M200), Rein, s=5,
                    alpha=0.3, edgecolors='none')
    plt.ylim(0, 10)
    plt.xlabel(r'$M_{vir} \quad [M_\odot/h$]')
    plt.ylabel(r'$\theta_{E} \quad [arcsec]$')
    plt.legend(loc=2)
    plt.savefig('./images/M200_Rein.png', bbox_inches='tight')
    plt.clf()


def deltat_mu(CPUs, sim_dir, sim_phy, sim_name, hfname, lc_dir, HQ_dir):
    for sim in range(len(sim_dir)):
        # LightCone file for lens & source properties
        lc_file = lc_dir[sim]+'LC_SN_'+sim_name[sim]+'_rndseed.h5'
        # LensingMap files
        lm_dir = HQ_dir+'/LensingMap/'+sim_phy[sim]+hfname+'/'+sim_name[sim]+'/'
        
        # Load LightCone Contents
        LC = rf.LightCone_with_SN_lens(lc_file, 'dictionary')

        rank = []
        delta_t = []
        mu = []
        for cpu in range(CPUs):
            lm_file = lm_dir+'LM_Proc_'+str(cpu)+'_0.pickle'
            # Load LensingMap Contents
            filed = open(lm_file, 'rb')
            LM = pickle.load(filed)

            #indx = np.nonzero(np.in1d(LC['Halo_ID'], LM['Halo_ID']))[0]
            for ll in range(len(LM['Halo_ID'])):
                for ss in range(len(LM['Sources']['Src_ID'][ll])):
                    t = LM['Sources']['delta_t'][ll][ss]
                    f = LM['Sources']['mu'][ll][ss]
                    indx_max = np.argmax(t)
                    t -= t[indx_max]
                    t = np.absolute(t[t != 0])
                    for ii in range(len(t)):
                        rank.append(ii)
                        delta_t.append(t[ii])
                   
                    print('flux ratio', f)
                    f_max = f[indx_max]
                    f = f[f != f[indx_max]]
                    for ii in range(len(f)):
                        #mu.append(-2.5*np.log10(f_max/f[ii]))
                        # logarithm base change
                        mu.append(np.log(abs(f_max/f[ii]))/np.log(2.512))
        print('mu', np.min(mu), np.max(mu)) 
        plt.scatter(mu, delta_t, s=5,
                    alpha=0.3, edgecolors='none')
    plt.xlabel(r'$\Delta m$')
    plt.ylabel(r'$\Delta t \quad [days]$')
    plt.legend(loc=2)
    plt.savefig('./images/deltat_mu.png', bbox_inches='tight')
    plt.clf()
                            

#            if LC['M200'][indx] > 1e-50:
#                M200[ll] = LC['M200'][indx]  #Mvir[ll]
#                A_E[ll] = LM['eqv_einstein_radius'].value  #[arcsec]
#                
#                t = LM['delta_t'].value
#                indx_max = np.argmax(t)
#                t -= t[indx_max]
#                t = np.absolute(t[t != 0])
#                for tt in t:
#                    delta_t.append(tt)
#                
#                mu = LM['mu'].value
#                mu -= mu[indx_max]
#                mag = np.absolute(mu[mu != 0])
#                for mm in mag:
#                    delta_mu.append(mm)
