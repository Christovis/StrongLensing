from __future__ import division
import sys
import h5py
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck15 as cosmo
from scipy.ndimage.filters import gaussian_filter
from sklearn.neighbors import KDTree
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('/cosma5/data/dp004/dc-beck3')
import readlensing as rf

def rotate_axes(pos, av, bv, cv, x, y, z):
    """
    Rotate coordinate axes
    """
    R0 = np.outer(av, x)
    R1 = np.outer(bv, y)
    R2 = np.outer(cv, z)
    R = np.asarray(R0 + R1 + R2)
    pos = [np.dot(R.T, vec) for vec in pos]
    return np.asarray(pos)

def adaptively_smoothed_maps(X, h, M, bins, smooth_fac=0.5):
    Ncells = bins
    hbins = int(np.log2(h.max()/h.min()))+2
    hbin_edges = 0.8*h.min()*2**np.arange(hbins)
    hbin_mids = np.sqrt(hbin_edges[1:]*hbin_edges[:-1])
    hmask = np.digitize(h,hbin_edges)-1
    sigmaS = np.zeros((len(hbin_mids),Ncells[0],Ncells[1]))
    for i in np.arange(len(hbin_mids)):
        maskpos = X[hmask==i]
        maskm = M[hmask==i]
        maskSigma, xedges, yedges = np.histogram2d(maskpos[:, 0],
                                                   maskpos[:, 1],
                                                   bins=Ncells,
                                                   #range=[[0.0, 5500],
                                                   #       [-30, 30]],
                                                   weights=maskm)
        pixelsmooth = smooth_fac*hbin_mids[i]/(xedges[1]-xedges[0])
        sigmaS[i] = gaussian_filter(maskSigma,pixelsmooth,truncate=3)
    return np.sum(sigmaS,axis=0), xedges, yedges

def projected_surface_density(X,M,bins=512,smooth=True,smooth_fac=None,neighbour_no=None):
    Ncells = bins
    
    # Shift particle coordinates to centre
    if smooth is True:
        if smooth_fac is None:
            print "Need to supply a value for smooth_fac when smooth=True"
            return
        if neighbour_no is None:
            print "Need to supply a value for neighbour_no when smooth=True"
            return
        # Find 'smoothing lengths'
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        dist, ids = kdt.query(X, k=neighbour_no, return_distance=True)
        h = np.max(dist,axis=1)
        mass_in_cells, xedges, yedges = adaptively_smoothed_maps(X, h, M,
                                                                 bins=Ncells,
                                                                 smooth_fac=smooth_fac)
    else:
        mass_in_cells, xedges, yedges = np.histogram2d(X[:,0], X[:,1],
                                                       bins=[Ncells, Ncells],
                                                       #range=[[0, 5500],
                                                       #       [-30, 30]],
                                                       weights=M)
    dx, dy = xedges[1]-xedges[0], yedges[1]-yedges[0]
    Sigma = mass_in_cells / (dx*dy)
    xs = 0.5*(xedges[1:]+xedges[:-1])
    ys = 0.5*(yedges[1:]+yedges[:-1])
    return Sigma, xs, ys

###############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
dd, sim_phy, sim_name, dd, dd, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

###############################################################################
# Load Simulated Lightcone with Subhalos and Supernovae
lc_file = lc_dir[0]+'LC_'+sim_name[0]+'.h5'
# Load LightCone Contents
LC = h5py.File(lc_file, 'r')

lc_sn_file = lc_dir[0]+'LC_SN_'+sim_name[0]+'.h5'
# Load LightCone Contents
LC_SN = h5py.File(lc_sn_file, 'r')

###############################################################################
# Select Subhalos and Supernovae
HaloPosLC = LC['position_lc'][::10]
M200 = LC['Mvir'][::10]*0.6774/1e10
print(len(M200))
#sys.exit("Error message")

indx = np.where((LC_SN['Halo_z'][:] < 0.4) & (0.3 < LC_SN['Halo_z'][:]))[0]
Halo_indx = indx[-1]
Halo_ID = LC_SN['Halo_ID'][:][Halo_indx]
indx = np.where(LC_SN['Src_ID'][:] == Halo_ID)[0]
SrcPosLC = LC_SN['SrcPosSky'][:][indx]
Halo_unit1 = np.asarray(LC_SN['HaloPosLC'][:][Halo_indx, :]/ \
                        np.linalg.norm(LC_SN['HaloPosLC'][:][Halo_indx, :]))
Halo_unit2 = [-Halo_unit1[1], Halo_unit1[0], 0]
Halo_unit3 = np.cross(Halo_unit1, Halo_unit2)
SrcPosLC1 = rotate_axes(SrcPosLC, [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     Halo_unit1, Halo_unit2, Halo_unit3)


indx = np.where((LC_SN['Halo_z'][:] < 0.2) & (0.1 < LC_SN['Halo_z'][:]))[0]
Halo_indx = indx[-1]
Halo_ID = LC_SN['Halo_ID'][:][Halo_indx]
indx = np.where(LC_SN['Src_ID'][:] == Halo_ID)[0]
SrcPosLC = LC_SN['SrcPosSky'][:][indx]
Halo_unit1 = np.asarray(LC_SN['HaloPosLC'][:][Halo_indx, :]/ \
                        np.linalg.norm(LC_SN['HaloPosLC'][:][Halo_indx, :]))
Halo_unit2 = [-Halo_unit1[1], Halo_unit1[0], 0]
Halo_unit3 = np.cross(Halo_unit1, Halo_unit2)
SrcPosLC2 = rotate_axes(SrcPosLC, [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     Halo_unit1, Halo_unit2, Halo_unit3)

indx = np.where((LC_SN['Halo_z'][:] < 0.8) & (0.7 < LC_SN['Halo_z'][:]))[0]
Halo_indx = indx[-1]
Halo_ID = LC_SN['Halo_ID'][:][Halo_indx]
indx = np.where(LC_SN['Src_ID'][:] == Halo_ID)[0]
SrcPosLC = LC_SN['SrcPosSky'][:][indx]
Halo_unit1 = np.asarray(LC_SN['HaloPosLC'][:][Halo_indx, :]/ \
                        np.linalg.norm(LC_SN['HaloPosLC'][:][Halo_indx, :]))
Halo_unit2 = [-Halo_unit1[1], Halo_unit1[0], 0]
Halo_unit3 = np.cross(Halo_unit1, Halo_unit2)
SrcPosLC3 = rotate_axes(SrcPosLC, [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     Halo_unit1, Halo_unit2, Halo_unit3)
###############################################################################
print(' Smooth distribution')
Ncells = [int(round(11/15*1000)), int(round(4/15*1000))]
smooth_fac = 0.5

Sigma, x, y = projected_surface_density(HaloPosLC, M200, Ncells, smooth=True,
                                        smooth_fac=0.8, neighbour_no=4)

print(np.max(Sigma), np.mean(Sigma))
###############################################################################
# PLOT 
xticks_Mpc = np.array([1, 2, 3, 4, 5])*1e3*u.Mpc
z = np.array([0.0, 0.5, 1, 1.5, 2])
xticks_z = cosmo.comoving_distance(z).to_value('Mpc')

fig = plt.figure(figsize=(15,4))
ax1 = fig.add_subplot(111)
ax1.imshow((Sigma).T, #extent=[x.min(), x.max(), y.min(), y.max()],
           cmap='jet', origin='lower')
#ax1.scatter(HaloPosLC[:, 0], HaloPosLC[:, 1], marker='.', s=1, c='w')
#ax1.scatter(SrcPosLC1[:, 0], SrcPosLC1[:, 1], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC2[:, 0], SrcPosLC2[:, 1], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC3[:, 0], SrcPosLC3[:, 1], marker='.', s=1, c='r')
#ax1.set_facecolor('k')
#ax1.set_xlim(0, 5500)
#ax1.set_ylim(-30, 30)
#ax2 = ax1.twiny()
#ax2.set_xticks(xticks_z)
#ax2.set_xticklabels(['{:g}'.format(redshift) for redshift in z]);
ax1.set_xlabel('Comoving Distance [Mpc]')
ax1.set_ylabel('Comoving Distance [Mpc]')
#ax2.set_xlabel('Redshift')
fig.savefig('Lightcone_xy.png', bbox_inches='tight')
fig.clf()

#fig = plt.figure(figsize=(15,4))
#ax1 = fig.add_subplot(111)
#ax1.scatter(HaloPosLC[:, 0], HaloPosLC[:, 2], marker='.', s=1, c='w')
#ax1.scatter(SrcPosLC1[:, 0], SrcPosLC1[:, 2], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC2[:, 0], SrcPosLC2[:, 2], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC3[:, 0], SrcPosLC3[:, 2], marker='.', s=1, c='r')
#ax1.set_facecolor('k')
#ax1.set_xlim(0, 5500)
#ax1.set_ylim(-40, 40)
#ax2 = ax1.twiny()
#ax2.set_xticks(xticks_z)
#ax2.set_xticklabels(['{:g}'.format(redshift) for redshift in z]);
#ax1.set_xlabel('Comoving Distance [Mpc]')
#ax1.set_ylabel('Comoving Distance [Mpc]')
#ax2.set_xlabel('Redshift')
#fig.savefig('Lightcone_xz.png', bbox_inches='tight')
#fig.clf()
#
#fig = plt.figure(figsize=(4,4))
#ax1 = fig.add_subplot(111)
#ax1.scatter(HaloPosLC[:, 1], HaloPosLC[:, 2], marker='.', s=1, c='w')
#ax1.scatter(SrcPosLC1[:, 1], SrcPosLC1[:, 2], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC2[:, 1], SrcPosLC2[:, 2], marker='.', s=1, c='r')
#ax1.scatter(SrcPosLC3[:, 1], SrcPosLC3[:, 2], marker='.', s=1, c='r')
#ax1.set_facecolor('k')
#ax1.set_xlim(-30, 30)
#ax1.set_ylim(-30, 30)
#ax1.set_xlabel('Comoving Distance [Mpc]')
#ax1.set_ylabel('Comoving Distance [Mpc]')
#fig.savefig('Lightcone_yz.png', bbox_inches='tight')
#fig.clf()
