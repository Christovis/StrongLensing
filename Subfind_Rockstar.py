import sys, os
import numpy as np
import readsubf
import readsnap
import read_hdf5
from matplotlib import pyplot as plt
import readlensing as rf


def blockprint():
    sys.stdout = open(os.devnull, 'w')


def enableprint():
    sys.stdout = sys.__stdout__

snapnum = 45

###############################################################################
# Subfind Results
lc_dir = '/cosma5/data/dp004/dc-beck3/LightCone/full_physics/'
hf_name = 'Subfind'
lc_file = lc_dir+hf_name+'/LC_SN_L62_N512_GR_kpc_rndseed1.h5'
LC = rf.LightCone_with_SN_lens(lc_file, hf_name)
print('length', len(LC['Rhalfmass']))

###############################################################################
# Subfind
hfdir = '/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/'
blockprint()
s = read_hdf5.snapshot(snapnum, hfdir)
s.group_catalog(["SubhaloVelDisp", "SubhaloHalfmassRad",
                 "SubhaloMass", "SubhaloPos", "GroupPos", "GroupMass"])
enableprint()
Subfind= {'Pos' : s.cat["SubhaloPos"]*s.header.hubble*1e-3,
          'Mass' : s.cat["SubhaloMass"],
          'Vrms' : s.cat["SubhaloVelDisp"]}

###############################################################################
# Rockstar
hfdir = ('/cosma5/data/dp004/dc-beck3/rockstar/full_physics/L62_N512_GR_kpc/' + \
         'halos_%d.dat' % snapnum)
data = open(hfdir, 'r')
data = data.readlines()
subpos = []
subMvir = []
subvrms = []
for k in range(len(data))[16:]:
    pos = [float(coord)*1e-3 for coord in data[k].split()[9:12]]
    subpos.append(pos)
    subMvir.append(float(data[k].split()[2]))
    subvrms.append(float(data[k].split()[4]))
subpos = np.array(subpos)  # Comoving Distance
subMvir = np.asarray(subMvir)
subvrms = np.asarray(subvrms)
Rockstar = {'Pos' : subpos,
            'Mass' : subMvir,
            'Vrms' : subvrms}

###############################################################################
# Compare Catalouges
print('Length of Catalouges:', len(Subfind['Mass']), len(Rockstar['Mass']),
                               len(s.cat['GroupPos']))
print(':: Mass Range :::::::::::::::')
print('Subfind', np.min(Subfind['Mass'])/1e10, np.max(Subfind['Mass'])/1e10)
print('Rockstar', np.min(Rockstar['Mass'])/1e10, np.max(Rockstar['Mass'])/1e10)
print(':: Vrms Range :::::::::::::::')
print('Subfind', np.min(Subfind['Vrms']), np.max(Subfind['Vrms']))
print('Rockstar', np.min(Rockstar['Vrms']), np.max(Rockstar['Vrms']))

plt.hist(np.log10(Rockstar['Mass']), 20, range=[7, 14], alpha=0.75, label='Rockstar')
plt.hist(np.log10(Subfind['Mass']), 20, range=[7, 14], alpha=0.75, label='Subfind')
plt.xlabel(r'$log(M_{\odot}/h)$')
plt.legend(loc=1)
plt.savefig('Mass_comp.png', bbox_inches='tight')
plt.clf()
plt.hist(Rockstar['Vrms'], 20, range=[0, 200], alpha=0.75, label='Rockstar')
plt.hist(Subfind['Vrms']+20, 20, range=[0, 200], alpha=0.75, label='Subfind')
plt.xlabel(r'$v_{rms}$')
plt.legend(loc=1)
plt.savefig('Vrms_comp.png', bbox_inches='tight')
plt.clf()

###############################################################################
# Compare Halo

#indx = np.argmin(Rockstar['Mass'])
tag = 10120
pos_tag = Rockstar['Pos'][tag, :]

dist_x = np.abs(np.asarray([pos_tag[0]-x for x in Subfind['Pos'][:, 0]]))
dist_y = np.abs(np.asarray([pos_tag[1]-x for x in Subfind['Pos'][:, 1]]))
dist_z = np.abs(np.asarray([pos_tag[2]-x for x in Subfind['Pos'][:, 2]]))
print('Indx of min(delta_x)', np.argmin(dist_x), np.min(dist_x))
print('Indx of min(delta_y)', np.argmin(dist_y), np.min(dist_y))
print('Indx of min(delta_z)', np.argmin(dist_z), np.min(dist_z))
dist = np.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
print(dist)
indx = np.argmin(dist)
print('indx', indx)
sf_mass = Subfind['Mass'][indx]
sf_vrms = Subfind['Vrms'][indx]

print(':: Mass diff.:::::::::::::::')
print((Rockstar['Mass'][tag] - sf_mass)/1e10)
print(':: Vrms diff :::::::::::::::')
print(Rockstar['Vrms'][tag] - sf_vrms)

