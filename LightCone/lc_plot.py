import numpy as np
import readsubf
import readsnap
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def angle(v1, v2):
	""" 
	Returns the angle in radians between vectors 'v1' and 'v2'
	"""
	angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
	return angle


def check_in_sphere(centre, pos, Rth):
	r = np.sqrt((pos[:, 0])**2 +
				(pos[:, 1])**2 +
				(pos[:, 2])**2)
	indx = np.where(r < Rth)
	return indx


def rotate_axes(pos, av, bv, cv, x, y, z):
	"""
	Rotate coordinate axes
	"""
	R0 = np.outer(av, x)
	R1 = np.outer(bv, y)
	R2 = np.outer(cv, z)
	R = np.asarray(R0 + R1 + R2)
	pos = [np.dot(R, vec) for vec in pos]
	return np.asarray(pos)


def check_in_ellipse(pos, a, b, c):
	"""
	centered on origin
	a = semi-major axis
	b = semi-intermediate axis
	c = sami-minor axis
	"""
	r_xy = (pos[:, 0]**2/a**2) + (pos[:, 1]**2/b**2)
	r_xz = (pos[:, 0]**2/a**2) + (pos[:, 2]**2/c**2)
	r_yz = (pos[:, 1]**2/b**2) + (pos[:, 2]**2/c**2)
	indx = np.where((r_xy <= 1.) & (r_xz <= 1.) & (r_yz <= 1.))
	return indx


def principal_axes(pos, s, q, centre):
	"""
	Reduced Moment of Inertia Tensor
	Ellipsoid described by dimensionless parameters:
	find shape in 10 iterations
	minor-to-major axis ratio			s=c/a
	intermediate-to-major axis ratio	q=b/a
	minor-to-intermediate axis ratio	p=c/b
	with a,b,c the principal axes

	Input:
	av, bv, cv, principal axes unit vectors
	"""
	x = pos[:, 0]
	y = pos[:, 1]
	z = pos[:, 2]
	d2 = x*x + y*y/q**2 + z*z/s**2
	Ixx = np.sum(x*x/d2)
	Iyy = np.sum(y*y/d2)
	Izz = np.sum(z*z/d2)
	Ixy = np.sum(x*y/d2)
	Ixz = np.sum(x*z/d2)
	Iyz = np.sum(y*z/d2)
	I = np.array([[Ixx, Ixy, Ixz],
				  [Ixy, Iyy, Iyz],
				  [Ixz, Iyz, Izz]])
	evalue, evector = np.linalg.eigh(I)
	return evalue, evector


def ellipticity(subpos, submass, subrad, parpos):
	"""
	Based on Allgood et al. 2006 (arXiv:astro-ph/0508497)
	Iteration stops if s&q have changed less than 1e-3 (arXiv:0705.2037),
	or if enclosed regions contains less than 200 particles (arXive:1402.0903.pdf).

	Input:
		subpos: positions of subhalo centres
		submass: mass of subhalos in solar masses
		subrad: radius of subhalo e.g. R_vir, R_vmax, R_halfmass, ...

	Output:
		sub_e: subhalo ellipticity along eigenvectors of reduced momentum of
		 	   inertia tensor
		sub_av, sub_bv, sub_cv: normalized eigenvectors of reduced momentum of
								inertia tensor
	"""
	sub_e = np.zeros((len(submass), 3))
	sub_av = np.zeros((len(submass), 3))
	sub_bv = np.zeros((len(submass), 3))
	sub_cv = np.zeros((len(submass), 3))
	subindx = np.where(submass > 10**11)
	subpos = subpos[subindx]
	subrad = subrad[subindx]
	submass = submass[subindx]

	# Iterate through Subhalos
	for i in range(0, len(submass)):
		centre = subpos[i]
		# Centre particles on coord. origin
		par_in_sub = np.asarray(parpos - centre)
		radius = 9*subrad[i]
		indx = check_in_sphere(centre, par_in_sub, radius)
		par_in_sub = par_in_sub[indx]
		# Axis ratios
		[s, q] = [1, 1]
		axes_m, axes_v = principal_axes(par_in_sub, s, q, centre)
		[c, b, a] = [np.sqrt(axes_m[0]), np.sqrt(axes_m[1]), np.sqrt(axes_m[2])]
		axes_ell = [radius, radius, radius]
		# Normalized principal axis
		cv = axes_v[0]/np.linalg.norm(axes_v[0])
		bv = axes_v[1]/np.linalg.norm(axes_v[1])
		av = axes_v[2]/np.linalg.norm(axes_v[2])
		# Rotate Coord. to new coord. frame
		par_in_sub = rotate_axes(par_in_sub, av, bv, cv,
								[1, 0, 0], [0, 1, 0], [0, 0, 1])
		while ((s-c/a >= 1e-3) and (q-b/a >= 1e-3)) and (len(indx[0]) >= 200):
			[s, q] = [c/a, b/a]
			axes_ell[0] *= 1
			axes_ell[1] = 2*subhmrad[i]*q
			axes_ell[2] = 2*subhmrad[i]*s
			indx = check_in_ellipse(par_in_sub, axes_ell[0],
									axes_ell[1], axes_ell[2])
			par_in_sub = par_in_sub[indx]
			axes_m, axes_v = principal_axes(par_in_sub, s, q, centre)
			[c, b, a] = [np.sqrt(axes_m[0]), np.sqrt(axes_m[1]), np.sqrt(axes_m[2])]
		
		if len(indx[0]) >= 200:
			[s, q, p] = [c/a, b/a, c/b]
			sub_e[subindx[i], 0] = 1 - s  # (a - b)/(a + b)
			sub_e[subindx[i], 1] = 1 - q  # (a - c)/(a + b)
			sub_e[subindx[i], 2] = 1 - p  # (b - c)/(b + c)
			sub_va[subindx[i], 0] = av
			sub_vb[subindx[i], 1] = bv
			sub_vc[subindx[i], 2] = cv

###############################################################################
# Load Simulation Header
simdir = '/cosma6/data/dp004/dc-arno1/SZ_project/non_radiative_hydro/L62_N512_GR_pure_rerun'
snapdir = simdir+'/snapdir_%03d/snap_%03d.0'
header = readsnap.snapshot_header(snapdir % (45, 45))

subhalos = readsubf.subfind_catalog(simdir, 45, masstab=True)
subpos = subhalos.sub_pos
submass = subhalos.sub_mass*1e10/header.hubble
subhmrad = subhalos.sub_halfmassrad
subvmaxrad = subhalos.sub_vmaxrad
snapdir = simdir+'/snapdir_%03d/snap_%03d'
parpos = readsnap.read_block(snapdir % (45, 45), 'POS ', parttype=0)

sub_e, sub_va, sub_vb, sub_vc = ellipticity(subpos, submass, subvmaxrad, parpos)


# [a, b, c] = [3, 1, 0.5]
# theta = [30, 30, 30]
indx = check_in_sphere(subpos[0], parpos, a)
parpos = parpos[indx]
plt.scatter(parpos[:, 0], parpos[:, 1], marker='.', c='k')
plt.savefig('sphere.png', bbox_inches='tight')
plt.clf()

indx = check_in_ellipse(subpos[0], parpos, a, b, c, theta)
parpos = parpos[indx]
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.scatter(parpos[:, 1], parpos[:, 2], marker='.', c='k')
plt.savefig('ellipse.png', bbox_inches='tight')
plt.clf()


# subpos = rotation_s(subpos)
## PLOT ##
plt.figure(1)
plt.scatter(subpos[:, 0], subpos[:, 1], marker='.', c='k')
plt.savefig('SimBox_Visual_xy.png', bbox_inches='tight')
plt.clf()

plt.figure(2)
plt.scatter(subpos[:, 1], subpos[:, 2], marker='.', c='k')
plt.savefig('SimBox_Visual_yz.png', bbox_inches='tight')
plt.clf()
#
plt.figure(3)
plt.scatter(subpos[:, 0], subpos[:, 2], marker='.', c='k')
plt.savefig('SimBox_Visual_xz.png', bbox_inches='tight')
plt.clf()
#PLOT
#plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(par_in_sub[::10, 0], par_in_sub[::10, 1], marker='.', c='k')
#plt.plot([0, av[0]], [0, av[1]], 'r')
#plt.plot([0, bv[0]], [0, bv[1]], 'r')
#plt.plot([0, cv[0]], [0, cv[1]], 'r')
#plt.savefig('ellipse_%02d_xy.png' % k, bbox_inches='tight')
#plt.clf()
#plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(par_in_sub[::10, 0], par_in_sub[::10, 1], marker='.', c='k')
#plt.plot([0, av[2]], [0, av[1]], 'r')
#plt.plot([0, bv[2]], [0, bv[1]], 'r')
#plt.plot([0, cv[2]], [0, cv[1]], 'r')
#plt.savefig('ellipse_%02d_zy.png' % k, bbox_inches='tight')
#plt.clf()
#plt.gca().set_aspect('equal', adjustable='box')
#plt.scatter(par_in_sub[:: 10, 0], par_in_sub[::10, 1], marker='.', c='k')
#plt.plot([0, av[2]], [0, av[0]], 'r')
#plt.plot([0, bv[2]], [0, bv[0]], 'r')
#plt.plot([0, cv[2]], [0, cv[0]], 'r')
#plt.savefig('ellipse_%02d_zx.png' % k, bbox_inches='tight')
#plt.clf()
