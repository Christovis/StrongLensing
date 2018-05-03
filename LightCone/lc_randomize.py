import numpy as np
import random as rnd

def translation_s(position, boxlength):
#	rand = rnd.randint(0, 1)  # random number 0, 1
#	if rand == 0:  # move along z-axes
#	# Y-AXES
	align = np.abs(np.mean(position[:, 1]))
	length = np.max(position[:, 1]) - align
	shift = rnd.random()*boxlength
	position[:, 1] += shift
	indx = np.where(position[:, 1] > length)
	position[indx, 1] -= boxlength  # align with observer again
#	# Z-AXES
	align = np.abs(np.mean(position[:, 2]))
	length = np.max(position[:, 2]) - align
	shift = rnd.random()*boxlength
	position[:, 2] += shift
	indx = np.where(position[:, 2] > length)
	position[indx, 2] -= boxlength  # align with observer again
	return position


def rotation_s(position):
	# centre on zero
	centre = [np.mean(position[:, 0]),
			np.mean(position[:, 1]),
			np.mean(position[:, 2])]
	position[:, 0] -= centre[0]
	position[:, 1] -= centre[1]
	position[:, 2] -= centre[2]
	# only 4 angles due to performance
	phi = rnd.choice([0, np.pi/2, np.pi, 3*np.pi/2])
	psi = rnd.choice([0, np.pi/2, np.pi, 3*np.pi/2])
	tau = rnd.choice([0, np.pi/2, np.pi, 3*np.pi/2])
	rot_matrix = np.array([
		[np.cos(tau)*np.cos(phi),
		-np.cos(psi)*np.sin(phi)+np.sin(psi)*np.sin(tau)*np.cos(phi),
		np.sin(psi)*np.sin(phi)+np.cos(psi)*np.sin(tau)*np.cos(phi)],
		[np.cos(tau)*np.sin(phi),
		np.cos(psi)*np.cos(phi)+np.sin(psi)*np.sin(tau)*np.sin(phi),
		-np.sin(psi)*np.cos(phi)+np.cos(psi)*np.sin(tau)*np.sin(phi)],
		[-np.sin(tau),
		np.sin(psi)*np.cos(tau),
		np.cos(psi)*np.cos(tau)]])
	position = np.dot(rot_matrix, position.T).T
	# centre on original position
	position[:, 0] += centre[0]
	position[:, 1] += centre[1]
	position[:, 2] += centre[2]
	return position

if __name__ == "__main__":
	import doctest
