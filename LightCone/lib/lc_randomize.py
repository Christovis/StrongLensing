import numpy as np
import random as rnd

# Fixing random state for reproducibility
rnd.seed(1872)  #1872, 2944, 5912, 7638

def translation_s(position, boxlength):
    """
    Translational randomization
    
    Input:

    Output:

    """
    # Y-AXES
    align = np.abs(np.mean(position[:, 1]))
    length = np.max(position[:, 1]) - align
    shift = rnd.random()*boxlength
    position[:, 1] += shift
    indx = np.where(position[:, 1] > length)
    position[indx, 1] -= boxlength  # align with observer again
    # Z-AXES
    align = np.abs(np.mean(position[:, 2]))
    length = np.max(position[:, 2]) - align
    shift = rnd.random()*boxlength
    position[:, 2] += shift
    indx = np.where(position[:, 2] > length)
    position[indx, 2] -= boxlength  # align with observer again
    return position


def inversion_s(position, boxlength):
    """
    Translational randomization
    Input:
    Output:
    """
    #print('box range1', np.min(position[:, 2]), np.max(position[:, 2]))
    if rnd.random() > 0.5:
        position[:, 1] *= -1
    if rnd.random() > 0.5:
        position[:, 2] *= -1
    #print('box range2', np.min(position[:, 2]), np.max(position[:, 2]))
    return position


def rotation_s(position, boxlength, cosmo_dist):
    """
    Rotational randomization.
    
    Input:

    Output:
    """
    # centre on zero
    centre = [boxlength/2-cosmo_dist, boxlength/2, boxlength/2]
    position[:, 0] += centre[0]
    position[:, 1] += centre[1]
    position[:, 2] += centre[2]
    tracking = np.array([0, 1, 2])  # to follow original axes
    if rnd.random() > 0.5:
        # change x & y axes = rotate around z
        x_indx = np.where(tracking == 0)[0]
        y_indx = np.where(tracking == 1)[0]
        position[:, [x_indx, y_indx]] = position[:, [y_indx, x_indx]]
        tracking[x_indx] = 1
        tracking[y_indx] = 0
    if rnd.random() > 0.5:
        # change x & z axes = rotate around y
        x_indx = np.where(tracking == 0)[0]
        z_indx = np.where(tracking == 2)[0]
        position[:, [x_indx, z_indx]] = position[:, [z_indx, x_indx]]
        tracking[x_indx] = 2
        tracking[z_indx] = 0
    if rnd.random() > 0.5:
        # change y & z axes = rotate around x
        y_indx = np.where(tracking == 1)[0]
        z_indx = np.where(tracking == 2)[0]
        position[:, [y_indx, z_indx]] = position[:, [z_indx, y_indx]]
        tracking[y_indx] = 2
        tracking[z_indx] = 1
    position[:, 0] -= centre[0]
    position[:, 1] -= centre[1]
    position[:, 2] -= centre[2]
    return position

#if __name__ == "__main__":
#    import doctest
