import os
import sys
#sys.path.insert(0, '../')
import numpy as np
import h5py

class readfile:
    """
    PURPOSE:
        Read the different files produced in the pipeline of strong lensed SNIa
    USE:
        import SLSNreadfile as rf
    INPUT:
        filename
    """
    def init(self):
        pass


def Simulation_Specs(filename):
    data = open(filename, 'r')
    Settings = data.readlines()
    sim_dir = []
    sim_phy = []
    sim_name = []
    sim_col = []
    hf_name = []
    hf_dir = []
    lc_dir = []
    glafic_dir = []
    for k in range(len(Settings)):
        if 'Python' in Settings[k].split():
            HQ_dir = Settings[k+1].split()[0]
        if 'Simulation' in Settings[k].split():
            [simphy, simname] = Settings[k+1].split()
            sim_phy.append(simphy)
            sim_name.append(simname)
            [simdir, simcol, simunit] = Settings[k+2].split()
            sim_dir.append(simdir)
            sim_col.append(simcol)
            [hfdir, hf_name] = Settings[k+3].split()
            hf_dir.append(hfdir)
            lcdir = Settings[k+4].split()[0]
            lc_dir.append(lcdir)
            glaficdir = Settings[k+5].split()[0]
            glafic_dir.append(glaficdir)
    return sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, glafic_dir, HQ_dir


def LightCone_without_SN(filename, hfname):
    print(filename)
    data = h5py.File(filename, 'r')
    if hfname == 'Subfind':
        # Get the data
        snapnum = data['snapnum'].value  # []
        Halo_ID = data['Halo_ID'].value                # []
        Halo_z = data['Halo_z'].value       # []
        Mvir = data['Mvir'].value            # [Msun/h]
        HaloPosBox = data['HaloPosBox'].value # [x, y, z] [Mpc] com. distance
        HaloPosLC = data['HaloPosLC'].value   # [x, y, z] [Mpc] com. distance
        HaloVel = data['HaloVel'].value   # [x, y, z] [Mpc] com. distance
        Vmax = data['Vmax'].value  # [km/s] com. distance
        Vrms = data['Vrms'].value  # [km/s] com. distance
        Rhalfmass = data['Rhalfmass'].value      # [kpc/h] com. distance
        Rvmax = data['Rvmax'].value    # [kpc/h] com. distance
        LCHalos = {'snapnum' : snapnum,
                  'Halo_ID' : Halo_ID,
                  'Halo_z' : Halo_z,
                  'M200' : Mvir,
                  'Rhalfmass' : Rhalfmass,
                  'Rvmax' : Rvmax,
                  'Vmax' : Vmax,
                  'HaloPosBox' : HaloPosBox,
                  'HaloPosLC' : HaloPosLC,
                  'HaloVel' : HaloVel,
                  'Vrms' : Vrms}
        return LCHalos
    elif hfname == 'Rockstar':
        # Get the data
        snapnum = data['snapnum'].value  # []
        Halo_ID = data['Halo_ID'].value                # []
        Halo_z = data['Halo_z'].value       # []
        Mvir = data['Mvir'].value            # [Msun/h]
        HaloPosBox = data['HaloPosBox'].value # [x, y, z] [Mpc] com. distance
        HaloPosLC = data['HaloPosLC'].value   # [x, y, z] [Mpc] com. distance
        HaloVel = data['HaloVel'].value   # [x, y, z] [Mpc] com. distance
        Vmax = data['Vmax'].value  # [km/s] com. distance
        Vrms = data['Vrms'].value  # [km/s] com. distance
        Rvir = data['Rvir'].value   # [kpc/h] com. distance
        Rsca = data['Rs'].value      # [kpc/h] com. distance
        Rvmax = data['Rvmax'].value    # [kpc/h] com. distance
        Ellip = data['ellipticity'].value  # [x, y, z] [Mpc] com. distance
        Pa = data['position_angle'].value    # [radiants]return
        LCHalos = {'snapnum' : snapnum,
              'Halo_ID' : Halo_ID,
              'Halo_z' : Halo_z,
              'M200' : Mvir,
              'Rvir' : Rvir,
              'Rsca' : Rsca,
              'Rvmax' : Rvmax,
              'Vmax' : Vmax,
              'HaloPosBox' : HaloPosBox,
              'HaloPosLC' : HaloPosLC,
              'HaloVel' : HaloVel,
              'Vrms' : Vrms,
              'Ellip' : Ellip,
              'Pa' : Pa}
        return LCHalos


def LightCone_with_SN_lens(filename, hfname):
    if hfname == 'Subfind':
        data = h5py.File(filename, 'r')
        Halo_ID = data['Halo_ID'].value
        Halo_Subfind_ID = data['Halo_Subfind_ID'].value
        snapnum = data['snapnum'].value
        Halo_z = data['Halo_z'].value
        M200 = data['M200'].value  #[Msun/h]
        Rhalfmass = data['Rhalfmass'].value*1e-3  #[Mpc/h]
        Rvmax = data['Rvmax'].value*1e-3  #[Mpc/h]
        Vmax = data['Vmax'].value  #[km/s]
        HaloPosBox = data['HaloPosBox'].value  #[Mpc/h]
        HaloPosLC = data['HaloPosLC'].value  #[Mpc/h]
        HaloVel = data['HaloVel'].value  #[Mpc/h]
        VelDisp= data['VelDisp'].value  #[km/h]
        FOV = data['FOV'].value  #[arcsec]
        Src_ID = data['Src_ID'].value
        Src_z = data['Src_z'].value
        SrcPosSky = data['SrcPosSky'].value
        Einstein_angle = data['Einstein_angle'].value  #[arcsec]
        #Einstein_radius = data['Einstein_radius'].value  #[kpc]
        LC = {'Halo_ID' : Halo_ID,
              'Halo_Subfind_ID' : Halo_Subfind_ID,
              'snapnum' : snapnum,
              'Halo_z' : Halo_z,
              'M200' : M200,  #[Msun/h]
              'Rhalfmass' : Rhalfmass,
              'Rvmax' : Rvmax,
              'Vmax' : Vmax,  #[km/s]
              'HaloPosBox' : HaloPosBox,
              'HaloPosLC' : HaloPosLC,
              'HaloVel' : HaloVel,
              'VelDisp' : VelDisp,  #[km/s]
              'FOV' : FOV,  #[arcsec]
              'Src_ID' : Src_ID,
              'Src_z' : Src_z,
              'SrcPosSky' : SrcPosSky,
              'Einstein_angle' : Einstein_angle}
              #'Einstein_radius' : Einstein_radius}
        return LC
    elif hfname == 'Rockstar':
        data = h5py.File(filename, 'r')
        Halo_ID = data['Halo_ID'].value
        Halo_Rockstar_ID = data['Halo_Rockstar_ID'].value
        snapnum = data['snapnum'].value
        Halo_z = data['Halo_z'].value
        M200 = data['M200'].value  #[Msun/h]
        Rvir = data['Rvir'].value*1e-3  #[Mpc/h]
        Rsca = data['Rsca'].value*1e-3  #[Mpc/h]
        Rvmax = data['Rvmax'].value*1e-3  #[Mpc/h]
        Vmax = data['Vmax'].value  #[km/s]
        HaloPosBox = data['HaloPosBox'].value  #[Mpc/h]
        HaloPosLC = data['HaloPosLC'].value  #[Mpc/h]
        HaloVel = data['HaloVel'].value  #[Mpc/h]
        VelDisp= data['VelDisp'].value  #[km/h]
        Ellip = data['Ellip'].value
        Pa = data['Pa'].value
        FOV = data['FOV'].value  #[arcsec]
        Src_ID = data['Src_ID'].value
        Src_z = data['Src_z'].value
        SrcPosSky = data['SrcPosSky'].value
        Einstein_angle = data['Einstein_angle'].value  #[arcsec]
        #Einstein_radius = data['Einstein_radius'].value  #[kpc]
        LC = {'Halo_ID' : Halo_ID,
              'Halo_Rockstar_ID' : Halo_Rockstar_ID,
               'snapnum' : snapnum,
               'Halo_z' : Halo_z,
               'M200' : M200,  #[Msun/h]
               'Rvir' : Rvir,
               'Rsca' : Rsca,
               'Rvmax' : Rvmax,
               'Vmax' : Vmax,  #[km/s]
               'HaloPosBox' : HaloPosBox,
               'HaloPosLC' : HaloPosLC,
               'HaloVel' : HaloVel,
               'VelDisp' : VelDisp,  #[km/s]
               'Ellip' : Ellip,
               'Pa' : Pa,
               'FOV' : FOV,  #[arcsec]
               'Src_ID' : Src_ID,
               'Src_z' : Src_z,
               'SrcPosSky' : SrcPosSky,
               'Einstein_angle' : Einstein_angle}
               #'Einstein_radius' : Einstein_radius}
        return LC


def LightCone_with_SN_source(filename):
    data = open(filename, 'r')
    lines = data.readlines()
    source_id = np.zeros(len(lines))
    source_red = np.zeros(len(lines))
    source_pos = np.zeros((len(lines), 2))
    theta_E = np.zeros(len(lines))
    radius_E = np.zeros(len(lines))
    for k in range(len(lines))[1:]:
        [ids, z, x, y, theta, radius] = lines[k].split()
        source_id[k] = int(ids)
        source_red[k] = float(z)
        source_pos[k, :] = [float(x), float(y)]
        theta_E[k] = float(theta)
        radius_E[k] = float(radius)
    return source_id, source_red, source_pos, theta_E, radius_E


def Glafic_lens(filename, dataformat):
    data = open(filename, 'r')
    lines = data.readlines()
    iid = []
    AE = []
    ME = []
    for k in range(len(lines))[1:]:
        [ids, einstein_angle, mass_in_RE] = lines[k].split()
        iid.append(int(ids))
        AE.append(float(einstein_angle))
        ME.append(float(mass_in_RE))  # [Msun/h]
    iid = np.asarray(iid)
    AE = np.asarray(AE)
    ME = np.asarray(ME)
    if dataformat == 'dictionary':
        Glafic = {'ID' : iid,
                  'AE' : AE,
                  'ME' : ME}
        return Glafic
    else:
        return iid, AE, ME


def Glafic_source(filename, dataformat):
    """
    Read the files containing Glafic (M. Oguri) results
    Input:
        filename: path to file
        dataformat: single arrays or dictionary
    """
    data = open(filename, 'r')
    lines = data.readlines()
    iid = []
    AE = []
    ME = []
    for k in range(len(lines))[1:]:
        [ids, einstein_angle, mass_in_RE] = lines[k].split()
        iid.append(int(ids))
        AE.append(float(einstein_angle))
        ME.append(float(mass_in_RE))  # [Msun/h]
    iid = np.asarray(iid)
    AE = np.asarray(AE)
    ME = np.asarray(ME)
    if dataformat == 'dictionary':
        Glafic = {'ID' : iid,
                  'AE' : AE,
                  'ME' : ME}
        return Glafic
    else:
        return iid, AE, ME


def simulation_units(name):
    """
    Read parameter file of simulation to find the units of outputs

    Input:
        name - name of simulation directory or parameter filename
    Output:
        unit_length - unit of distance (e.g. kpc, Mpc)
    """
    if os.path.isdir(name):
        dirname = name
        if os.path.exists(dirname+'/arepo/param.txt'):
            filename = dirname+'arepo/param.txt'
            shell_command = 'grep UnitLength_in_cm ' + filename
            line = os.popen(shell_command).read()
            strings = line.split()
            unit_conversion = float(strings[1])
            exp = np.floor(np.log10(np.abs(unit_conversion))).astype(int)
            if exp == 21:  # output in [kpc]
                scale_length = 1e-3
            elif exp == 23:  # output in [Mpc]
                scale_length = 1.0
        else:
            print( "directory of parameter file not found -> ", dirname)
    elif os.path.isfile(name):
        filename = name
        if os.path.exists(filename):
            shell_command = 'grep UnitLength_in_cm ' + filename
            line = os.popen(shell_command).read()
            strings = line.split()
            unit_conversion = float(strings[1])
            exp = np.floor(np.log10(np.abs(unit_conversion))).astype(int)
            if exp == 21:  # output in [kpc]
                scale_length = 1e-3
            elif exp == 23:  # output in [Mpc]
                scale_length = 1.0
        elif (not os.path.exists(filename)):
            print( "parameter file not found -> ", filename)
    return scale_length
