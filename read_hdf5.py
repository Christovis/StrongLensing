'''Routines for reading Gadget and AREPO snapshots and group catalogs in a convenient way
The library needs the readsnap and readsubf routines if Gadget formats are to be read.
In order to read data from a snapshot or group catalog, an instance of the snapshot class has to be created specifying simulation directory and snapshot number.
Author: Christian Anrold'''

from pylab import *
import readsnap as rs
import readsubf
from scipy.optimize import curve_fit
from gc import collect
import pickle
import os
import sys
import h5py


class constants:
    '''Class to easily access commonly used constants within a snapshot class'''
    def __init__(self, snapshot):
        self.h = snapshot.header.hubble
        self.G = float64(6.67384e-11/((3.08567758e22)**3)*1.9891e30) # Mpc**3/(M_solar*s**2) ## 6.67398e-11 m3 kg-1 s-2
        self.H = 100*self.h/3.08567758e22*1000#1/s
        self.rho_crit = float64(3*(self.H**2)/(8*pi*self.G))#M_solar/Mpc**3
        self.rho200 = 200*self.rho_crit # M_solar/Mpc**3
        self.keV = float64(1000.0*1.602e-19)#J/keV
        self.mproton = float64(1.67262e-27) #kg 
        self.f = 0.76
        self.mmean = (1.0+(1.0-self.f)/self.f)/(2.0+3.0*(1.0-self.f)/(4.0*self.f))
        self.Mpc = 3.08567758e22 # m
        self.c = 3.e8  # m/s
        self.c_Mpc = 3.e8 / self.Mpc # Mpc/s

class header:
    '''Class containing the header part of a snapshot'''
    def __init__(self, snapshot):
        self.filename = snapshot.headername
        self.format = 3
        
        f = h5py.File(self.filename,'r')
      
        self.attrs = list(f['/Header'].attrs.items())
        self.npart = f['/Header'].attrs['NumPart_ThisFile']
        self.num_total = f['/Header'].attrs['NumPart_Total']
        self.massarr = f['/Header'].attrs['MassTable']
        self.time = f['/Header'].attrs['Time']
        self.redshift = f['/Header'].attrs['Redshift']
        self.sfr = f['/Header'].attrs['Flag_Sfr']
        self.feedback = f['/Header'].attrs['Flag_Feedback']
        self.nall = f['/Header'].attrs['NumPart_Total']
        self.cooling = f['/Header'].attrs['Flag_Cooling']
        self.filenum = f['/Header'].attrs['NumFilesPerSnapshot']
        self.boxsize = f['/Header'].attrs['BoxSize']
        self.omega_m = f['/Header'].attrs['Omega0']
        self.omega_l = f['/Header'].attrs['OmegaLambda']
        self.hubble = f['/Header'].attrs['HubbleParam']
        self.swap = 0
      
        f.close()

class hdf5_names:
    '''Class to translate the old four-letter identifiers to the hdf5 names in the snaopsnot and group files'''
    def __init__(self, snapshot):
        self.name = {}
        self.name['POS '] = 'Coordinates'
        self.name['MASS'] = 'Masses'
        self.name['VEL '] = 'Velocities'
        self.name['ID  '] = "ParticleIDs"            
        self.name['U   '] = "InternalEnergy"      
        self.name['RHO '] = "Density"    
        self.name['VOL '] = "Volume"             
        self.name['NE  '] = "ElectronAbundance"  
        self.name['NH  '] = "NeutralHydrogenAbundance"   
        self.name['HSML'] = "SmoothingLength"       
        self.name['SFR '] = "StarFormationRate"      
        self.name['AGE '] = "StellarFormationTime"    
        self.name['Z   '] = "Metallicity"          
        self.name['BHMA'] = "BH_Mass"         
        self.name['ACCE'] = "Acceleration"          
        self.name['MGPH'] = "ModifiedGravityPhi"        
        self.name['MGGP'] = "ModifiedGravityGradPhi"     
        self.name['MGAC'] = "ModifiedGravityAcceleration"


class snapshot:
    '''Snapshot class; reads header and constants at initialisation; contains methods for reading particle data and group catalogs.
Create an instance of the class using:
my_snapshot = snapshot(snapnum, <directory>, <dirbases>, <snapbases>, <exts>)

Arguments:
snapnum     The simulation output number.
<directory>     The output directory of the simualtion, optional, default './'.
<dirbases>      A list of possible directory names for the snapshot directories, optional (normally not needed), default '["snapdir_", ""]'
<snapbases>     A list of possible snapshot names, optional (normally not needed), default '["snap_"]'    
<exts>      A list of possible file extensions, optional (normally not needed), default '["", ".hdf5"]'


Usage Example:

my_snapshot = snapshot(30, "/my/simulation/directory")

This will load snapshot number 30 in the specified directory.
'''
    def __init__(self, snapnum, directory = "./", dirbases = ["snapdir_", ""], snapbases = ["/snap_"], exts = ["", ".hdf5"]):
        self.directory = directory
        self.snapnum = snapnum
        found_files = False

        for dirbase in dirbases:
            for snapbase in snapbases:
                for dirnum in ["", str(snapnum).zfill(3)]:
                    for ext in exts:
                        try_file  = directory + dirbase + dirnum + snapbase  + str(snapnum).zfill(3) + ".0" + ext
                        print("trying file " + try_file)
                        if os.path.exists(try_file):
                            self.headername = try_file
                            self.snapname = directory + dirbase + dirnum + snapbase  + str(snapnum).zfill(3)
                            found_files = True

        if not found_files:
            print("Headerfiles not found." + directory) 
            sys.exit()
        else:
            print("Headername: " + self.headername)
            print("Sanpname: " + self.snapname)

        #--- use new routine only for hdf5 snapshots ---
        if self.headername[-4:] == 'hdf5':
            self.hdf5 = True
            self.header = header(self)
            hn = hdf5_names(self)
            self.hdf5_name = hn.name

        #--- otherwise import readsnap and readsubf to read the old gadget format ---
        else:
            self.hdf5 = False
            rs = __import__('readsnap')
            readsubf = __import__('readsubf')
            self.header = rs.snapshot_header(self.headername)

            
        self.time = self.header.time
        self.const = constants(self)
        self.data = {}
        
    def read(self, blocklist, parttype = -1):
        '''Reading method to load particle data from snapshots.
        my_snapshot.read(blocklist, parttype = [0,1])

        Arguments:
        blocklist    List of hdf5 block names to be read (see: 'my_snapshot.show_snapshot_contents()')
        parttype     List of parttypes for which the data should be read, optional, default '-1' (read all types)

        Usage Example: 

        my_snapshot.read(['Velocities', 'Coordinates'], parttype = [0,1])

        Will read coordinates and velocities for gas and dm from the snapshot.
        The data is accessible through 

        my_snapshot.data
        '''
        print("Reading " + str(blocklist) + "from snapshot")
        if type(blocklist) == str:
            blocklist = [blocklist]
        if not self.hdf5: #use the old method
            for block in blocklist:
                if block == "POS ":
                    self.pos = rs.read_block(self.snapname, "POS ", parttype = parttype)/self.const.h
                    self.data["POS "] = self.pos
                elif block == "VEL ":
                    self.vel = rs.read_block(self.snapname, "VEL ", parttype = parttype)
                    self.data["VEL "] = self.vel
                elif block == "MASS":
                    self.mass = rs.read_block(self.snapname, "MASS", parttype = parttype)*1e10/self.const.h
                    self.data["MASS"] = self.mass
                else:
                    self.data[block] = rs.read_block(self.snapname, block, parttype=parttype)
        else: #use the faster hdf5 reading routines
            self.read_hdf5(blocklist, parttype)

    def get_unit_factor(self, block):
        '''Helper method'''
        self.length_blocks = ['GroupCM', "Coordinates", 'GroupPos', 'Group_R_Crit200', 'Group_R_Crit500', 'Group_R_Mean200', 'Group_R_TopHat200', 'SubhaloCM', 'SubhaloHalfmassRad', 'SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloVmaxRad']
        self.mass_blocks = ['Masses', 'SubhaloMass', 'SubhaloMassInHalfRad', 'SubhaloMassInHalfRadType', 'SubhaloMassInMaxRad', 'SubhaloMassInMaxRadType', 'SubhaloMassInRad', 'SubhaloMassInRadType', 'SubhaloMassType', 'Group_M_Crit200', 'Group_M_Crit500', 'Group_M_Mean200', 'Group_M_TopHat200', 'Group_MassType_Crit200', 'Group_MassType_Crit500', 'Group_MassType_Mean200', 'Group_MassType_TopHat200', 'GroupMass', 'GroupMassType']
        if block in self.length_blocks:
            factor = 1./self.const.h
        elif block in self.mass_blocks:
            factor = 1e10/self.const.h
        else:
            factor = 1.

        return factor

    def check_for_blocks(self, f, blocklist, parttype):
        '''Helper method'''
        self.blockpresent = {}

        if parttype == -1:
            parttype = []
            for pt in range(6):
                if 'PartType' + str(pt) in list(f.keys()):
                    parttype.append(pt)

        for block in blocklist:

            self.blockpresent[block] = []
            for pt in parttype:
                if block in list(f['PartType' + str(pt) + '/'].keys()):
                    self.blockpresent[block].append(pt)
                elif block == 'Masses' and f['Header/'].attrs['MassTable'][pt] > 0:
                    self.blockpresent[block].append(-pt)
                    

        print("Found the following data in " + f.filename + str(self.blockpresent))

    def get_tot_num_part(self, parttype):
        '''helper method'''
        self.header.num_total = zeros(6, dtype = int64)
        files = self.determine_files(self.snapname + '.')

        for fn in files:
            fname = self.snapname + '.' + str(fn) + '.hdf5'
            self.f = h5py.File(fname)
            part_this_file = self.f['/Header/'].attrs['NumPart_ThisFile']
            self.header.num_total += part_this_file
        print("Total number of particles:" + str(self.header.num_total[parttype]))
        return self.header.num_total[parttype]

    def create_data_array(self, f, blocklist):
        '''Helper method.'''
        self.partcounter = zeros(6, dtype = int64)
        for block in blocklist:
            self.data[block] = {}
            for pt in self.blockpresent[block]:
                if pt >= 0:
                    datashape = 1
                    if len(f['PartType' + str(pt) + '/' + block + '/'].shape) > 1:
                        datashape = f['PartType' + str(pt) + '/' + block + '/'].shape[1]
                    datatype = f['PartType' + str(pt) + '/' + block + '/'].dtype
                    datalen = self.header.num_total[pt]
                    if datalen < self.header.npart[pt]:
                        datalen = self.get_tot_num_part(pt)
                    

                    if datashape > 1:
                        self.data[block][self.parttypes(pt)] = zeros((datalen, datashape), dtype = datatype)
                    else: 
                        self.data[block][self.parttypes(pt)] = zeros(datalen, dtype = datatype)
                else:
                    factor = self.get_unit_factor(block)
                    self.data[block][self.parttypes(-pt)] = ones(self.header.num_total[-pt]) * f['Header/'].attrs['MassTable'][-pt] * factor

    def show_snapshot_contents(self):
        '''This function prints the available data fields contined in this snapshot.
        Usage:
        my_snapshot.show_snapshot_contents()
        '''
        fname = self.snapname + '.' + str(0) + '.hdf5'
        self.f = h5py.File(fname)
        
        print("")
        print("----------------------------------------")
        print("Snapshot data for file: " + fname)
        print("----------------------------------------")
        print("Available data fields:")
        print("----------------------------------------")
        for k in self.f.keys():
            print(k)
        
        for k in self.f.keys():
            if k in ["Header", "Config", "Parameters"]:
                print("----------------------------------------")
                print(k + " contents: ")
                print("----------------------------------------")
                for i in self.f[k].attrs.keys():
                    print(i)
                print("")
            else:
                print("")
                print("----------------------------------------")
                print("Contents of data field: " + k)
                print("----------------------------------------")
                for i in self.f[k].keys():
                    print(i)
        self.f.close()

    def translate_blocklist(self, blocklist):
        '''Helper method'''
        if type(blocklist) == str:
            blocklist = [blocklist]
            
        translate = True
        for block in blocklist:
            if len(block) != 4 or block == "Mass":
                translate = False
                break

        if not translate:
            return blocklist

        else:
            new_blocklist = []
            for block in blocklist:
                new_blocklist.append(self.hdf5_name[block])
            

    def read_hdf5(self, blocklist, parttype):
        '''helper method'''
        files = self.determine_files(self.snapname + '.')
        blocklist = self.translate_blocklist(blocklist)

        for fn in files:
            fname = self.snapname + '.' + str(fn) + '.hdf5'
            if fn%10 == 0:
                print("reading file" + fname)
            f = h5py.File(fname)
            
            if fn == 0:
                self.check_for_blocks(f, blocklist, parttype)
                self.create_data_array(f, blocklist)

            part_this_file = f['/Header/'].attrs['NumPart_ThisFile']
            print(part_this_file)
            for block in blocklist:
                factor = self.get_unit_factor(block)
                name = block
                    
                for pt in self.blockpresent[block]:
                    if pt >= 0:
                        self.data[block][self.parttypes(pt)][self.partcounter[pt]:self.partcounter[pt]+part_this_file[pt]] = f['PartType' + str(pt) + '/' + block + '/'].value * factor
                    
            self.partcounter += part_this_file
            f.close()

    def parttypes(self, type_id):
        '''Helper method'''
        if type_id == 0:
            return 'gas'
        elif type_id == 1:
            return 'dm'
        elif type_id == 3:
            return 'traces'
        elif type_id == 4:
            return 'stars'
        elif type_id == 5:
            return 'bh'
        else:
            raise ValueError

    
    def group_catalog(self, hdf5_names = ['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200'], masstab = True, group_veldisp = True, file_prefix = "", files = -1, path = '', dirname = 'groups_', filename = 'fof_subhalo_tab_'):
        '''Read data from the group catalog corresponding to the snapshot.
        Usage:
        my_snapshot.group_catalog(<hdf5_names>, <masstab>, <group_veldisp>, <file_prefix>, <files>, <path>, <dirname>, <filename>)

        Arguments:
        hdf5_names       List of hdf5 names of the data fields to be loaded (see  'my_snapshot.show_group_catalog_contents()'), optional, default '['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200']'
        masstab      Only needed for Gadget format, optional
        group_veldisp    Only needed for Gadget format, optional 
        file_prefix      Prefix for the group directory, optional, default ''
        files        List of files to be loaded from the group catalog, optional, default '-1' (all files)
        path         path where the group catalog is stored, optional, default: same path as snapshot data
        dirname      directory name for the group catalog subdirectories, optional, default 'groups_'
        filename     filename for the individual catalog files, optional, default '/fof_subhalo_tab_'

        Example:
        my_snapshot.group_catalog(['GroupPos', 'SubhaloPos']) 
        This will load the positions of all groups and subhalos.
        '''
        if not self.hdf5:
            self.cat = readsubf.subfind_catalog(self.directory + file_prefix, self.snapnum, masstab=masstab, group_veldisp = group_veldisp)
        else:
            self.fast_group_catalog(hdf5_names = hdf5_names, files = files, path = path, dirname = dirname, filename = filename, file_prefix = file_prefix)

    def show_group_catalog_contents(self, path = '', dirname = 'groups_', filename = 'fof_subhalo_tab_', file_prefix = ''):
        '''This Function will print the available data fields for the group catalog corresponding to this snapshot. 
        
        Usage:
        my_snapshot.show_group_catalog_contents()

        See 'group_catalog()' for optional arguments.

        '''
        if path == '':
            path = self.directory + file_prefix + '/' + dirname + str(self.snapnum).zfill(3) + '/' + filename + str(self.snapnum).zfill(3) + '.'
        
        fname = path + str(0) + '.hdf5'
        self.f = h5py.File(fname)
        print("")
        print("----------------------------------------")
        print("Group catalog data for file: " + fname)
        print("----------------------------------------")
        print("Header contents: ")
        print("----------------------------------------")
        for k in self.f["Header"].attrs.keys():
            print(k)
        print("")
        print("----------------------------------------")
        print("Group data: ")
        print("----------------------------------------")
        for k in self.f["Group/"].keys():
            print(k)
        print("")
        print("----------------------------------------")
        print("Subhalo data: ")
        print("----------------------------------------")
        for k in self.f["Subhalo/"].keys():
            print(k)
        print("----------------------------------------")
        self.f.close()

    def fast_group_catalog(self, hdf5_names = ['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200'], files = -1, path = '', dirname = 'groups_', filename = 'fof_subhalo_tab_', file_prefix = '', show_data = False):
        '''Helper method'''
        if path == '':
            path = self.directory + file_prefix + '/' + dirname + str(self.snapnum).zfill(3) + '/' + filename + str(self.snapnum).zfill(3) + '.'

        print("Reading" + str(hdf5_names) + "from hdf5 group catalog" + path)

            # --- find the files ---
        if files == -1:
            files  = self.determine_files(path)

        self.cat = {}
        
        # --- iterate over files ---
        group_counter  = 0
        sub_counter = 0
        for fn in files:
            fname = path + str(fn) + '.hdf5'
            self.f = h5py.File(fname)
        
            if fn % 10 == 0:
                print("Reading file" + fname)

            ng = self.f['Header/'].attrs['Ngroups_ThisFile']
            ns = self.f['Header/'].attrs['Nsubgroups_ThisFile']

            # --- create empty arrays for the data ---
            if fn == 0:
                #--- read header of the first file --- 
                self.cat['n_groups'] = self.f['Header/'].attrs['Ngroups_Total']
                self.cat['n_subgroups'] = self.f['Header/'].attrs['Nsubgroups_Total']
                for key in self.f["/Header"].attrs.keys():
                    self.cat[key] = self.f["/Header"].attrs[key]
                    
                # --- create data arrasys for groups and subhalos ---
                for hn in hdf5_names:
                    sh = 1


                    if hn[0] == 'G': 
                        if len(self.f['Group/'+hn].value.shape) > 1:
                            sh = self.f['Group/'+hn].value.shape[1]
                        if sh>1:
                            self.cat[hn] = zeros((self.cat['n_groups'], sh))
                        else:
                            self.cat[hn] = zeros(self.cat['n_groups'])

                    elif hn[0] == 'S':
                        if len(self.f['Subhalo/'+hn].value.shape) > 1:
                            sh = self.f['Subhalo/'+hn].value.shape[1]
                        if sh > 1:
                            self.cat[hn] = zeros((self.cat['n_subgroups'], sh))
                        else: 
                            self.cat[hn] = zeros(self.cat['n_subgroups'])

                    else:
                        raise ValueError("can't deal with that", hn, hn[0])
            
            # --- read the data ---
            for hn in hdf5_names:
                unit_factor = self.get_unit_factor(hn)
                if hn[0] == 'G' and ng > 0:
                    self.cat[hn][group_counter:group_counter + ng] = self.f['Group/'+hn].value * unit_factor
                elif hn[0] == 'S' and ns > 0:
                    self.cat[hn][sub_counter:sub_counter + ns] = self.f['Subhalo/'+hn].value * unit_factor

            group_counter += ng
            sub_counter += ns
            self.f.close()

    def determine_files(self, path):
        '''Helper Routine'''
        if not os.path.exists(path + '0.hdf5'):
            raise ValueError("File", path + '0.hdf5', " not found")
        
        files = 0
        while True:
            if not os.path.exists(path + str(files) + '.hdf5'):
                print("Found " + str(files) + "files")
                return arange(files)
            else:
                files += 1




if __name__ == "__main__": # doctest for the module
    #--- the code below is an example of how the library can be used ---
    filename = "../../dark_matter_only/L62_N512_GR/" #the directory containing the "snapdir_xxx" or "groups_xxx" subdirectories
    #--- create an object of the class snapshot, give directory and snapshot number as parameters ---
    s = snapshot(45, filename)

    #--- display the contents of the group catalog and snapshot ---
    s.show_group_catalog_contents()
    s.show_snapshot_contents()

    #--- read the velocities, coordinates and masses for all particle types ---
    s.read(["Velocities", "Coordinates", "Masses"], parttype = -1)

    #--- read Subhalo_Vmax, Geroup length table and Group positions ---
    s.group_catalog(["SubhaloVmax", "GroupLenType", "GroupPos"])
    
    #--- get the coordinates for the dm particles ---
    print(s.data['Coordinates']['dm'])

    #--- access group catalog ---
    print(s.cat['GroupLenType'])


    #--- further tests --- 
    t = snapshot(44, filename)
    t.read(["ParticleIDs"])
    t.group_catalog(["Subhalo_Jgas"])

