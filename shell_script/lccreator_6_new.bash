#!/bin/bash -l

# SBATCH -L /bin/bash
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -J lensingmaps
#SBATCH -o lensingmaps.out
#SBATCH -e lensingmaps.err
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive

# Change to the directory where the job was submitted
cd /cosma5/data/dp004/dc-beck3/shell_script/

# Simulate Directory
szproject=/cosma6/data/dp004/dc-arno1/SZ_project/
# Directory to Codes Head-Quater
codes_HQ=/cosma5/data/dp004/dc-beck3/
# Directory of glafic scripts
glaficdir=/cosma5/data/dp004/dc-beck3/glafic_lc/
# Shell Scripts
shells=/cosma5/data/dp004/dc-beck3/shell_script/
# Directory of Lightcone scripts
lightcone=/cosma5/data/dp004/dc-beck3/LightCone/
# Directory of Lensing scripts
lensing=/cosma5/data/dp004/dc-beck3/LensingMap
# Halo Finder Directory
declare -a HaloFinder=(# Christians Subfind output
					   '0 /cosma6/data/dp004/dc-arno1/SZ_project/non_radiative_hydro/ Subfind'
					   # Rockstar Halo Finder
					   '1 /cosma5/data/dp004/dc-beck3/rockstar/ Rockstar'
					   # AHF Halo Finder
					   '0 /cosma5/data/dp004/dc-beck3/AHF/ AHF'
					   )

declare -a todo=('0'    # Rockstar
                 # Select halos and subhalos
                 '0'    # Light Cone creation
                 '0'    # Populate Light Cone with supernovae
				 '0'    # Run M. Oguris grav. lensing code
                 '1'	# Create l.o.s. surface density map of lens
				 '0'    # Sub-grid halo mass app. with NFW
				 ) 
# 0 : run simulation
# 1 : don't run simulation
declare -a Simulations=('0 non_radiative_hydro/ L62_N512_GR red Mpc'
						'0 non_radiative_hydro/ L62_N512_F5 blue Mpc'
						'0 non_radiative_hydro/ L62_N512_F6 green Mpc'
						'1 full_physics/ L62_N512_GR_kpc red kpc'
						'0 full_physics/ L62_N512_F5_kpc blue kpc' # not finished yet
						'1 full_physics/ L62_N512_F6_kpc green kpc'
						'0 full_physics/ L62_N512_GR red Mpc'
						'0 full_physics/ L62_N512_F5 blue Mpc'
						'0 full_physics/ L62_N512_F6 green Mpc'
						)

#declare -a lc_param=('alpha 0.8'
#				 	 'zmax 1.'
#				 	 'observer_position [0, 0, 0]'
#					 'cone_axis [1, 0, 0]'
#				 	 )

# Check if AnSimSettings.txt exist
# if so empty it, if not create
if [ -e LCSettings.txt ]; then
	echo 'Rewrite LCSettings.txt'
	> LCSettings.txt
else
	echo 'Create LCSettings.txt'
	touch LCSettings.txt
fi

# Create LCSettings.txt file
#echo 'Directory of Simulations ---------------------------' >> LCSettings.txt
#echo $simdir >> LCSettings.txt
echo 'Directory of Python Scripts ---------------------------' >> LCSettings.txt
echo $codes_HQ >> LCSettings.txt
for hf in "${HaloFinder[@]}"
do
	set -- $hf  # Identify strings in string
	onoff_hf=$1
	if [ $onoff_hf = '1' ]; then
		hfdir=$2
		hfname=$3
		break
	else
		continue
	fi
done

# Create LCSettings.txt file
echo -e >> LCSettings.txt  # write blank line
for sims in "${Simulations[@]}"
do
	set -- $sims  # Identify strings in string
	onoff_sim=$1
	astrophy=$2
	simname=$3
	rest=$4
    unit=$5
	if [ $onoff_sim = '1' ]; then
		echo 'Simulation to analyse ----------------------' >> LCSettings.txt
		echo "$astrophy $simname" >> LCSettings.txt
		echo "$szproject$astrophy$simname/ $rest $unit" >> LCSettings.txt
		echo "$hfdir$astrophy$simname/ $hfname" >> LCSettings.txt
		echo "$lightcone$astrophy" >> LCSettings.txt
		echo "$glaficdir$astrophy" >> LCSettings.txt
	else
		continue
	fi
done

# Load Module
# For some reason hdfview causes error on purge if not loaded
module load hdfview/2.13.0
module purge
module load intel_comp/c4/2015 platform_mpi/9.1.2 python/2.7.13

# Run through light cone types
for ((i=0; i<${#todo[*]}; i++));
do
	if (( $i == 0 )) && [ ${todo[i]} != '0' ]; then
		echo 'Create Halo catalogue'
		/cosma5/data/dp004/dc-beck3/shell_script/rockstar_6.bash LCSettings.txt
    fi
    # ------------
	if (( $i == 1 )) && [ ${todo[i]} != '0' ]; then
		echo 'Create Light-cone'
		python $lightcone/LC_create.py && echo 'Created Light-cone'
	fi
	# ------------
	if (( $i == 2 )) && [ ${todo[i]} != '0' ]; then
		echo 'Populate Light-cone with SNeIa'
		python $lightcone/LC_SNeIa.py && echo 'Populated with SNeIa'
	fi
	# ------------
	if (( $i == 3 )) && [ ${todo[i]} != '0' ]; then
		echo 'Solve lensing equation'
		for sims in "${Simulations[@]}"
		do
			set -- $sims  # Identify strings in string
			onoff_sim=$1
			sim="simname=$2"
			glaficex=glafic_5.bash
			if [ $onoff_sim = '1' ]; then
				sed -i "13s/.*/$sim/" "$shells$glaficex"
				bsub < $shells$glaficex
			fi
			# Only continue if checkfile exists
			checkfile=$shells"DONE_"$2".txt"
			#while read i; do if [ "$i" = $checkfile ]; then break; fi; done \
			#	   < <(inotifywait  -e create,open --format '%f' --quiet /tmp --monitor)
			while [ ! -f $checkfile ]; do sleep 1; done
		done
	fi
	# ------------
	if (( $i == 4 )) && [ ${todo[i]} != '0' ]; then
		echo 'L.o.s. Surface Density Map of Lens'
        MPLBACKEND=Agg python $lensing/LM_create.py 'Density Map created'
	fi
    # -----------
    # for Power spectrum load modules:
    # module load python/2.7.13 gnu_comp/c4/4.7.2
done
