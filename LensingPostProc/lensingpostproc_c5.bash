#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# ulimit -a (report limits) (e.g. -n -s -q hard)

#SBATCH -t 02:00:00
#SBATCH -J F5LPPRvir 
#SBATCH -o F5LPPRvir.out
#SBATCH -e F5LPPRvir.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

ulimit -n hard
ulimit -s hard
ulimit -q hard

# Load Module
module purge
module load python/2.7.15 intel_comp/2018 intel_mpi/2018 fftw valgrind
#module load python/2.7.15 gnu_comp/7.3.0 openmpi fftw/3.3.7

simname=L62_N512_GR_kpc
snapnum=30
format=Lightcone
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/${simname}/
labase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/${simname}/${format}/
rksbase=/cosma5/data/dp004/dc-beck3/rockstar/full_physics/${simname}/halos_${snapnum}.dat
outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/full_physics/Rockstar/${format}/

rad_for_mdyn=Rvir

# Execute script
python -u ./LPP_main_lc.py $snapnum $simdir $labase $rksbase $outbase $rad_for_mdyn

