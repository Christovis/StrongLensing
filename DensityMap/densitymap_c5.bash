#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# SBATCH --ntasks 

#SBATCH --nodes 2
#SBATCH --tasks-per-node=16
#SBATCH -t 46:00:00
#SBATCH -J F5DMz35 
#SBATCH -o F5DMz35.out
#SBATCH -e F5DMz35.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/3.6.5

simname=F5
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_${simname}_kpc/
halofinderdir=/cosma5/data/dp004/dc-beck3/rockstar/full_physics/L62_N512_${simname}_kpc/
snapnum=35
gridres=1024
outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_${simname}_kpc/
nfileout=10

# Execute script
mpirun -np $SLURM_NTASKS python3 ./DM_main.py \
    $simdir $halofinderdir $snapnum $gridres $outbase $nfileout
