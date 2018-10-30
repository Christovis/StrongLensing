#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# 40   35   30   23
# 0.16 0.35 0.57 0.95

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH -t 40:00:00
#SBATCH -J DM2lc 
#SBATCH -o DM2lc.out
#SBATCH -e DM2lc.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/2.7.15

simname=GR  #[GR, F6, F5,]
format=Lightcone  #[Box, Lightcone]
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_${simname}_kpc/
halofinderdir=/cosma5/data/dp004/dc-beck3/rockstar/full_physics/L62_N512_${simname}_kpc/
gridres=512  #[pixels]
walltime=40  #[h]

# Execute script
if [ ${format} == 'Box' ]; then
    snapnum=40
    outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_${simname}_kpc/Box//
    mpirun -np 3 python ./DM_main_box.py \
        $simdir $halofinderdir $snapnum $gridres $outbase
elif [ ${format} == 'Lightcone' ]; then
    outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_${simname}_kpc/Lightcone/
    lightconedir=/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/Rockstar/LC_SN_L62_N512_${simname}_kpc_2.h5
    python ./DM_main_lc.py \
        $simdir $halofinderdir $lightconedir $gridres $walltime $outbase
    #mpirun -np 2 python ./DM_main_lc_mpi.py \
    #    $simdir $halofinderdir $lightconedir $gridres $walltime $outbase
else
    echo 'This format does not exit.'
fi

