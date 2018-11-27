#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# ulimit -a (report limits) (e.g. -n -s -q hard)

#SBATCH -t 02:00:00
#SBATCH -J GRLPPRvir 
#SBATCH -o GRLPPRvir.out
#SBATCH -e GRLPPRvir.err
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

simname=L62_N512_F6_kpc
hfname=Subfind  #[Subfind, Rockstar]
format=Box  #[Box, Lightcone]
lenses=1    #[1=True, 0=False]
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/${simname}/
labase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/${hfname}/${simname}/${format}/
rksbase=/cosma5/data/dp004/dc-beck3/rockstar/full_physics/${simname}/ #halos_${snapnum}.dat
outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingPostProc/full_physics/${hfname}/${format}/

if [ ${hfname} == 'Subfind' ]; then
    halofinderdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/${simname}/
elif [ ${hfname} == 'Rockstar' ]; then
    halofinderdir=/cosma5/data/dp004/dc-beck3/rockstar/full_physics/${simname}/
fi
rad_for_mdyn=Rvir

# Execute script
if [ ${format} == 'Box' ]; then
    snapnum=39
    python -u ./LPP_main_box.py $snapnum $simdir $labase $hfname $halofinderdir $outbase $rad_for_mdyn $lenses

elif [ ${format} == 'Lightcone' ]; then
    python -u ./LPP_main_lc.py $simdir $labase $hfname $halofinderdir $outbase $rad_for_mdyn $lenses
fi
