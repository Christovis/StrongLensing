#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# ulimit -a (report limits) (e.g. -n -s -q hard)

#SBATCH -t 3:00:00
#SBATCH -J GRLAz40
#SBATCH -o GRLAz40.out
#SBATCH -e GRLAz40.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

ulimit -n hard
ulimit -s hard
ulimit -q hard

# Load Module
module purge
module load python/2.7.15 gnu_comp/7.3.0 openmpi fftw/3.3.7

snapnum=40
ncells=1024
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/
dmbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/z_$snapnum/
outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/L62_N512_GR_kpc/Box/
nproc="$(find /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/z_${snapnum}/*.h5 -type f -size +4M | wc -l)"
# nproc="$(ls -lR /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/*.h5 | wc -l)"

echo "Start process with $nproc cpus"

# Execute script
#mpirun -np 1 python3 ./LM_main_mpi.py \
#python3 -u -X faulthandler ./LM_main.py $snapnum $ncells $simdir $dmbase $outbase
python -u ./LM_main.py $snapnum $ncells $simdir $dmbase $outbase

#echo "run LM_main_mpi.py $simdir $dmbase $snapnum $ncells $outbase" > gdb.in
#mpirun -np 1 xterm -e "gdb -x gdb.in python3"

#mpirun -np 1 python3 ./LM_main_mpi.py /cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/ /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/ 40 1024 /cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/L62_N512_GR_kpc/
