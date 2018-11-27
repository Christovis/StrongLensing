#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# ulimit -a (report limits) (e.g. -n -s -q hard)

#SBATCH -t 20:00:00
#SBATCH -J GRLA 
#SBATCH -o GRLA.out
#SBATCH -e GRLA.err
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
format=Box      #[Box, Lightcone]
gridres=128     #[pixels]
hfname=Subfind  #[Rockstar, Subfind]
lenses=0        #[1=True, 0=False]
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/${simname}/

# Execute script
if [ ${format} == 'Box' ]; then
    snapnum=39
    dmbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/${hfname}/${simname}/Box/z_${snapnum}/
    outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/${hfname}/${simname}/Box/
    python -u ./LM_main_box.py $snapnum $simdir $hfname $dmbase $gridres $outbase $lenses

elif [ ${format} == 'Lightcone' ]; then
    outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/${hfname}/${simname}/Lightcone/
    lightconedir=/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/${hfname}/LC_SN_${simname}
    dmbase=/cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/${hfname}/${simname}/Lightcone/
    python -u ./LM_main_lc.py $simdir $hfname $dmbase $lightconedir $outbase $gridres
    #valgrind --log-file="valgrind_output_2.txt" python -u ./LM_main_lc.py
fi

#nproc="$(find /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/z_${snapnum}/*.h5 -type f -size +4M | wc -l)"
# nproc="$(ls -lR /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/*.h5 | wc -l)"
#echo "Start process with $nproc cpus"
#valgrind --log-file="valgrind_output_2.txt" python LM_main.py
#mpirun -np 1 python3 ./LM_main_mpi.py \
#python3 -u -X faulthandler ./LM_main.py $snapnum $ncells $simdir $dmbase $outbase
#echo "run LM_main_mpi.py $simdir $dmbase $snapnum $ncells $outbase" > gdb.in
#mpirun -np 1 xterm -e "gdb -x gdb.in python3"
#mpirun -np 1 python3 ./LM_main_mpi.py /cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/ /cosma5/data/dp004/dc-beck3/StrongLensing/DensityMap/full_physics/L62_N512_GR_kpc/ 40 1024 /cosma5/data/dp004/dc-beck3/StrongLensing/LensingMap/full_physics/Rockstar/L62_N512_GR_kpc/
