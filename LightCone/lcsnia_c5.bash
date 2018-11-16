#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos
# ulimit -a (report limits) (e.g. -n -s -q hard)

#SBATCH -t 20:00:00
#SBATCH -J f61LCSbf 
#SBATCH -o f61LCSbf.out
#SBATCH -e f61LCSbf.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

ulimit -n hard
ulimit -s hard
ulimit -q hard

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/3.6.5
# sim_dir, sim_phy, sim_name, sim_col, hf_dir, hf_name, lc_dir, dd, HQ_dir


simname=L62_N512_F6_kpc
lcnumber=1
zmax_snia=2  # maximum redshift of supernovae type 1a
hfname=Subfind  #[Subfind, Rockstar]
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/${simname}/
lcname=/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/${hfname}/LC_${simname}_${lcnumber}.h5
outbase=/cosma5/data/dp004/dc-beck3/StrongLensing/LightCone/full_physics/${hfname}/

# Execute script
if [ ${hfname} == 'Subfind' ]; then
    hfdir=/cosma6/data/dp004/dc-arno1/SZ_project/
    python3 -u ./LC_SNeIa.py $simname $simdir $hfname $hfdir $lcname $lcnumber $zmax_snia $outbase
elif [ ${hfname} == 'Rockstar' ]; then
    hfdir=/cosma5/data/dp004/dc-beck3/rockstar/
    python3 -u ./LC_SNeIa.py $simname $simdir $hfname $hfdir $lcname $lcnumber $zmax_snia $outbase
fi
