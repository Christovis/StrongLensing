from __future__ import absolute_import, division, print_function
import os
import sys
import logging
import argparse
import scipy
import numpy as np
from astropy import units as u
from astropy.cosmology import LambdaCDM
import h5py
sys.path.insert(0, '..')
import readsnap
import readlensing as rf
import LA_create as LA
import la_tools as la

############################################################################
# Set up logging and parse arguments
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG, datefmt='%H:%M:%S')
logging.info('Holiii! Como estas hoy?')

# Parse arguments
parser = argparse.ArgumentParser(
        description='Toolbox to analyze strong grav. lenses and sources')

parser.add_argument("-ldlm", "--dynlensmass", action="store_true",
        help="Comparisson between dynamical and lensing mass within Rein.")
parser.add_argument("-lmr", "--mvirtheta", action="store_true",
        help="M200 mass vs. Einstein radii.")
parser.add_argument("-dtmu", "--timedelaymagnification", action="store_true",
        help="Time delay vs. Magnification.")

parser.add_argument("-lh", "--histogramoflenses", default="False",
        help='Histogram types: "redshift", "mass"')
parser.add_argument("-sh", "--histogramofsources", default="False",
        help='Histogram types: "redshift", "mass"')

args = parser.parse_args()

logging.info('Startup options:')
logging.info('  Dyn. vs. Lens Mass:            %s', args.dynlensmass)
logging.info('  M200 vs. theta_E:              %s', args.mvirtheta)
logging.info('  Delta T vs. Mu:                %s', args.timedelaymagnification)
logging.info('  Histograms of Lenses:          %s', args.histogramoflenses)
logging.info('  Histograms of Sources:         %s', args.histogramofsources)
#logging.info('  ML-based strategies available: %s', loaded_ml_strategies)

# Sanity checks
#assert args.histogramoflenses in ['mass', 'redshift']
#assert args.histogramofsources in ['mass', 'redshift']

############################################################################
# Load Simulation Specifications
LCSettings = '/cosma5/data/dp004/dc-beck3/shell_script/LCSettings.txt'
sim_dir, sim_phy, sim_name, sim_col, dd, hf_name, lc_dir, dd, HQ_dir = rf.Simulation_Specs(LCSettings)

# Node = 16 CPUs
CPUs = 5 # Number of CPUs to use has to be the same as in LensingMap/LM_create.py

###########################################################################
# Start calculation

if args.dynlensmass:
    assert LA
    LA.dyn_vs_lensing_mass(CPUs, sim_dir, sim_phy, sim_name, hf_name, lc_dir, HQ_dir)

elif args.mvirtheta:
    assert LA
    LA.M200_Rein(CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)

elif args.timedelaymagnification:
    assert LA
    LA.deltat_mu(CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)

elif args.histogramoflenses == 'redshift':
    assert LA
    LA.histogram_lens('Halo_z', CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)

elif args.histogramoflenses == 'mass':
    assert LA
    LA.histogram_lens('M200', CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)

elif args.histogramofsources== 'delta_t':
    assert LA
    LA.histogram_src('delta_t', CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)

elif args.histogramofsources== 'magnification':
    assert LA
    LA.histogram_src('mu', CPUs, sim_dir, sim_phy, sim_name, lc_dir, HQ_dir)
logging.info("Terminado -- ten un buen dia!")
