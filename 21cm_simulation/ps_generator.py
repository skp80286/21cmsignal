#!python

import numpy as np
import math
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os
# We change the default level of the logger so that
# we can see what's happening with caching.
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

import py21cmfast as p21c

# For plotting the cubes, we use the plotting submodule:
from py21cmfast import plotting

# For interacting with the cache
from py21cmfast import cache_tools

from compute_power_spectrum import ComputePowerSpectrum as CPS

import pickle

import argparse

parser = argparse.ArgumentParser(description='Simulate 21cm cosmological signal and compute powerspectrum.')
parser.add_argument('-d', '--demo', action='store_true', help='Run in demo mode, 10 rows with plots and informational screen output')
parser.add_argument('-n', '--nsets', type=int, default=10000, help='Limit processing to specified number of rows')
parser.add_argument('-f', '--filepath', type=str, default=".", help='directory path for data files')
parser.add_argument('-s', '--sliceindex', type=int, default=40, help='Slice index to plot. Used in demo mode.')
parser.add_argument('-z', '--redshift', type=float, default=9.1, help='redshift')
parser.add_argument('-c', '--cells', type=int, default=80, help='number of cells, each side of cube')
parser.add_argument('-l', '--length', type=int, default=100, help='length of each side of cube in Mpc')
parser.add_argument('-r', '--randomseed', type=int, default=101, help='random seed for simulations')
parser.add_argument('--zetalow', type=int, default=18, help='lower bound of zeta')
parser.add_argument('--zetahigh', type=int, default=200, help='upper bound of zeta')
parser.add_argument('--mminlow', type=float, default=3.69897, help='lower bound of m_min')
parser.add_argument('--mminhigh', type=float, default=5.69897, help='upper bound of m_min')

args = parser.parse_args()
if args.demo:
    if args.nsets > 20:
        args.nsets = 20

##############
# Utility method for plotting
##############
def plot(dT):
    print ('Plotting signal without and with noise')
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.suptitle(f'$z={args.redshift:.2f}')
    plt.subplot(121)
    plt.title('noiseless cube slice')
    plt.pcolormesh(dT[:][args.sliceindex][:])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(dT.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

def plot_power_spectra(psset):
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.title('Spherically averaged power spectra.')
    for i, row in enumerate(psset[:10]):
        label = f'Zeta:{row["zeta"]:.2f}-M_min:{row["m_min"]:.2f}'
        plt.loglog(row['k'][1:40], row['ps'][1:40], label=label)
        plt.annotate(text=label, xy=(row['k'][2*i+1], row['ps'][2*i+1]))
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.legend(loc='lower right')
    plt.show()

print(f"Using 21cmFAST version {p21c.__version__}")

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
pid = str(os.getpid())
cache_dir = args.filepath + "/_p21c_cache-" + timestamp + "-" + pid
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
    print("created " + cache_dir)

p21c.config['direc'] = cache_dir
cache_tools.clear_cache(direc=cache_dir)
#print("Cache cleared")

## Params to be used if using Mass for Zeta
#user_params = { "HII_DIM": 20, "BOX_LEN": 200, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, 
#               "N_THREADS": 6, "USE_FFTW_WISDOM": True, "USE_RELATIVE_VELOCITIES": True, 
#               "POWER_SPECTRUM": 5} # POWER_SPECTRUM: CLASS
#flag_options = { "USE_MINI_HALOS": True, "M_MIN_in_Mass": True, "USE_MASS_DEPENDENT_ZETA": True, 
#                "INHOMO_RECO": True, "US_TS_FLUCT": True}
user_params = { "HII_DIM": 80, "BOX_LEN": 100, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "USE_FFTW_WISDOM": True}
flag_options = { }

# File for storing Brightness Temperature maps
bt_filename = args.filepath + "/output/bt-" + timestamp + "-" + pid + ".pkl"
print(bt_filename)

# File for storing Power Spectra
ps_filename = args.filepath  + "/output/ps-" + timestamp + "-" + pid + ".pkl"
print(ps_filename)

#zeta_base = 30.0
#zeta_low = zeta_base*0.5  # -50%
#zeta_high = zeta_base*1.5 # +50%
# Following values from paper by chaudhary 2022


# Following values for M_min from paper by chaudhary 2022, they are specified as Mass
# Using these requires change in AstroParams and FlagOptions
#m_min_low = math.log10(1.09e+09)
#m_min_high = math.log10(1.19e+11) 

k_len = -1

logtimestamp = datetime.now().strftime("%H:%M:%S")
print(f'{logtimestamp}: nsets: {args.nsets}, zeta:[{args.zetalow},{args.zetahigh}], M_min:[{args.mminlow},{args.mminhigh}]')

cps = CPS(user_params['HII_DIM'], user_params['BOX_LEN'])
start_time = time.time()
coeval_time = 0
ps_compute_time = 0
bt_write_time = 0
coeval = None
psset = []

for i in range(args.nsets):
    zeta = np.random.uniform(args.zetalow, args.zetahigh)
    m_min = np.random.uniform(args.mminlow, args.mminhigh)
    astro_params = {   
        "HII_EFF_FACTOR": zeta,
        "ION_Tvir_MIN": m_min
    }
    time1 = time.time_ns()
    cache_tools.clear_cache(direc=cache_dir)
    coeval = p21c.run_coeval(redshift=args.redshift, user_params = user_params, astro_params=astro_params, flag_options=flag_options, random_seed=args.randomseed)
    time2 = time.time_ns()
    coeval_time += time2 - time1
    #print(f'Brightness temp shape {coeval.brightness_temp.shape}')
    with open(bt_filename, 'a+b') as f:  # open a text file
        pickle.dump({"zeta": zeta, "m_min": m_min, "bt": coeval.brightness_temp}, f)
    time3 = time.time_ns()
    bt_write_time += time3 - time2

    #ps = p21mc.Likelihood1DPowerCoeval.compute_power(coeval.brightness_temp, L=100, n_psbins = 10) 
    ps, k = cps.compute_power_spectrum_opt(coeval.brightness_temp)
    time4 = time.time_ns()
    ps_compute_time += time4 - time3

    # Data validity - skip invalid records
    if (k_len < 0):
        print(ps)
        k_len = len(ps)
    elif k_len != len(ps):
        print ("Invalid powerspectrum record: skipping...")
        continue
    
    row = {"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}
    psset.append(row)
    with open(ps_filename, 'a+b') as f:  # open a text file
        pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}, f)

    if i%100 == 0: 
        elapsed = time.time() - start_time
        remaining = elapsed * (float(args.nsets)/(i+1)) - elapsed
        logtimestamp = datetime.now().strftime("%H:%M:%S")
        print(f'{logtimestamp}: set#{i+1}, {elapsed}s elapsed, {remaining}s remaining') 
        print(f'coeval_time={coeval_time/1e6}ms , ps_compute_time={ps_compute_time/1e6}ms, bt_write_time={bt_write_time/1e6}ms') 

if args.demo:
    plot(coeval.brightness_temp)
    plot_power_spectra(psset)
print("--- %s seconds ---" % (time.time() - start_time))
