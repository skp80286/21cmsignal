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

##############
# Utility method for plotting
##############
def plot(dT):
    print ('Plotting signal without and with noise')
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.suptitle('$z=%.2f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
    plt.subplot(121)
    plt.title('noiseless cube slice')
    plt.pcolormesh(dT[:][10][:])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(dT.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

print(f"Using 21cmFAST version {p21c.__version__}")

if not os.path.exists('_cache'):
    os.mkdir('_cache')
    print("created _cache folder")

p21c.config['direc'] = '_cache'


cache_tools.clear_cache(direc="_cache")
print("Cache cleared")

## Params to be used if using Mass for Zeta
#user_params = { "HII_DIM": 20, "BOX_LEN": 200, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, 
#               "N_THREADS": 6, "USE_FFTW_WISDOM": True, "USE_RELATIVE_VELOCITIES": True, 
#               "POWER_SPECTRUM": 5} # POWER_SPECTRUM: CLASS
#flag_options = { "USE_MINI_HALOS": True, "M_MIN_in_Mass": True, "USE_MASS_DEPENDENT_ZETA": True, 
#                "INHOMO_RECO": True, "US_TS_FLUCT": True}
user_params = { "HII_DIM": 80, "BOX_LEN": 100, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "USE_FFTW_WISDOM": True}
flag_options = { }

# File for storing Brightness Temperature maps
bt_filename = datetime.now().strftime("output/bt-%Y%m%d%H%M%S.pkl")
print(bt_filename)

# File for storing Power Spectra
ps_filename = datetime.now().strftime("output/ps-%Y%m%d%H%M%S.pkl")
print(ps_filename)

#zeta_base = 30.0
#zeta_low = zeta_base*0.5  # -50%
#zeta_high = zeta_base*1.5 # +50%
# Following values from paper by chaudhary 2022
zeta_low = 18 
zeta_high = 200

m_min_base = math.log10(49999.9995007974)
m_min_low = m_min_base+math.log10(0.1) # divide by 10
m_min_high = m_min_base+math.log10(10) # multiply by 10

# Following values for M_min from paper by chaudhary 2022, they are specified as Mass
# Using these requires change in AstroParams and FlagOptions
#m_min_low = math.log10(1.09e+09)
#m_min_high = math.log10(1.19e+11) 

z = 9.1
nsets = 20 # number of powerspectra datasets to generate

k_len = -1

timestamp = datetime.now().strftime("%H:%M:%S")
print(f'{timestamp}: nsets: {nsets}, zeta:[{zeta_low},{zeta_high}], M_min:[{m_min_low},{m_min_high}]')

cps = CPS(user_params['HII_DIM'], user_params['BOX_LEN'])
start_time = time.time()
coeval_time = 0
ps_compute_time = 0
bt_write_time = 0
for i in range(nsets):
    zeta = np.random.uniform(zeta_low, zeta_high)
    m_min = np.random.uniform(m_min_low, m_min_high)
    astro_params = {   
        "HII_EFF_FACTOR": zeta,
        "ION_Tvir_MIN": m_min
    }
    time1 = time.time_ns()
    coeval = p21c.run_coeval(redshift=9.1, user_params = user_params, astro_params=astro_params, flag_options=flag_options)
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
    
    with open(ps_filename, 'a+b') as f:  # open a text file
        pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}, f)

    if i%100 == 0: 
        elapsed = time.time() - start_time
        remaining = elapsed * (float(nsets)/(i+1)) - elapsed
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f'{timestamp}: set#{i+1}, {elapsed}s elapsed, {remaining}s remaining') 
        print(f'coeval_time={coeval_time/1e6}ms , ps_compute_time={ps_compute_time/1e6}ms, bt_write_time={bt_write_time/1e6}ms') 
    if (i == 5):
        plot(coeval.brightness_temp)
        print("Printing powerspectrum")
        print(ps)
print("--- %s seconds ---" % (time.time() - start_time))
