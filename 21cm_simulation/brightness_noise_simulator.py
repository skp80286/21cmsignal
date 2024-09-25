#!/usr/bin/env python

import numpy as np
import collections.abc
import math
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os

import numpy as np
import tools21cm as t2c

import warnings
warnings.filterwarnings("ignore")

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

print(f"Using 21cmFAST version {p21c.__version__}")

if not os.path.exists('_cache'):
    os.mkdir('_cache')
    print("created _cache folder")

p21c.config['direc'] = '_cache'


cache_tools.clear_cache(direc="_cache")
print("Cache cleared")

user_params = { "HII_DIM": 20, "BOX_LEN": 100, "FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "USE_FFTW_WISDOM": True}
flag_options = { } #"USE_MINI_HALOS": True}

filename = datetime.now().strftime("output/ps-%Y%m%d%H%M%S.pkl")
print(filename)


zeta_base = 30.0
zeta_low = zeta_base*0.5  # -50%
zeta_high = zeta_base*1.5 # +50%

m_min_base = math.log10(49999.9995007974)
m_min_low = m_min_base+math.log10(0.5) # -50%
m_min_high = m_min_base+math.log10(1.5) # -50%

z = 9.1
nsets = 1 # number of powerspectra datasets to generate

k_len = -1

start_time = time.time()
for i in range(nsets):
    zeta = np.random.uniform(zeta_low, zeta_high)
    m_min = np.random.uniform(m_min_low, m_min_high)
    astro_params = {   
        "HII_EFF_FACTOR": zeta,
        "ION_Tvir_MIN": m_min
    }
    coeval = p21c.run_coeval(redshift=9.1, user_params = user_params, astro_params=astro_params, flag_options=flag_options)
    dT = coeval.brightness_temp
    ps, k = CPS.compute_power_spectrum(user_params["HII_DIM"], dT, user_params["BOX_LEN"])

    plt.rcParams['figure.figsize'] = [7, 5]

    plt.title('Noiseless power spectrum.')
    plt.loglog(k, ps)
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.show()

    ###
    # simulate 21-cm brightness temperature cube
    ###
    noise_cube = t2c.noise_cube_coeval(ncells=dT.shape[0],
                                    z=z,
                                    depth_mhz=None,   #If None, estimates the value such that the final output is a cube.
                                    obs_time=1000,
                                    filename=None,
                                    boxsize=user_params["BOX_LEN"],
                                    total_int_time=6.0,
                                    int_time=10.0,
                                    declination=-30.0,
                                    uv_map=uv,
                                    N_ant=Nant,
                                    verbose=True,
                                    fft_wrap=False)

    dT = dT + noise_cube

    ps, k = CPS.compute_power_spectrum(user_params["HII_DIM"], coeval.brightness_temp, user_params["BOX_LEN"])

    # Data validity - skip invalid records
    if (k_len < 0):
        print(ps)
        k_len = len(ps)
    elif k_len != len(ps):
        print ("Invalid powerspectrum record: skipping...")
        continue
    
    with open(filename, 'a+b') as f:  # open a text file
        pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}, f)
print("--- %s seconds ---" % (time.time() - start_time))