#!python

import numpy as np
import collections.abc
#py21cmmc needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#py21cmmc needs the below
np.int = np.int32
#Now import py21cmmc
from py21cmmc import analyse
from py21cmmc import mcmc
import py21cmmc as p21mc
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

print(f"Using 21cmFAST version {p21c.__version__}")

if not os.path.exists('_cache'):
    os.mkdir('_cache')
    print("created _cache folder")

p21c.config['direc'] = '_cache'


cache_tools.clear_cache(direc="_cache")
print("Cache cleared")

user_params = {"FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "USE_FFTW_WISDOM": True}
flag_options = { "USE_MINI_HALOS": True}

filename = datetime.now().strftime("output/ps-%Y%m%d%H%M%S")
print(filename)

zeta_base = 30.0
zeta_low = zeta_base*0.8  # -20%
zeta_high = zeta_base*1.2 # +20%

m_min_base = math.log10(49999.9995007974)
m_min_low = m_min_base+math.log10(0.8) # -20%
m_min_high = m_min_base+math.log10(1.2) # -20%

z = 9.1
nsets = 1000 # number of powerspectra datasets to generate

user_params = {"FAST_FCOLL_TABLES": True, "USE_INTERPOLATION_TABLES": True, "N_THREADS": 6, "USE_FFTW_WISDOM": True}
flag_options = {}# "USE_MINI_HALOS": True}

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
    ps = p21mc.Likelihood1DPowerCoeval.compute_power(coeval.brightness_temp, L=100, n_psbins = 10) 
    

    # Data validity - skip invalid records
    if (k_len < 0):
        print(ps)
        k_len = len(ps[0])
    elif k_len != len(ps[0]):
        print ("Invalid powerspectrum record: skipping...")
        continue
    ####
    
    with open(filename, 'ab') as f:  # open a text file
        pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps[0]}, f)
print("--- %s seconds ---" % (time.time() - start_time))
