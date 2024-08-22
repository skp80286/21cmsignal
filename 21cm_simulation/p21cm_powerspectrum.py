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

import math

print(f"Using 21cmFAST version {p21c.__version__}")

if not os.path.exists('_cache'):
    os.mkdir('_cache')
    print("created _cache folder")

p21c.config['direc'] = '_cache'
# cache_tools.clear_cache(direc="_cache")
# print("Cache cleared")

BOX_LEN = 100
HII_DIM = 100
# Define global py21cmFAST parameters
user_params = {"HII_DIM": HII_DIM, "BOX_LEN": BOX_LEN, "USE_INTERPOLATION_TABLES": False}

# define functions to calculate PS, following py21cmmc
from powerbox.tools import get_power
import numpy as np

def compute_power(
   box,
   length,
   n_psbins,
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
):
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    res = get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k
    return res

def powerspectra(brightness_temp, n_psbins=50, nchunks=10, min_k=0.1, max_k=1.0, logk=False):
    data = []
    chunk_indices = list(range(0,brightness_temp.n_slices,round(brightness_temp.n_slices / nchunks),))

    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(brightness_temp.n_slices)

    for i in range(nchunks):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * brightness_temp.cell_size

        power, k = compute_power(
            brightness_temp.brightness_temp[:, :, start:end],
            (BOX_LEN, BOX_LEN, chunklen),
            n_psbins,
            log_bins=logk,
        )

        # Filter values outside min and max k range specified
        #print (k)
        filter = [(x >= min_k and x <= max_k) for x in k]
        #print(filter)
        k = k[filter]
        #print (k)
        power = power[filter]

        # Filter invalid values of power
        filter = [not(math.isnan(x) or math.isinf(x)) for x in power]
        k = k[filter]
        power = power[filter]

        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    return data
    
def powerspectra1(brightness_temp, n_psbins=50, min_k=0.1, max_k=1.0, logk=False):
    data = []
    power, k = compute_power(
            brightness_temp,
            (BOX_LEN, BOX_LEN, 1),
            n_psbins,
            log_bins=logk,
        )
        # Filter values outside min and max k range specified
        #print (k)
        filter = [(x >= min_k and x <= max_k) for x in k]
        #print(filter)
        k = k[filter]
        #print (k)
        power = power[filter]

        # Filter invalid values of power
        filter = [not(math.isnan(x) or math.isinf(x)) for x in power]
        k = k[filter]
        power = power[filter]
    data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    return data

def get_spectra(z):
    lightcone = p21c.run_lightcone(redshift = (z - 0.01), max_redshift= (z + 0.01))
    return powerspectra(lightcone, n_psbins = [], nchunks = 1, min_k =  0.5, max_k = 10, logk = True) 