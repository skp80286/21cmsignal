#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants
from astropy import units as un
from astropy.cosmology import Planck15
from astropy.cosmology.units import littleh

import py21cmsense as p21cs
import pickle

###
# Load the noiseless power spectrum data
###

def load_noiseless_ps(filename, plot = False):
    """
    Load the noiseless power spectrum data from a pickle file.

    Parameters:
        filename (str): The path to the pickle file containing the power spectrum data.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    if plot:
        row = data[0]
        plt.plot(row['k'], row['ps'])
        plt.show()
    return data

###
# Simulate the noise power spectrum
###

def simulate_noise_ps(noiseless_ps, noise_level):
    """
    Simulate the noise power spectrum by adding white noise to the noiseless power spectrum.
    """

    return noise_ps

load_noiseless_ps('../21cm_simulation/output/ps-20240908194837.pkl', plot = True)  

