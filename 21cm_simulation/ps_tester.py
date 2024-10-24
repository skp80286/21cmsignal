import numpy as np
import matplotlib.pyplot as plt
from compute_power_spectrum import ComputePowerSpectrum as CPS

import numpy as np
import matplotlib.pyplot as plt
from compute_power_spectrum import ComputePowerSpectrum as CPS
import argparse
import pickle

parser = argparse.ArgumentParser(description='Simulate 21cm cosmological signal with noise and foreground.')
parser.add_argument('-s', '--sliceindex', type=int, default=40, help='Slice index to plot. Used in demo mode.')
parser.add_argument('-f', '--psfilename', type=str, default="./saved_output/ps-80-7000.pkl", help='Filename to read power sprectrum to read from.')
parser.add_argument('-n', '--numrows', type=int, default=100, help='number of rows to plot from powersectrum file.')
args = parser.parse_args()

def create_gaussian_noise_cube(shape, mean=0, std=1):
    """
    Create a cube of Gaussian noise.
    
    Args:
    shape (tuple): The shape of the cube (e.g., (80, 80, 80))
    mean (float): The mean of the Gaussian distribution (default: 0)
    std (float): The standard deviation of the Gaussian distribution (default: 1)
    
    Returns:
    np.ndarray: A cube of Gaussian noise with the specified shape
    """
    return np.random.normal(mean, std, shape)

##############
# Utility method for plotting
##############
def plot_cube_slice(dT, title):
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.subplot(121)
    plt.title(title + ' cube slice')
    plt.pcolormesh(dT[:][args.sliceindex][:])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title(title + ' signal distribution')
    plt.hist(dT.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

def plot_power_spectrum(ps, ks, title):
    print(ks)
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.title(title + ' Spherically averaged power spectrum.')
    plt.loglog(ks[1:40], ps[1:40])
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.ylim(bottom=1)
    plt.show()

def plot_multiple_ps(ks, ps, zetas, mmins):
    #print(ks[:10])
    print(ps[:10])
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.title('Spherically averaged power spectrum.')
    for k, p, zeta, mmin in zip(ks, ps, zetas, mmins):
        if p[1] > 0 and p[1] < 1.5e-7:
            plt.loglog(k[1:41], p[1:41], label=f'Zeta:{zeta:.2f}-M_min:{mmin:.2f}')
        else:
            plt.loglog(k[1:41], p[1:41])

    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.legend(loc='lower right')
    plt.show()

"""
# Initialize powerspectrum computation
cps = CPS(80, 100)
noise_cube = create_gaussian_noise_cube((80, 80, 80), mean=0, std=1)
plot_cube_slice(noise_cube, "Gaussian")
ps, k = cps.compute_power_spectrum_unbinned(grid=noise_cube)
plot_power_spectrum(ps, k, "Gaussian unbinned")

"""
psset=[]
kset=[]
zetaset=[]
mminset=[]
with open(args.psfilename, 'rb') as input_file:  # open a text file
    for i in range(args.numrows): # Find a raw where the parameters are near the default values and not outliers
        e = pickle.load(input_file)
        psset.append(e["ps"])
        kset.append(e["k"])
        zetaset.append(e["zeta"])
        mminset.append(e["m_min"])

plot_multiple_ps(kset, psset, zetaset, mminset)