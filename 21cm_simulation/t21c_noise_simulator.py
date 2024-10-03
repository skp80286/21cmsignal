#! /usr/bin/env python3

import tools21cm as t2c
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from datetime import datetime
import time
import pickle
from compute_power_spectrum import ComputePowerSpectrum as CPS
import argparse

parser = argparse.ArgumentParser(description='Simulate 21cm cosmological signal with noise and foreground.')
parser.add_argument('-f', '--foreground', action='store_true', help='Add Galactic Synchrotron foreground')
parser.add_argument('-d', '--demo', action='store_true', help='Run in demo mode, single row with plots and informational screen output')
parser.add_argument('-n', '--nsets', type=int, default=100000, help='Limit processing to specified number of rows')
parser.add_argument('-s', '--sliceindex', type=int, default=10, help='Slice index to plot. Used in demo mode.')
parser.add_argument('-r', '--rowindex', type=int, default=18, help='Row index to plot. Used in demo mode.')
parser.add_argument('-p', '--datapath', type=str, default='./data/', help='Path to data files')
parser.add_argument('-z', '--redshift', type=float, default=9.1, help='redshift')
parser.add_argument('-c', '--cells', type=int, default=80, help='number of cells, each side of cube')
parser.add_argument('-l', '--length', type=int, default=100, help='length of each side of cube in Mpc')

args = parser.parse_args()

if args.demo:
    print("### Demo Mode ###")
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

def plot_power_spectrum(ps1, ks1, ps2, ks2, title1, title2):
    print(ks1)
    print(ks2)
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.subplot(121)
    plt.title(title1 + ' Spherically averaged power spectrum.')
    plt.loglog(ks1, ps1)
    #plt.xlim(right = 2)
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')

    plt.subplot(122)
    plt.title(title2 + ' Spherically averaged power spectrum.')
    plt.loglog(ks2, ps2)
    #plt.xlim(right = 2)
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.show()


###
# Constants
###
z=9.1
user_params = { "HII_DIM": args.cells, "BOX_LEN": args.length }
ncells = user_params["HII_DIM"]
# Create grid useful for plotting
dx, dy = (user_params["BOX_LEN"]/ncells, user_params["BOX_LEN"]/ncells)
y, x = np.mgrid[slice(dy/2, user_params["BOX_LEN"], dy),
                slice(dx/2, user_params["BOX_LEN"], dx)]

# Input and output filenames
now = datetime.now()
noise_filename = now.strftime("output/noise-%Y%m%d%H%M%S-__PH__")
bt_filename = 'output/bt-80-7000.pkl'
ps_filename = 'output/ps-80-7000.pkl'
#bt_filename = 'output/bt-20240925113738.pkl'
bt_noise_filename = now.strftime("output/bt-noise-%Y%m%d%H%M%S.pkl")
ps_noise_filename = now.strftime("output/ps-noise-%Y%m%d%H%M%S.pkl")
print(noise_filename)
print(bt_filename)
print(bt_noise_filename)
###
# simulate 21-cm noise cube
###
# noise_cube = np.random.normal(0,1000,(80,80,80)) # random Gaussian noise for testing
uv, Nant = None, None
# We try to avoid uv map generation by storing it in the data directory
uv_store = f'{args.datapath}uv-{z}-{ncells}-{user_params["BOX_LEN"]}.npy'
Nant_store = f'{args.datapath}Nant-{z}-{ncells}-{user_params["BOX_LEN"]}.npy'
try:
    uv = np.load(uv_store)
    Nant = np.load(Nant_store)
except OSError:
    print('Computing UV map')
    uv, Nant = t2c.get_uv_daily_observation(ncells=ncells, # The number of cell used to make the image
                                        z=z,                # Redhsift of the slice observed
                                        filename=None,      # If None, it uses the SKA-Low 2016 configuration.
                                        total_int_time=6.0, # Observation per day in hours.
                                        int_time=10.0,      # Time period of recording the data in seconds.
                                        boxsize=user_params["BOX_LEN"],    # Comoving size of the sky observed
                                        declination=-30.0,  # SKA-Low setting
                                        verbose=False)

    np.save(uv_store, uv)
    np.save(Nant_store, Nant)

noise_cube = t2c.noise_cube_coeval(ncells=ncells,
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
                                    verbose=False,
                                    fft_wrap=False)

# Generate mock Galactic foreground and store it as a cube.
if args.foreground:
    fg_2d = t2c.foreground_model.galactic_synch_fg(z=z, ncells=ncells, boxsize=user_params['BOX_LEN'])
    fg_3d = np.zeros((ncells, ncells, ncells), dtype=float)
    for k in range(ncells):
        fg_3d[k] = fg_2d 

###
# Load the noiseless brightness temp cube
###
lines = 0
k_len = -1
dT, e = (None, None)
start_time = time.time()
smooth_time = 0
ps_compute_time = 0

# Initialize powerspectrum computation
cps = CPS(user_params['HII_DIM'], user_params['BOX_LEN'])

if args.demo: # skip some lines and load a ps to plot
    with open(ps_filename, 'rb') as input_file:  # open a text file
        for i in range(args.rowindex):
            e = pickle.load(input_file)
        e = pickle.load(input_file)
        ps_orig = e["ps"]
        k_orig = e["k"]

# Read Brightness Temp cubes from file one-by-one and process it
with open(bt_filename, 'rb') as input_file:  # open a text file
    while True:
        if args.demo: # skip a few lines
            for i in range(args.rowindex):
                e = pickle.load(input_file)

        if lines%100 == 1: # Useful logging for monitoring a longrunning computation
            elapsed = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f'{timestamp}: line#{lines}, {elapsed}s elapsed, {elapsed/(lines)}s per line') 
            print(f'{smooth_time/1e6}ms smooth_time, {ps_compute_time/1e6}ms ps_comute_time') 
        try:
            e = pickle.load(input_file)
            dT = e["bt"]
            zeta = e["zeta"]
            m_min = e["m_min"]
            if args.demo: 
                print(f"dT.shape={dT.shape}, noise_cube.shape={noise_cube.shape}")
                print("Printing noiseless cube slice:")
                print(dT[:][args.sliceindex][:])
                print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT[:][args.sliceindex][:].flatten().mean(), dT[:][args.sliceindex][:].flatten().min(), dT[:][args.sliceindex][:].flatten().max()))
                plot_cube_slice(dT, "Noiseless")
                ps_noiseless, k_noiseless = CPS.compute_power_spectrum(user_params["HII_DIM"], dT, user_params["BOX_LEN"])

            dT = t2c.subtract_mean_signal(dT, 0)
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT[:][10][:].flatten().mean(), dT[:][10][:].flatten().min(), dT[:][10][:].flatten().max()))

            #print('Mean of first noise channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(noise_cube[:][10][:].flatten().mean(), noise_cube[:][10][:].flatten().min(), noise_cube[:][10][:].flatten().max()))
            if args.demo: plot_cube_slice(noise_cube, "Noise")
            dT_noise = dT + noise_cube
            if args.demo: plot_cube_slice(dT_noise, "Signal with noise")
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT_noise[:][10][:].flatten().mean(), dT_noise[:][10][:].flatten().min(), dT_noise[:][10][:].flatten().max()))
            if args.foreground:
                dT_noise = noise_cube + fg_3d
                if args.demo: plot_cube_slice(dT_noise, "Signal with noise and foreground")
        
            time1 = time.time_ns() 
            dT_noise = t2c.smooth_coeval(cube=dT_noise,    # Data cube that is to be smoothed
                              z=z,                  # Redshift of the coeval cube
                              box_size_mpc=user_params["BOX_LEN"], # Box size in cMpc
                              max_baseline=2.0,     # Maximum baseline of the telescope
                              ratio=1.0,            # Ratio of smoothing scale in frequency direction
                              nu_axis=2, 
                              verbose=False)            # frequency axis
            time2 = time.time_ns()
            smooth_time += (time2 - time1)
            if args.demo: plot_cube_slice(dT_noise, "Smoothed signal with noise")
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT_noise[:][10][:].flatten().mean(), dT_noise[:][10][:].flatten().min(), dT_noise[:][10][:].flatten().max()))

            #with open(bt_noise_filename, 'ab') as output_file:  # open a binary file for appending
            #    pickle.dump({"zeta": zeta, "m_min": m_min, "bt": dT_noise}, output_file)
            ps, k = cps.compute_power_spectrum_opt(dT_noise)
            time3 = time.time_ns()
            ps_compute_time += (time3 - time2)
            #print('Printing powerspectrum:')
            #print(ps)
            if args.demo: 
                plot_power_spectrum(ps_orig, k_orig, ps_noiseless, k_noiseless, "Original", "Noiseless")
                plot_power_spectrum(ps_noiseless, k_noiseless, ps, k, "Noiseless", "With noise")

            # Data validity - skip invalid records
            if (k_len < 0):
                k_len = len(ps)
            elif k_len != len(ps):
                print ("Invalid powerspectrum record: skipping...")
                continue
            
            # Store the noisy powerspectrum
            with open(ps_noise_filename, 'a+b') as f:  # open a text file
                pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}, f)
            
            lines += 1
            #if(lines == 3): plot(x, y, dT, dT_noise, ps, k, ps1, k1)
            #if (lines == 10): break # artificial limit for testing
        except EOFError:
            break
        if args.demo or lines >= args.nsets: break 

print("--- processed %d lines ---" % lines)
