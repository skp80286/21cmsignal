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

##############
# Utility method for plotting
##############
def plot(x, y, dT1, dT2, ps, ks, ps1, ks1):
    print ('Plotting signal without and with noise')
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.suptitle('$z=%.2f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
    plt.subplot(121)
    plt.title('noiseless cube slice')
    plt.pcolormesh(dT1[:][10][:])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(dT1.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

    plt.rcParams['figure.figsize'] = [15, 6]
    plt.suptitle('$z=%.2f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
    plt.subplot(121)
    plt.title('noisy cube slice')
    plt.pcolormesh(dT2[:][10][:])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(dT2.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

    plt.rcParams['figure.figsize'] = [15, 6]

    plt.subplot(121)
    plt.title('Noiseless Spherically averaged power spectrum.')
    plt.loglog(ks, ps*ks**3/2/np.pi**2)
    #plt.xlim(right = 2)
    plt.ylim(0.6, 40)
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')

    plt.subplot(122)
    plt.title('With-Noise Spherically averaged power spectrum.')
    plt.loglog(ks1, ps1*ks1**3/2/np.pi**2)
    #plt.xlim(right = 2)
    plt.ylim(0.6, 40)
    plt.xlabel('k (Mpc$^{-1}$)')
    plt.ylabel('P(k) k$^{3}$/$(2\pi^2)$')
    plt.show()


###
# Constants
###
z=9.1
path_to_datafiles = './data/'
user_params = { "HII_DIM": 80, "BOX_LEN": 100 }
ncells = user_params["HII_DIM"]
dx, dy = (user_params["BOX_LEN"]/ncells, user_params["BOX_LEN"]/ncells)
y, x = np.mgrid[slice(dy/2, user_params["BOX_LEN"], dy),
                slice(dx/2, user_params["BOX_LEN"], dx)]
now = datetime.now()
noise_filename = now.strftime("output/noise-%Y%m%d%H%M%S-__PH__")
bt_filename = 'output/bt-80-7000.pkl'
#bt_filename = 'output/bt-20240925113738.pkl'
bt_noise_filename = now.strftime("output/bt-noise-%Y%m%d%H%M%S.pkl")
ps_noise_filename = now.strftime("output/ps-noise-%Y%m%d%H%M%S.pkl")
print(noise_filename)
print(bt_filename)
print(bt_noise_filename)
###
# calculate the uv-coverage for SKA1-Low configuration
###

#uv, Nant = t2c.get_uv_daily_observation(ncells=ncells, # The number of cell used to make the image
#                                        z=z,                # Redhsift of the slice observed
#                                        filename=None,      # If None, it uses the SKA-Low 2016 configuration.
#                                        total_int_time=6.0, # Observation per day in hours.
#                                        int_time=10.0,      # Time period of recording the data in seconds.
#                                        boxsize=user_params["BOX_LEN"],    # Comoving size of the sky observed
#                                        declination=-30.0,
#                                        verbose=True)

#np.save(noise_filename.replace('__PH__', 'uv_map.npy'), uv)
#np.save(noise_filename.replace('__PH__', 'Nant.npy'), Nant)

###
# Plot the uv map
###
#plt.rcParams['figure.figsize'] = [5, 5]

#plt.title(r'$z=%.3f$ $\nu_{obs}=%.2f$ MHz' %(z, t2c.z_to_nu(z)))
#plt.pcolormesh(x, y, np.log10(np.fft.fftshift(uv)))
#plt.xlabel('u [$Mpc^-1$]'), plt.ylabel('v [$Mpc^-1$]')
#plt.colorbar()
#plt.show()

###
# plot the location of SKA-Low antennas
###

#ska_ant = t2c.SKA1_LowConfig_Sept2016()

#fig, ax = plt.subplots(figsize=(5, 5))
#plt.plot(ska_ant[:,0], ska_ant[:,1], '.')
#x1, x2, y1, y2 = 116.2, 117.3, -26.45, -27.25
#ax.set_xlim(x1, x2)
#ax.set_ylim(y1, y2)
#ax.grid(b=True, alpha=0.5)

#axins = inset_axes(ax, 1, 1, loc=4, bbox_to_anchor=(0.2, 0.2))
#plt.plot(ska_ant[:,0], ska_ant[:,1], ',')
#x1, x2, y1, y2 = 116.75, 116.78, -26.815, -26.833
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
##axins.grid(b=True, alpha=0.5)
#mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="0.5")
#axins.axes.xaxis.set_ticks([]);
#axins.axes.yaxis.set_ticks([]);
#plt.show()

###
# simulate 21-cm noise cube
###
# noise_cube = np.random.normal(0,1000,(80,80,80))
uv, Nant = None, None
uv_store = f'{path_to_datafiles}uv-{z}-{ncells}-{user_params["BOX_LEN"]}.npy'
Nant_store = f'{path_to_datafiles}Nant-{z}-{ncells}-{user_params["BOX_LEN"]}.npy'
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

# %% [markdown]
# We suggest that you save the uv map as it is computationally expensive. Expecially when computed for an array of redshifts.

# %%
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

#plt.rcParams['figure.figsize'] = [16, 6]

#plt.suptitle('$z=%.3f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
#plt.subplot(121)
#plt.title('noise cube slice')
#plt.pcolormesh(noise_cube[0])
#plt.colorbar(label='$\delta T^{noise}$ [mK]')
#plt.subplot(122)
#plt.title('noise distribution')
#plt.hist(noise_cube.flatten(), bins=150, histtype='step');
#plt.xlabel('$\delta T^{noise}$ [mK]'), plt.ylabel('$N_{sample}$');
#plt.show()

###
# Load the noiseless brightness temp cube
###
lines = 0
k_len = -1
dT, e = (None, None)
start_time = time.time()
smooth_time = 0
ps_compute_time = 0
cps = CPS(user_params['HII_DIM'], user_params['BOX_LEN'])

with open(bt_filename, 'rb') as input_file:  # open a text file
    while True:
        if lines%100 == 1: 
            elapsed = time.time() - start_time
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f'{timestamp}: line#{lines}, {elapsed}s elapsed, {elapsed/(lines)}s per line') 
            print(f'{smooth_time/1e6}ms smooth_time, {ps_compute_time/1e6}ms ps_comute_time') 
        try:
            e = pickle.load(input_file)
            dT = e["bt"]
            zeta = e["zeta"]
            m_min = e["m_min"]
            #print(f"dT.shape={dT.shape}, noise_cube.shape={noise_cube.shape}")
            #print("Printing noiseless cube slice:")
            #print(dT[:][0][:])
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT[:][10][:].flatten().mean(), dT[:][10][:].flatten().min(), dT[:][10][:].flatten().max()))
            #ps, k = CPS.compute_power_spectrum(user_params["HII_DIM"], dT, user_params["BOX_LEN"])

            dT = t2c.subtract_mean_signal(dT, 0)
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT[:][10][:].flatten().mean(), dT[:][10][:].flatten().min(), dT[:][10][:].flatten().max()))

            #print('Mean of first noise channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(noise_cube[:][10][:].flatten().mean(), noise_cube[:][10][:].flatten().min(), noise_cube[:][10][:].flatten().max()))
            dT_noise = dT + noise_cube
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT_noise[:][10][:].flatten().mean(), dT_noise[:][10][:].flatten().min(), dT_noise[:][10][:].flatten().max()))
            
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
            #print('Mean of first channel: {0:.10f}, {1:.10f}, {2:.10f}'.format(dT_noise[:][10][:].flatten().mean(), dT_noise[:][10][:].flatten().min(), dT_noise[:][10][:].flatten().max()))

            #with open(bt_noise_filename, 'ab') as output_file:  # open a binary file for appending
            #    pickle.dump({"zeta": zeta, "m_min": m_min, "bt": dT_noise}, output_file)
            ps, k = cps.compute_power_spectrum_opt(dT_noise)
            time3 = time.time_ns()
            ps_compute_time += (time3 - time2)
            #print('Printing powerspectrum:')
            #print(ps)

            # Data validity - skip invalid records
            if (k_len < 0):
                k_len = len(ps)
            elif k_len != len(ps):
                print ("Invalid powerspectrum record: skipping...")
                continue
            
            with open(ps_noise_filename, 'a+b') as f:  # open a text file
                pickle.dump({"zeta": zeta, "m_min": m_min, "ps": ps, "k": k}, f)
            
            lines += 1
            #if(lines == 3): plot(x, y, dT, dT_noise, ps, k, ps1, k1)
            #if (lines == 10): break # artificial limit for testing
        except EOFError:
            break

print("--- processed %d lines ---" % lines)
