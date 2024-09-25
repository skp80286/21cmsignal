#! /usr/bin/env python3

import tools21cm as t2c
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from datetime import datetime
import pickle

###
# Constants
###
z=9.1
user_params = { "HII_DIM": 20, "BOX_LEN": 100 }
ncells = user_params["HII_DIM"]
dx, dy = (user_params["BOX_LEN"]/ncells, user_params["BOX_LEN"]/ncells)
y, x = np.mgrid[slice(dy/2, user_params["BOX_LEN"], dy),
                slice(dx/2, user_params["BOX_LEN"], dx)]
now = datetime.now()
noise_filename = now.strftime("output/noise-%Y%m%d%H%M%S-__PH__")
bt_filename = 'output/bt-20240911162753.pkl'
bt_noise_filename = now.strftime("output/bt-noise-%Y%m%d%H%M%S-__PH__")
print(noise_filename)
print(bt_filename)
print(bt_noise_filename)
###
# calculate the uv-coverage for SKA1-Low configuration
###

uv, Nant = t2c.get_uv_daily_observation(ncells=ncells, # The number of cell used to make the image
                                        z=z,                # Redhsift of the slice observed
                                        filename=None,      # If None, it uses the SKA-Low 2016 configuration.
                                        total_int_time=6.0, # Observation per day in hours.
                                        int_time=10.0,      # Time period of recording the data in seconds.
                                        boxsize=user_params["BOX_LEN"],    # Comoving size of the sky observed
                                        declination=-30.0,
                                        verbose=True)

#np.save(noise_filename.replace('__PH__', 'uv_map.npy'), uv)
#np.save(noise_filename.replace('__PH__', 'Nant.npy'), Nant)

###
# Plot the uv map
###
plt.rcParams['figure.figsize'] = [5, 5]

plt.title(r'$z=%.3f$ $\nu_{obs}=%.2f$ MHz' %(z, t2c.z_to_nu(z)))
plt.pcolormesh(x, y, np.log10(np.fft.fftshift(uv)))
plt.xlabel('u [$Mpc^-1$]'), plt.ylabel('v [$Mpc^-1$]')
plt.colorbar()
plt.show()

###
# plot the location of SKA-Low antennas
###

ska_ant = t2c.SKA1_LowConfig_Sept2016()

fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(ska_ant[:,0], ska_ant[:,1], '.')
x1, x2, y1, y2 = 116.2, 117.3, -26.45, -27.25
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
#ax.grid(b=True, alpha=0.5)

axins = inset_axes(ax, 1, 1, loc=4, bbox_to_anchor=(0.2, 0.2))
plt.plot(ska_ant[:,0], ska_ant[:,1], ',')
x1, x2, y1, y2 = 116.75, 116.78, -26.815, -26.833
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#axins.grid(b=True, alpha=0.5)
mark_inset(ax, axins, loc1=4, loc2=2, fc="none", ec="0.5")
axins.axes.xaxis.set_ticks([]);
axins.axes.yaxis.set_ticks([]);
plt.show()

###
# simulate 21-cm noise cube
###
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
                                    verbose=True,
                                    fft_wrap=False)

plt.rcParams['figure.figsize'] = [16, 6]

plt.suptitle('$z=%.3f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
plt.subplot(121)
plt.title('noise cube slice')
plt.pcolormesh(x, y, noise_cube[0])
plt.colorbar(label='$\delta T^{noise}$ [mK]')
plt.subplot(122)
plt.title('noise distribution')
plt.hist(noise_cube.flatten(), bins=150, histtype='step');
plt.xlabel('$\delta T^{noise}$ [mK]'), plt.ylabel('$N_{sample}$');
plt.show()

###
# Load the noiseless brightness temp cube
###
lines = 0
dT, e = (None, None)
with open(bt_filename, 'rb') as input_file:  # open a text file
    while True:
        try:
            e = pickle.load(input_file)
            dT = e["bt"]
            print(f"dT.shape={dT.shape}, noise_cube.shape={noise_cube.shape}")
            e["bt"] = dT + noise_cube
            with open(bt_noise_filename, 'ab') as output_file:  # open a binary file for appending
                pickle.dump(e, output_file)
            lines += 1
        except EOFError:
             break
    print("--- read %d lines ---" % lines)
if lines ==1:
    print ('Plotting signal without and with noise')
    plt.rcParams['figure.figsize'] = [16, 6]
    plt.suptitle('$z=%.3f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
    plt.subplot(121)
    plt.title('noiseless cube slice')
    plt.pcolormesh(x, y, dT[0])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(dT.flatten(), bins=150, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

    plt.rcParams['figure.figsize'] = [16, 6]
    plt.suptitle('$z=%.3f$ $x_v=%.3f$' %(z, 0), size=18) # xfrac.mean()
    plt.subplot(121)
    plt.title('noisy cube slice')
    plt.pcolormesh(x, y, e["bt"][0])
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title('signal distribution')
    plt.hist(e["bt"].flatten(), bins=150, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()


