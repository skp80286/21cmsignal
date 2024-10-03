# %% [markdown]
# # Reading and visualising data

# %%
import numpy as np
import tools21cm as t2c
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot(data, caption):
    plt.rcParams['figure.figsize'] = [15, 6]
    plt.subplot(121)
    plt.title(caption)
    plt.pcolormesh(data)
    plt.colorbar(label='$\delta T^{signal}$ [mK]')
    plt.subplot(122)
    plt.title(caption + ' distribution')
    plt.hist(data.flatten(), bins=149, histtype='step')
    plt.xlabel('$\delta T^{signal}$ [mK]'), plt.ylabel('$S_{sample}$')
    plt.show()

path_to_datafiles = './data/'
z=9.1
ncells = 80
user_params={'BOX_LEN': 100}
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

    np.save(uv_store, uv)
    np.save(Nant_store, Nant)

noise = t2c.noise_cube_coeval(ncells=ncells,
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
plot(noise[:][10][:], "Noise")

fg = t2c.foreground_model.galactic_synch_fg(z=z, ncells=ncells, boxsize=user_params['BOX_LEN'])
plot(fg, "Foreground")
