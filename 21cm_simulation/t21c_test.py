# %% [markdown]
# # Reading and visualising data

# %%
import numpy as np
import tools21cm as t21c
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

noise = t21c.noise_model.noise_cube_coeval(80, 9.1, verbose = True)
plot(noise[:][10][:], "Noise")

fg = t21c.foreground_model.galactic_synch_fg(z=9.1, ncells=80, boxsize=100)
plot(fg, "Foreground")
