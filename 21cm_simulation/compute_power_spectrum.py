import numpy as np
from numpy.fft import fftn

class ComputePowerSpectrum:
    def compute_power_spectrum(nbox, grid, box_size):
        # Total number of points in the grid (cubic)
        N = nbox * nbox * nbox 

        #print(f"ps: N={N}")
        #print(f"ps: nbox={nbox}")
        #print(f"ps: box_size={box_size}")

        kf = 2 * np.pi / box_size  # h/Mpc smallest 'k'
        kn = np.pi * nbox / box_size  # h/Mpc; Nyquist wavenumber or largest 'k'
        coeff = (box_size / (2.0 * np.pi)) ** 2  # Mpc/h FT coefficient
        #print(f"ps: kf={kf}, kn={kn}")

        # Generate wavenumbers. This is similar to fftfreq, but with 
        # slight difference in convention 
        nhalf = nbox // 2
        w = np.zeros(nbox, dtype=np.float64)
        for i in range(nbox):
            iw = i - nbox if i > nhalf else i
            w[i] = kf * iw

        #print(f"ps: w={w}")

        #print(f"myps: grid={grid}")
        # Do the FFT transform with forward normalization and multiply by N to match FFTW convention
        fft_data = fftn(grid, norm='forward') * N
        #print(f"myps: out_data={out_data}")

        # Calculate power spectrum
        powspec = np.zeros(nbox, dtype=np.float64)
        for i in range(nbox):
            for j in range(nbox):
                for k in range(nbox):
                    # consider k as a 3D vector and calculate its magnitude squared
                    g = w[i] ** 2 + w[j] ** 2 + w[k] ** 2

                    if g != 0:
                        # Calculate the index of the 'k' bin in the power spectrum
                        i1 = int(0.5 + np.sqrt(g * coeff))
                        # Calculate the contribution to the power spectrum
                        contrib = (np.abs(fft_data[i,j,k]) ** 2)
                        #if (i==1 and j==1):
                        #    print(f"Contrib: {i} {j} {k} {g} {coeff} {i1} {fft_data[i,j,k]} {contrib}")
                        powspec[i1] += contrib

        #for i in range(nhalf):
        #    print(f"{w[i]}  {powspec[i]}")
        #print("######")
        
        iweights = np.zeros(int(0.5 + np.sqrt(3) * nbox), dtype=np.int64)
        max_ind = 0
        for i in range(nbox):
            i1 = i
            if i1 >= nhalf:
                i1 = nbox - i1 
            for j in range(nbox):
                i2 = j
                if i2 >= nhalf:
                    i2 = nbox - i2 
                for k in range(nbox):
                    i3 = k
                    if i3 >= nhalf:
                        i3 = nbox - i3 
                    m = 0.5 + np.sqrt(i1**2 + i2**2 + i3**2)
                    iweights[int(m)] += 1
                    if (int(m) > max_ind):
                        max_ind = int(m)
        #print(f"ps: iweights max_ind={max_ind}")

        # ----------------------
        # Calculate the DeltaSquaredK power spectrum
        deltasqk = np.zeros(nbox, dtype=np.float64)
        
        for i in range(nhalf):
            powspec[i] = powspec[i] * (box_size ** 3.0) / (nbox ** 6.0) 
            powspec[i] /= float(iweights[i])
            deltasqk[i] = (w[i] ** 3.0) * powspec[i] / (2.0 * np.pi * np.pi)
        
        #for i in range(nhalf):
        #    print(f"{w[i]}  {powspec[i]}  {deltasqk[i]}  {iweights[i]}")
        return (deltasqk, w)

    def compute_power_spectrum_opt(nbox, grid, box_size):
        # Total number of points in the grid (cubic)
        N = nbox * nbox * nbox 

        #print(f"ps: N={N}")
        #print(f"ps: nbox={nbox}")
        #print(f"ps: box_size={box_size}")

        kf = 2 * np.pi / box_size  # h/Mpc smallest 'k'
        kn = np.pi * nbox / box_size  # h/Mpc; Nyquist wavenumber or largest 'k'
        coeff = (box_size / (2.0 * np.pi)) ** 2  # Mpc/h FT coefficient
        #print(f"ps: kf={kf}, kn={kn}")

        # Generate wavenumbers. This is similar to fftfreq, but with 
        # slight difference in convention 
        nhalf = nbox // 2
        w = np.zeros(nbox, dtype=np.float64)
        for i in range(nbox):
            iw = i - nbox if i > nhalf else i
            w[i] = kf * iw

        #print(f"ps: w={w}")

        #print(f"myps: grid={grid}")
        # Do the FFT transform with forward normalization and multiply by N to match FFTW convention
        fft_data = fftn(grid, norm='forward') * N
        #print(f"myps: out_data={out_data}")

        # Calculate power spectrum
        powspec = np.zeros(nbox, dtype=np.float64)
        for i in range(nbox):
            for j in range(nbox):
                for k in range(nbox):
                    # consider k as a 3D vector and calculate its magnitude squared
                    g = w[i] ** 2 + w[j] ** 2 + w[k] ** 2

                    if g != 0:
                        # Calculate the index of the 'k' bin in the power spectrum
                        i1 = int(0.5 + np.sqrt(g * coeff))
                        # Calculate the contribution to the power spectrum
                        contrib = (np.abs(fft_data[i,j,k]) ** 2)
                        #if (i==1 and j==1):
                        #    print(f"Contrib: {i} {j} {k} {g} {coeff} {i1} {fft_data[i,j,k]} {contrib}")
                        powspec[i1] += contrib

        return (powspec, w)
