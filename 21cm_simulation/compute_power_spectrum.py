import numpy as np
from numpy.fft import fftn

class ComputePowerSpectrum:
    def __init__(self, nbox, box_size):
        self.nbox = nbox
        self.box_size = box_size
        
        # Pre-calculate values that depend on nbox and box_size
        self.N = nbox * nbox * nbox
        self.kf = 2 * np.pi / box_size
        self.coeff = (box_size / (2.0 * np.pi)) ** 2
        self.powspec_coeff = (self.box_size ** 3.0) / (self.nbox ** 6.0) 
        self.nhalf = nbox // 2
        # Generate wavenumbers
        self.w = np.zeros(nbox, dtype=np.float64)
        for i in range(nbox):
            iw = i - nbox if i > self.nhalf else i
            self.w[i] = self.kf * iw
        self.w_cube = (self.w ** 3) / (2.0 * np.pi * np.pi)

        self.radial = np.zeros((nbox, nbox, nbox), dtype=np.int32)
        for i in range(self.nbox):
            for j in range(self.nbox):
                for k in range(self.nbox):
                    # consider k as a 3D vector and calculate its magnitude squared
                    g = self.w[i] ** 2 + self.w[j] ** 2 + self.w[k] ** 2
                    if g != 0: self.radial[i][j][k] = int(0.5 + np.sqrt(g * self.coeff))

        self.iweights = np.zeros(int(0.5 + np.sqrt(3) * self.nbox), dtype=np.int64)
        for i in range(nbox):
            i1 = i
            if i1 >= self.nhalf:
                i1 = self.nbox - i1 
            for j in range(self.nbox):
                i2 = j
                if i2 >= self.nhalf:
                    i2 = self.nbox - i2 
                for k in range(self.nbox):
                    i3 = k
                    if i3 >= self.nhalf:
                        i3 = self.nbox - i3 
                    m = 0.5 + np.sqrt(i1**2 + i2**2 + i3**2)
                    self.iweights[int(m)] += 1

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

    def compute_power_spectrum_opt(self, grid):
        # Do the FFT transform with forward normalization and multiply by N to match FFTW convention
        fft_data = fftn(grid, norm='forward') * self.N

        # Calculate power spectrum
        powspec = np.zeros(self.nbox, dtype=np.float64)
        for i in range(self.nbox):
            for j in range(self.nbox):
                for k in range(self.nbox):
                    # consider k as a 3D vector and calculate its magnitude squared
                    r = self.radial[i][j][k]
                    if r != 0:
                        # Calculate the index of the 'k' bin in the power spectrum
                        # Calculate the contribution to the power spectrum
                        contrib = (np.abs(fft_data[i,j,k]) ** 2)
                        #if (i==1 and j==1):
                        #    print(f"Contrib: {i} {j} {k} {g} {coeff} {i1} {fft_data[i,j,k]} {contrib}")
                        powspec[r] += contrib

        deltasqk = np.zeros(self.nbox, dtype=np.float64)
        
        for i in range(self.nhalf):
            powspec[i] = powspec[i] * self.powspec_coeff
            powspec[i] /= float(self.iweights[i])
            deltasqk[i] = self.w_cube[i] * powspec[i] 
        
        #for i in range(nhalf):
        #    print(f"{w[i]}  {powspec[i]}  {deltasqk[i]}  {iweights[i]}")
        return (deltasqk, self.w)
