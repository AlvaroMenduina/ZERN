## ------------------------------------------------------- ##
#-                         Phase Map                       -#
## ------------------------------------------------------- ##

"""
Example of how to use ZERN to generate series expansions of Zernike polynomials
to be used as wavefront maps

After the map is created, the Power Spectral Density profile is computed along the radial direction
showing a typical power law PSD ~ freq ^ alfa behaviour, with alfa ~= - 2

"""

import numpy as np
from numpy.fft import fft2, fftshift, fftfreq
from numpy.random import RandomState
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import zern_core as zern

# Parameters
N = 1024
N_zern = 100
rho_max = 0.9
eps_rho = 1.
randgen = RandomState(12345)

extents = [-1, 1, -1, 1]

# Construct the coordinates
x = np.linspace(-rho_max, rho_max, N)
rho_spacing = x[1] - x[0]
xx, yy = np.meshgrid(x, x)
rho = np.sqrt(xx ** 2 + yy ** 2)
theta = np.arctan2(xx, yy)
aperture_mask = rho <= eps_rho * rho_max
rho, theta = rho[aperture_mask], theta[aperture_mask]
rho_max = np.max(rho)
extends = [-rho_max, rho_max, -rho_max, rho_max]

# Compute the Zernike series
coef = randgen.normal(size=N_zern)
z = zern.ZernikeNaive(mask=aperture_mask)
phase_map = z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option=None)

phase_map = zern.rescale_phase_map(phase_map, peak=1)
phase_2d = zern.invert_mask(phase_map, aperture_mask)

plt.figure()
plt.imshow(phase_2d, extent=extends, cmap='jet')
plt.title("Zernike Series (%d polynomials)" %N_zern)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

# Compute the Power Spectral Density of the Zernike map
phase_f = fftshift(fft2(phase_2d))
power_f = (np.abs(phase_f) / N / N )**2
spatial_frequencies = fftfreq(N, d=rho_spacing)
freq_plus = spatial_frequencies[:N//2]

# Slice the 2D result for all 4 sub-axis: +X, +Y, -X, -Y
power_px, power_py = power_f[N//2:, N//2], power_f[N//2, N//2:]
power_mx, power_my = power_f[:N//2, N//2][::-1], power_f[N//2, :N//2][::-1]

plt.figure()
plt.loglog(freq_plus, power_px, label='+X')    # Along +X
plt.loglog(freq_plus, power_py, label='+Y')    # Along +Y
plt.loglog(freq_plus, power_mx, label='-X')    # Along -X
plt.loglog(freq_plus, power_my, label='-Y')    # Along -Y
plt.xlim([1, freq_plus.max()])
plt.xlabel('Spatial frequency')
plt.ylabel('Power')
plt.title('Power Spectral Density of the Zernike surface')
plt.legend()

# Fit the PSD to get the frequency exponent PSD ~ freq ** alfa
# Typically alfa = - 2

def residual_ls_fit(param):
    """
    Model: log(PSD) = log(p0) + alfa * log(freq)
    """
    return np.log10(power_px[1:]) - param[0] - param[1] * np.log10(freq_plus[1:])

param0 = [1e-2, -2.]
result = least_squares(residual_ls_fit, param0)
print('\nLeast Squares fit of PSD to a power law: PSD ~ freq ^ (%.2f)' %(result.x[1]))



plt.show()
