## ------------------------------------------------------- ##
#-                    Least Squares fit                    -#
## ------------------------------------------------------- ##



import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import zern_core as zern
from time import time

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
phase_map = z(coef=coef, rho=rho, theta=theta,
              normalize_noll=False, mode='Jacobi', print_option=None)

# phase_map = zern.rescale_phase_map(phase_map, peak=1)
phase_2d = zern.invert_mask(phase_map, aperture_mask)

# Introduce some noise in the map
noised_phase_map = phase_map + 0.5*np.random.normal(size=phase_map.shape[0])
noised_2d = zern.invert_mask(noised_phase_map, aperture_mask)

plt.figure()
plt.imshow(phase_2d, extent=extends, cmap='jet')
plt.title("Zernike Series (%d polynomials)" %N_zern)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

plt.figure()
plt.imshow(noised_2d, extent=extends, cmap='jet')
plt.title("Zernike Series with Noise")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

# The Model Matrix contains the Zernike polynomials ordered column-wise
# Very convinient when you need to recalculate series, like when you do
# some iterative fitting
# for i in range(5):
#     plt.figure()
#     plt.imshow(zern.invert_mask(z.model_matrix[:,i], aperture_mask),
#                extent=extends, cmap='jet')
#     plt.title("Model matrix [%d]" %i)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.colorbar()

# Generate a guess of the phase map
coef_guess = RandomState(54321).normal(size=z.coef.shape[0])
result_ls = least_squares(fun=zern.least_squares_zernike, x0=coef_guess,
                          args=(noised_phase_map, z))

# Show the residuals and the final phase estimation
plt.figure()
plt.plot(z.coef - result_ls.x)
plt.xlabel('Coefficient #')
plt.ylabel('Residual')
plt.title('Deviation of LS coefficients from true value')

phase_guess = zern.invert_mask(np.dot(z.model_matrix, result_ls.x), aperture_mask)
plt.figure()
plt.imshow(phase_guess, extent=extends, cmap='jet')
plt.title("Guessed map (Least Squares fit)")
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()


plt.show()

