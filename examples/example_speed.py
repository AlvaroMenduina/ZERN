## ------------------------------------------------------- ##
#-                     Speed Comparison                    -#
## ------------------------------------------------------- ##

"""
Example to show how each of the methods in ZERN perform in terms of speed

    (1) ZernikeNaive(method='Standard') is the simplest implementation but it scales poorly
        with the amount and order of Zernike polynomials being evaluated

    (2) ZernikeNaive(method='Jacobi') is much more competitive. The cost per polynomial barely
        increases with order (n, m)

    (3) ZernikeNaive(method='ChongKintner') is faster than the Standard, but not as good as Jacobi

    (4) ZernikeSmart (based on Jacobi) brings slightly better performance than the previous one
"""
import zern.zern_core as zern
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import logging

plt.rc('font', family='sans-serif')

# Parameters
N = 1024
N_zern = 50
rho_max = 1.0
randgen = RandomState(12345)  # random seed


if __name__ == """__main__""":

    # [0] Construct the coordinates and the aperture mask - simple circ
    x = np.linspace(-rho_max, rho_max, N)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)
    aperture_mask = rho <= rho_max
    rho, theta = rho[aperture_mask], theta[aperture_mask]

    # [1] Create an instance of the ZernikeNaive class
    z = zern.ZernikeNaive(mask=aperture_mask, log_level=logging.DEBUG)
    coef = randgen.normal(size=10)      # Coefficients of the Zernike Series expansion
    result = z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Standard', print_option='All')
    print(result.shape)
    result_2d = zern.invert_mask(result, z.mask)
    plt.figure()
    plt.imshow(result_2d)
    plt.show()
    # Show the first few Zernike Polynomials
    # raise ValueError

    print('\n Comparing the speed of several methods')

    # Show a speed comparison of Naive and Jacobi modes
    coef = randgen.normal(size=N_zern)
    z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Standard', print_option=None)
    times_naive = z.times
    z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option=None)
    times_jacobi = z.times
    z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='ChongKintner', print_option=None)
    times_chong = z.times

    z_smart = zern.ZernikeSmart(mask=aperture_mask)
    z_series = z_smart(coef, rho, theta, normalize_noll=False, print_option=None)
    times_smart = z_smart.times

    # plt.figure()
    # plt.imshow(zern.invert_mask(z_series, aperture_mask), cmap='jet')
    # plt.title("Zernike Series containing %d polynomials" % N_zern)
    # plt.colorbar()

    plt.figure()
    plt.scatter(np.arange(len(times_naive)), times_naive, label='Naive.Standard', s=12)
    plt.scatter(np.arange(len(times_jacobi)), times_jacobi, label='Naive.Jacobi', s=12)
    plt.scatter(np.arange(len(times_chong)), times_chong, c='g', s=12)
    plt.plot(np.arange(len(times_chong)), times_chong, c='g', label='Naive.Chong')
    plt.scatter(np.arange(len(times_smart)), times_smart, c='r', s=12)
    plt.plot(np.arange(len(times_smart)), times_smart, c='r', label='Smart.Jacobi')
    plt.legend()
    plt.xlim([0, len(times_naive)])
    plt.ylim([0, max(times_naive)])
    plt.xlabel('Number of polynomials evaluated')
    plt.ylabel('Time [sec]')
    plt.title('Time spent in each polynomial')

    # Get the trends: total_time vs number polynomials
    avg_naive = [np.sum(times_naive[:(i+1)]) for i in range(len(times_naive))]
    avg_jacobi = [np.sum(times_jacobi[:(i+1)]) for i in range(len(times_jacobi))]
    avg_chong = [np.sum(times_chong[:(i+1)]) for i in range(len(times_chong))]
    # avg_smart = [np.sum(times_smart[:(i+1)]) for i in range(len(times_smart))]

    plt.figure()
    plt.plot(np.arange(len(avg_naive)), avg_naive, label='Standard')
    plt.plot(np.arange(len(avg_jacobi)), avg_jacobi, label='Jacobi')
    plt.plot(np.arange(len(avg_chong)), avg_chong, label='Chong')
    # plt.plot(np.arange(len(avg_smart)), avg_smart, label='Smart Jacobi')
    plt.legend()
    plt.xlim([0, len(times_naive)])
    plt.ylim([0, max(avg_naive)])
    plt.xlabel('Number of polynomials evaluated')
    plt.ylabel('Time [sec]')
    plt.title('Total time to compute %d polynomials' %(len(times_naive)))

    plt.show()