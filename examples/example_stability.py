## ------------------------------------------------------- ##
#-                   Stability Comparison                  -#
## ------------------------------------------------------- ##

"""
Example to show how each of the methods in ZERN perform in terms of Numerical Stability

    (1) ZernikeNaive(method='Standard')

    (2) ZernikeNaive(method='Jacobi')

    (3) ZernikeSmart (based on Jacobi) brings slightly better performance than the previous one
"""

import zern.zern_core as zern
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='sans-serif')

# Parameters
N = 1024
N_zern = 100
rho_max = 1.0
mask = []

# Construct the coordinates
rho = np.linspace(0., rho_max, N)

Z_naive = zern.ZernikeNaive(mask)

# Zernike orders
n_list = [40, 44, 48]
m = 0
n_max = 101

# for n in n_list:
#     # Compute the Radial Zernike R_nm
#     R_naive = Z_naive.R_nm(n, m, rho)
#     R_jacobi = Z_naive.R_nm_Jacobi(n, m, rho)
#
#     plt.figure()
#     plt.plot(rho, R_naive, label='Standard')
#     plt.plot(rho, R_jacobi, label='Jacobi')
#     plt.legend()
#     # Show only detail of the edge
#     plt.xlim([0.8*rho.max(), rho.max()])
#     plt.ylim([R_naive.min()-1, R_naive.max()+1])
#     plt.xlabel('rho')
#     plt.ylabel('R')
#     plt.title('Radial Zernike R_{n=%d,m=%d}'%(n,m))

# Error matrix
error = np.zeros((n_max, (n_max-1)//2))
for i in np.arange(2, n_max):  # n index loop
    j_start = zern.parity(i)    # Decide whether m starts at 0 or 1
    for j in np.arange(j_start, i+1, 2):    # m index loop
        R_naive = Z_naive.R_nm(i, j, rho)
        R_jacobi = Z_naive.R_nm_Jacobi(i, j, rho)

        # As the Zernike layers are (n=0, m=0) (n=1, m=1) (n=2, m=0)
        # the positions like (n=1, m=0) (n=2, m=1) are not filled
        # so we shift the columns depending on the parity to use those
        # positions
        j_new = (j-j_start)//2
        if i != j :
            err = np.abs(R_naive - R_jacobi)
            print(np.max(err))

            if np.max(err) > 0.5:
                fig, ax = plt.subplots(1, 1)
                ax.scatter(rho, R_naive, s=2, c='r')
                ax.plot(rho, R_jacobi)
                plt.show()

            error[i, j_new] = np.log10(np.max(err))

np.place(error, error==0., np.nan)


plt.figure()
plt.imshow(error, aspect='0.5', origin='upper')
# plt.axis('off')
plt.xlabel('M')
plt.ylabel('N')
plt.title('log_10(error) between Standard and Jacobi \n(n_max=%d)' %(n_max-1))
plt.colorbar()


plt.show()

