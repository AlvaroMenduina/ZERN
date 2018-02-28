### --------------------------------------- ###
#-#                   ZERN                  #-#
### --------------------------------------- ###

"""
Python package for the evaluation of Zernike polynomials

Date: Jan 2018
Author: Alvaro Menduina Fernandez - University of Oxford
Email: alvaro.menduinafernandez@physics.ox.ac.uk
Version: 0.1
Description: this package implements several methods to compute
Zernike polynomials which can be summarised as follows
    (1) Standard: naive implementation of the Zernike formulas. Very slow
    (2) Jacobi: uses the relation between Jacobi and Zernike polynomials
        and recurrence formulas to speed up the computation. Significantly Faster!
    (3) Improved Jacobi: the same as Jacobi but exploiting symmetries and
        re-using previously computed polynomials. Even faster than normal Jacobi
"""

import numpy as np
from math import factorial as fact
import matplotlib.pyplot as plt
from time import time as tm
from numba import jit

counter = 0

def parity(n):
    """ Returns 0 if n is even and 1 if n is odd """
    return int((1 + (-1)**(n+1))/2)

def invert_mask(x, mask):
    """
    Takes a vector X which is the result of masking a 2D with the Mask
    and reconstructs the 2D array
    Useful when you need to evaluate a Zernike Surface and most of the array is Masked
    """
    N = mask.shape[0]
    ij = np.argwhere(mask==True)
    i, j = ij[:,0], ij[:,1]
    result = np.zeros((N, N))
    result[i,j] = x
    return result

def invert_model_matrix(H, mask):
    """
    Take the Zern Model Matrix H (whichs has the M(Nx*Ny and flattened) * N_Zern shape
    and restructure it back to a Nx * Ny * N_zern tensor
    """
    N, N_zern = mask.shape[0], H.shape[1]
    new_H = np.zeros((N, N, N_zern))
    for k in range(N_zern):
        zern = H[:, k]
        zern2D = invert_mask(zern, mask)
        new_H[:,:,k] = zern2D
    return new_H

def rescale_phase_map(phase_map, peak=1):
    """
    Rescales a given phase map (Zernike expansion) by shifting it to (max - min)/2
    and limiting its peak_to_valley so that max(new_map) = peak
    and min(new_map) = - peak
    """
    new_origin = (phase_map.max() + phase_map.min())/2
    zero_mean_map = phase_map - new_origin
    rescaled_map = (peak) * zero_mean_map / np.max(zero_mean_map)
    return rescaled_map

def get_limit_index(N):
    """
    Computes the 'n' Zernike index required to generate a
    Zernike series expansion containing at least N polynomials.

    It is based on the fact that the total amount of polynomials is given by
    the Triangular number T(n + 1) defined as:
        T(x) = x (x + 1) / 2
    """
    n = int(np.ceil(0.5 * (np.sqrt(1 + 8*N) - 3)))
    return n

def least_squares_zernike(coef_guess, zern_data, zern_model):
    """
    Computes the residuals (in the least square sense) between a given
    Zernike phase map (zern_data) and a guess (zern_guess) following the model:
        observations = model * parameters + noise
        zern_data ~= zern_model.model_matrix * coef_guess

    This function can be passed to scipy.optimize.least_squares

    :param coef_guess: an initial guess to start the fit.
    In scipy.optimize.least_squares this is your 'x'
    :param zern_data: a given surface map which you want to fit to Zernikes
    :param zern_model: basically a ZernikeNaive object
    """
    zern_guess = np.dot(zern_model.model_matrix, coef_guess)
    residuals = zern_data - zern_guess
    return residuals

class ZernikeNaive(object):
    def __init__(self, mask):
        """
        Object which computes a Series expansion of Zernike polynomials.
        It is based on true different methods:

            (1) Naive and slow application of the Zernike formulas

            (2) Faster and more elegant version using Jacobi polynomials
                The time required to evaluate each polynomial in the Jacobi version
                scales very mildly with its order, leading to quite fast evaluations.
                In contrast, the Zernike version scales dramatically

        Even when using the Jacobi method, the implementation is not the smartest
        and several optimizations can be made, which are exploited in ZernikeSmart (below)
        """
        self.mask = mask

    def R_nm(self, n, m, rho):
        """
        Computes the Radial Zernike polynomial of order 'n', 'm'
        using a naive loop based on the formal definition of Zernike polynomials
        """
        n, m = np.abs(n), np.abs(m)
        r = np.zeros_like(rho)

        if (n - m) % 2 != 0:
            return r
        else:
            for j in range(int((n - m) / 2) + 1):
                coef = ((-1) ** j * fact(n - j)) / (fact(j) * fact((n + m) / 2 - j) * fact((n - m) / 2 - j))
                r += coef * rho ** (n - 2 * j)
            return r

    def R_nm_Jacobi(self, n, m, rho):
        """
        Computes the Radial Zernike polynomial of order 'n', 'm' R_nm
        but this version uses a method which is faster than the Naive R_nm.

        It exploits the relation between the Radial Zernike polynomial and Jacobi polynomials
            R_nm(rho) = (-1)^[(n-m)/2] * rho^|m| * J_{[(n-m)/2]}^{|m|, 0} (1 - 2*rho^2)

        In simpler terms, the R_nm polynomial evaluated at rho, is related to the J_{k}^{alfa, beta},
        the k-th Jacobi polynomial of orders {alfa, beta} evaluated at 1 - 2 rho^2,
        with k = (n-m)/2, alfa = |m|, beta = 0

        To calculate each Jacobi polynomial, it takes advantage of recurrence formulas
        """
        n, m = np.abs(n), np.abs(m)
        m_m = (n - m) / 2
        x = 1. - 2 * rho ** 2
        R = (-1) ** (m_m) * rho ** m * self.Jacobi(x, n=m_m, alfa=m, beta=0)
        return R

    def Jacobi(self, x, n, alfa, beta):
        """
        Returns the Jacobi polynomial J_{n}^{alfa, beta} (x)
        For the sake of efficiency and numerical stability it relies on a 3-term recurrence formula
        """
        J0 = np.ones_like(x)
        J1 = 0.5 * ((alfa - beta) + (alfa + beta + 2) * x)
        if n == 0:
            return J0
        if n == 1:
            return J1
        if n >= 2:
            J2 = None
            n_n = 2
            # Recurrence Relationship
            # a1n' * J_{n'+1} (x) = (a2n' + a3n' * x) * J_{n'} (x) - a4n' * J_{n'-1} (x)
            alfa_beta = alfa + beta
            while n_n <= n:
                # Update recurrence coefficients
                n2_alfa_beta = 2 * n_n + alfa_beta
                a1n = 2 * n_n * (n_n + alfa_beta) * (n2_alfa_beta - 2)
                a2n = (n2_alfa_beta - 1) * (x * n2_alfa_beta * (n2_alfa_beta - 2) + alfa ** 2 - beta ** 2)
                a3n = 2 * (n_n + alfa - 1) * (n_n + beta - 1) * n2_alfa_beta

                J2 = (a2n * J1 - a3n * J0) / a1n
                J0 = J1  # Update polynomials
                J1 = J2
                n_n += 1

            return J2

    def R_nm_ChongKintner(self, n, m, rho):
        """
        Computes the Radial Zernike polynomial of order 'n', 'm' R_nm
        This one uses a similar approach to the one implemented by R_nm_Jacobi.

        This time, the Q-recursive method developed by Chong [1] is used in combination with
        the modified Kintner's method to implement a direct recurrence on the Zernike R_nm.
        The method and formulas are described in [2]

        The main differences with respect to R_nm_Jacobi is that this method directly uses
        the radial Zernike R_nm, and that its recurrence operates along the order 'm' (row-wise)
        for a fixed 'n'. In contrast, R_nm_Jacobi operates along the order 'n' (column-wise)
        for a fixed 'm'.

        This method is not as competitive as the Jacobi because it relies on the evaluation of
        R_{n,n} = rho ^ n   and    R_{n, n-2} = n rho^n - (n - 1) rho^(n-2)
        which scales badly with 'n'
        In contrast, Jacobi keeps the order of the polynomial to k = (n - m) / 2 which is much smaller

        References:
            [1] C.W. Chong, P. Raveendran, R. Mukundan. "A comparative analysis of algorithms for fast computation
                of Zernike moments. Pattern Recognition 36 (2003) 731-742
            [2] Sun-Kyoo Hwang, Whoi-Yul Kim "A novel approach to the fast computation of Zernike moments"
                Pattern Recognition 39 (2006) 2065-2076
        """
        n, m = np.abs(n), np.abs(m)

        if m == n:  # Right at the boundary
            R_nm = rho ** n
            return R_nm

        if m == (n - 2):    # One before the boundary
            R_nm = n * rho ** n - (n - 1) * rho ** (n - 2)
            return R_nm

        else:   # Interior polynomial
            R_nn_4 = rho ** n # Compute the one at the boundary R_{n, n}
            R_nn_2 = n * rho ** n - (n - 1) * rho ** (n - 2)    # R_{n, n-2}

            mm = n - 4
            while mm >= m:  # iterative along m
                H3 = - 4 * (m + 2) * (m + 1) / ((n + m + 2) * (n - m))
                H2 = H3 * (n + m + 4) * (n - m - 2) / (4*(m + 3)) + (m + 2)
                H1 = (m + 4)* (m + 3)/2 - (m + 4) * H2 + H3 * (n + m + 6) * (n - m - 4) / 8

                R_nn = H1 * R_nn_4 + (H2 + H3 / rho ** 2) * R_nn_2

                R_nn_4 = R_nn_2
                R_nn_2 = R_nn
                mm -= 2
            return R_nn

    def Z_nm(self, n, m, rho, theta, normalize_noll, mode):
        """
        Main function to evaluate a single Zernike polynomial of order 'n', 'm'

        You can choose whether to normalize the polynomilas depending on the order,
        and which mode (Naive, Jacobi or ChongKintner) to use.

        :param rho: radial coordinate (ideally it should come normalized to 1)
        :param theta: azimuth coordinate
        :param normalize_noll: True {Applies Noll coefficient}, False {Does nothing}
        :param mode: whether to use 'Standard' (naive Zernike formula),
                'Jacobi' (Jacobi-based recurrence) or 'ChongKintner' (Zernike-based recurrence)
        """

        if mode == 'Standard':
            R = self.R_nm(n, m, rho)
        if mode == 'Jacobi':
            R = self.R_nm_Jacobi(n, m, rho)
        if mode == 'ChongKintner':
            R = self.R_nm_ChongKintner(n, m, rho)

        if m == 0:
            if n == 0:
                return np.ones_like(rho)
            else:
                norm_coeff = np.sqrt(n + 1) if normalize_noll else 1.
                return norm_coeff * R
        if m > 0:
            norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if normalize_noll else 1.
            return norm_coeff * R * np.cos(np.abs(m) * theta)
        if m < 0:
            norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if normalize_noll else 1.
            return norm_coeff * R * np.sin(np.abs(m) * theta)

    def evaluate_series(self, rho, theta, normalize_noll, mode, print_option='Result'):
        """
        Iterates over all the index range 'n' & 'm', computing each Zernike polynomial
        """

        try:
            n_max = self.n
        except AttributeError:
            raise AttributeError('Maximum n index not defined')

        rho_max = np.max(rho)
        extends = [-rho_max, rho_max, -rho_max, rho_max]

        zern_counter = 0
        Z_series = np.zeros_like(rho)
        self.times = []  # List to save the times required to compute each Zernike
        for n in range(n_max + 1):  # Loop over the Zernike index
            for m in np.arange(-n, n + 1, 2):
                start = tm()
                Z = self.Z_nm(n, m, rho, theta, normalize_noll, mode)
                self.times.append((tm() - start))

                # Fill the column of the Model matrix H
                # Important! The model matrix contains all the polynomials of the
                # series, so one can use it to recompute a new series with different
                # coefficients, without redoing all the calculation!
                self.model_matrix[:, zern_counter] = Z


                Z_series += self.coef[zern_counter] * Z
                zern_counter += 1

                if print_option == 'All':
                    print('n=%d, m=%d' % (n, m))
                    if m>=0:    # Show only half the Zernikes to save Figures
                        plt.figure()
                        plt.imshow(invert_mask(Z, self.mask), extent=extends, cmap='jet')
                        plt.title("Zernike(%d, %d)" %(n,m))
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.colorbar()

        if print_option == 'Result':
            plt.figure()
            plt.imshow(invert_mask(Z_series, self.mask), extent=extends, cmap='jet')
            plt.title("Zernike Series (%d polynomials)" %self.N_zern)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar()

        return Z_series

    def __call__(self, coef, rho, theta, normalize_noll=False, mode='Standard', print_option=None):
        self.N_zern = coef.shape[0]
        # Compute the radial index 'n' needed to have at least N_zern
        self.n = get_limit_index(self.N_zern)
        N_new = int((self.n + 1) * (self.n + 2) / 2)    # Total amount of Zernikes
        if N_new > self.N_zern:  # We will compute more than we need
            self.coef = np.pad(coef, (0, N_new - self.N_zern), 'constant')  # Pad to match size
        elif N_new == self.N_zern:
            self.coef = coef

        # Check whether the Model matrix H was already created
        # Observations Z(rho, theta) = H(rho, theta) * zern_coef
        try:
            H = self.model_matrix
        except AttributeError:
            self.model_matrix = np.empty((rho.shape[0], N_new))

        result = self.evaluate_series(rho, theta, normalize_noll, mode, print_option)

        if print_option != 'Silent':
            print('\n Mode: ' + mode)
            print('Total time required to evaluate %d Zernike polynomials = %.3f sec' % (N_new, sum(self.times)))
            print('Average time per polynomials: %.3f ms' % (1e3 * np.average(self.times)))
        return result

class ZernikeSmart(object):

    def __init__(self, mask):
        """
        Improved version of ZernikeNaive, completely based on Jacobi polynomials
        but more sophisticaded to gain further speed advantage

        Advantages:
            (1) It only computes the Radial Zernike polynomial R_nm, for m >= 0 (right side of the triangle)
                thus avoiding repetition in -m +m

            (2) To exploit the Jacobi recurrence even further, it creates a dictionary with the corresponding
                Jacobi polynomials needed to build the rest.
                Each time a new Jacobi polynomial is created, it's added to the dictionary to be reused later on

        Explanation of (2):
        Every Jacobi P_{k}^{alfa, beta} can be recovered by recurrence along its alfa column, based on
        P_{0}^{alfa, beta} and P_{1}^{alfa, beta}. Zernike and Jacobi polynomials are related such that:

            k = (n-m)/2    alfa = |m|    beta = 0

        Beta is always 0 for Zernike so it doesn't play a role

        By definition, P_{0}^{alfa, 0} = 1, no matter the alfa. So the first side-layer of the pyramid is always 1
        The second side-layer P_{1}^{alfa, 0} = 1/2 * [(alfa - beta=0) + (alfa + beta=0 + 2)x]

        In conclusion, for a Maximum index n=N_max, one can create an initial dictionary containing the corresponding
        first side-layer P_{0}^{alfa, 0} (all Ones), the second layer P_{1}^{alfa, 0}, and use the recurrence
        formula of Jacobi polynomials to expand the dictionary.

        Zernike     Jacobi

                        alfa=0          alfa=1          alfa=2          alfa=3
        ------------------------------------------------------------------------------
        n=0         n=0
                    m=0  P_{0}^{0,0}
                    k=0

        n=1                         n=1
                                    m=1  P_{0}^{1,0}
                                    k=0

        n=2         n=2                             n=2
                    m=0  P_{1}^{0,0}                m=2 P_{0}^{2,0}
                    k=1                             k=0

        n=3                         n=3                             n=3
                                    m=1  P_{1}^{1,0}                 m=1  P_{0}^{3,0}
                                    k=1                             k=0

        """

        self.mask = mask

    def create_jacobi_dictionary(self, n_max, x, beta=0):
        """
        For a given maximum radial Zernike index 'n_mx' it creates a dictionary containing
        all the necessary Jacobi polynomials to start the recurrence formulas
        """

        jacobi_polynomials = dict([('P00', np.ones_like(x))])
        for i in range(n_max + 1):
            # In principle this loop is unnecessary because the are all Ones
            # You could just rely on the P00 key, but the dictionary is only
            # created once so it's not a big deal...
            new_key_P0 = 'P0%d' % i
            jacobi_polynomials[new_key_P0] = np.ones_like(x)

        alfa_max = n_max - 2
        for alfa in range(alfa_max + 1):
            new_key_P1 = 'P1%d' % alfa
            jacobi_polynomials[new_key_P1] = 0.5 * ((alfa - beta) + (alfa + beta + 2) * x)

        self.dict_pol = jacobi_polynomials

    def smart_jacobi(self, x, n, alfa, beta):
        """
        Returns the Jacobi polynomial J_{n}^{alfa, beta} (x)
        It relies in the existence of a dictionary containing the initial
        J_{0}^{alfa, 0} (x)  and J_{1}^{alfa, 0} (x)
        """

        if n == 0:
            J0 = self.dict_pol['P0%d' % alfa]
            return J0
        if n == 1:
            J1 = self.dict_pol['P1%d' % alfa]
            return J1
        if n >= 2:
            # Check if previous is already in the dictionary
            # J_prev = self.dict_pol['P%d%d' %(n-1, alfa)]
            # print(J_prev)

            J0 = self.dict_pol['P%d%d' %(n-2, alfa)]
            J1 = self.dict_pol['P%d%d' %(n-1, alfa)]
            J2 = None
            n_n = n

            # J0 = self.dict_pol['P0%d' % alfa]
            # J1 = self.dict_pol['P1%d' % alfa]
            # J2 = None
            # n_n = 2

            # Recurrence Relationship
            # a1n' * J_{n'+1} (x) = (a2n' + a3n' * x) * J_{n'} (x) - a4n' * J_{n'-1} (x)
            alfa_beta = alfa + beta
            while n_n <= n:     # In theory this loop should only be accessed once!
                # print(n_n)
                # Update recurrence coefficients
                n2_alfa_beta = 2 * n_n + alfa_beta
                a1n = 2 * n_n * (n_n + alfa_beta) * (n2_alfa_beta - 2)
                a2n = (n2_alfa_beta - 1) * (x * n2_alfa_beta * (n2_alfa_beta - 2) + alfa ** 2 - beta ** 2)
                a3n = 2 * (n_n + alfa - 1) * (n_n + beta - 1) * n2_alfa_beta

                J2 = (a2n * J1 - a3n * J0) / a1n
                J0 = J1  # Update polynomials
                J1 = J2
                n_n += 1

            return J2

    def fill_in_dictionary(self, rho, theta, normalize_noll=False, print_option=None):
        """
        Takes the dictionary containing the Jacobi Polynomials needed to start the
        recurrence and updates the dictionary with the newly computed polynomials

        At the same time, it translates the Jacobi polynomials into Zernike polynomials
        and adds them into a Zernike series
        """

        # Transform rho to Jacobi coordinate x = 1 - 2 * rho**2
        x = 1. - 2 * rho ** 2

        rho_max = np.max(rho)
        extends = [-rho_max, rho_max, -rho_max, rho_max]

        zern_counter = 0
        Z_series = np.zeros_like(rho)
        self.times = []  # List to save the times required to compute each Zernike

        # Fill up the dictionary
        for n in range(self.n + 1):
            for m in np.arange(parity(n), n + 1, 2):
                n_n = (n - m) // 2
                alfa = m
                # Compute the corresponding Jacobi polynomial via Recursion
                start = tm()
                P_n_alfa = self.smart_jacobi(x=x, n=n_n, alfa=alfa, beta=0)
                self.dict_pol['P%d%d' % (n_n, alfa)] = P_n_alfa
                # Transform Jacobi to Zernike Radial polynomial R_nm
                R = (-1)**(n_n) * rho**m * P_n_alfa

                # Transform to complete Zernike Z_nm
                if m == 0:
                    norm_coeff = np.sqrt(n + 1) if normalize_noll else 1.
                    Z = norm_coeff * R
                    end = tm()
                    self.times.append((end - start))
                    Z_series += self.coef[zern_counter] * Z
                    zern_counter += 1
                    if print_option == 'All':
                        print('n=%d, m=%d' % (n, m))
                        plt.figure()
                        plt.imshow(invert_mask(Z, self.mask), extent=extends, cmap='jet')
                        plt.title("Zernike(%d, %d)" %(n,m))
                        plt.colorbar()

                else:   # m > 0
                    norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if normalize_noll else 1.
                    # Compute the m+ Zernike
                    Zpos = norm_coeff * R * np.cos(np.abs(m) * theta)
                    end1 = tm()
                    Z_series += self.coef[zern_counter] * Zpos
                    zern_counter += 1
                    # Compute the m- Zernike
                    Zneg = norm_coeff * R * np.sin(np.abs(m) * theta)
                    end2 = tm()
                    self.times.append((end1 - start))
                    self.times.append((end2 - end1))
                    Z_series += self.coef[zern_counter] * Zneg
                    zern_counter += 1

                    if print_option == 'All':   # Show only m > 0 to save Figures
                        print('n=%d, m=%d' % (n, m))
                        plt.figure()
                        plt.imshow(invert_mask(Zpos, self.mask), extent=extends, cmap='jet')
                        plt.title("Zernike(%d, %d)" %(n,m))
                        plt.colorbar()
                        # plt.figure()
                        # plt.imshow(invert_mask(Zneg, self.mask), cmap='jet')
                        # plt.title("Zernike(%d, %d)" %(n,-m))
                        # plt.colorbar()
        return Z_series

    def __call__(self, coef, rho, theta, normalize_noll=False, print_option=None):

        self.N_zern = coef.shape[0]
        self.n = get_limit_index(self.N_zern)   # Compute the radial index 'n' needed to have at least N_zern
        N_new = int((self.n + 1) * (self.n + 2) / 2)    # Total amount of Zernikes
        if N_new > self.N_zern:  # We will compute more than we need
            self.coef = np.pad(coef, (0, N_new - self.N_zern), 'constant')  # Pad to match size
        elif N_new == self.N_zern:
            self.coef = coef

        # Transform rho to Jacobi coordinate x = 1 - 2 * rho**2
        x = 1. - 2 * rho ** 2

        try:    # Check if dictionary already exists
            jac_dict = self.dict_pol
        except:
            self.create_jacobi_dictionary(n_max=self.n, x=x, beta=0)

        # Fill in dictionary
        result = self.fill_in_dictionary(rho=rho, theta=theta, normalize_noll=normalize_noll, print_option=print_option)

        print('\n Mode: Improved Jacobi ')
        print('Total time required to evaluate %d Zernike polynomials = %.3f sec' % (N_new, sum(self.times)))
        print('Average time per polynomials: %.3f ms' %(1e3*np.average(self.times)))

        return result

if __name__ == "__main__":

    pass