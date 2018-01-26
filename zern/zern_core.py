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
from numpy.random import RandomState
from math import factorial as fact
import matplotlib.pyplot as plt
from time import time as tm

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
        but this version uses a faster method.

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

    def Z_nm(self, n, m, rho, theta, normalize_noll, mode):
        """
        Main function to evaluate a single Zernike polynomial of order 'n', 'm'

        You can choose whether to normalize the polynomilas depending on the order,
        and which mode (Naive or Jacobi) to use.

        :param rho: radial coordinate (ideally it should come normalized to 1)
        :param theta: azimuth coordinate
        :param normalize_noll: True {Applies Noll coefficient}, False {Does nothing}
        :param mode: whether to use 'Standard' (naive Zernike formula) or 'Jacobi' (recurrence)
        """

        if mode == 'Standard':
            R = self.R_nm(n, m, rho)
        if mode == 'Jacobi':
            R = self.R_nm_Jacobi(n, m, rho)

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

        result = self.evaluate_series(rho, theta, normalize_noll, mode, print_option)

        if mode == 'Standard':
            print('\n Mode: Standard Zernike (naive)')
        if mode == 'Jacobi':
            print('\n Mode: Jacobi (faster)')
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
        P_{0}^{alfa, beta} and P_{1}^{alfa, beta}. Beta is always 0 for Zernike so it doesn't play a role

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
            # You could just rely on the P00 key
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