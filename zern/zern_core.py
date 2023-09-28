### --------------------------------------- ###
#-#                   ZERN                  #-#
### --------------------------------------- ###

"""
Python package for the evaluation of Zernike polynomials

Date: Sept 2023
Author: Alvaro Menduina Fernandez - University of Oxford
Email: alvaro.menduina@gmail.com
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

def parity(n):
    """ Returns 0 if n is even and 1 if n is odd """
    res = 0 if n % 2 == 0 else 1
    return res

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

def get_limit_index(N):
    """
    Computes the 'n' Zernike index required to generate a
    Zernike series expansion containing at least N polynomials.

    It is based on the fact that the total amount of polynomials is given by
    the Triangular number T(n + 1) defined as:
        T(x) = x (x + 1) / 2
    """
    if N < 0:
        raise RuntimeError("'N' should be positive or zero!")

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
    :param zern_model: basically a Zernike object
    """
    zern_guess = np.dot(zern_model.model_matrix_flat, coef_guess)
    residuals = zern_data - zern_guess
    return residuals

class Zernike(object):
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
                F = int((n + m) / 2 - j)
                G = int((n - m) / 2 - j)
                coef = ((-1) ** j * fact(n - j)) / (fact(j) * fact(F) * fact(G))
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
        R = (-1) ** (m_m) * rho ** m * self.get_jacobi_polymonial(x, n=m_m, alfa=m, beta=0)
        return R

    def get_jacobi_polymonial(self, x, n, alfa, beta):
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
        and which mode (Naive, Jacobi or ChongKintner) to use.

        :param rho: radial coordinate (ideally it should come normalized to 1)
        :param theta: azimuth coordinate
        :param normalize_noll: True {Applies Noll coefficient}, False {Does nothing}
        :param mode: whether to use 'Standard' (naive Zernike formula) or 'Jacobi' (Jacobi-based recurrence)
        """

        # [1] - get the Radial polynomial R_nm
        if mode == 'Standard':
            R = self.R_nm(n, m, rho)
        elif mode == 'Jacobi':
            R = self.R_nm_Jacobi(n, m, rho)
        else:
            raise ValueError(f"Unknown mode: [{mode}]. Choose either 'Standard' or 'Jacobi'")

        # [2] - apply azimuth dependency to get Zernike polynomial Z_nm
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
        
    def create_model_matrix(self, rho, theta, n_zernike, mode='Jacobi', normalize_noll=False):

        # Compute the limit radial index 'n' needed to have at least N_zern, and pad zeroes
        # self.N_zern = coef.shape[0] if n_zernike is None else n_zernike
        self.N_zern = n_zernike
        self.n_lim = get_limit_index(self.N_zern)
        self.N_total = int((self.n_lim + 1) * (self.n_lim + 2) / 2)    # Total amount of Zernikes

        self.model_matrix_flat = np.empty((rho.shape[0], self.N_total))
            
        zern_counter = 0
        for n in range(self.n_lim + 1):  # Loop over the Zernike index
            for m in np.arange(-n, n + 1, 2):
                zernike_poly = self.Z_nm(n, m, rho, theta, normalize_noll, mode)
                # Fill the column of the Model matrix H
                # Important! The model matrix contains all the polynomials of the
                # series, so one can use it to recompute a new series with different
                # coefficients, without redoing all the calculation!
                self.model_matrix_flat[:, zern_counter] = zernike_poly
                zern_counter += 1
    
    def get_zernike(self, coef):
        """
        Fast calculation
        TODO: implement a quick method that just does the dot(H, coef)
        """

        # [0] See if the model matrix exists
        try:
            _matrix = self.model_matrix_flat
        except AttributeError:
            # Throw an error if the model matrix does not exist yet
            raise AttributeError("Model matrix does not yet exist, please run Zernike.__call__() first")

        # [1] See if the coef needs padding
        if self.N_total > coef.shape[0]:
            _coef = np.pad(coef, (0, self.N_total - coef.shape[0]), 'constant')  # Pad to match size
        elif self.N_total == coef.shape[0]:
            _coef = coef
        else:
            raise ValueError(f"Model matrix of shape {self.model_matrix_flat.shape} and coefficient array of size {coef.shape}. Consider recreating the model matrix!")

        result_flat = np.dot(self.model_matrix_flat, _coef)
        result = invert_mask(result_flat, self.mask)

        return result


if __name__ == "__main__":

    pass