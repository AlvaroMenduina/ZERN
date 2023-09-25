# ZERN
Python package for fast evaluation of Zernike polynomials Z_{n,m} (rho, theta)

# Description
This package implements several methods to compute Zernike polynomials which can be summarised as follows:

  1) A naive implementation of the Zernike formulas, used to show the worst case scenario. It scales very poorly with the amount of polynomials being evaluated and their order {n, m}.
  2) A method developed by Chong which uses direct recurrence of Zernikes to speed up computation.
  3) A method based on the relationship between Jacobi polynomials J_{k}^{alfa, beta} and the Zernike polynomials. It takes advantage of 3-term recurrence formulas of Jacobi polynomials to speed up the computation. Not only is it much faster than the naive implementation but it is also more robust in terms of numerical stability.
  4) An extension of the Jacobi method to improve its efficiency


# Installation

```python
python setup.py sdist
pip install .
```
