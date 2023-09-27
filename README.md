# ZERN
Python package for fast evaluation of Zernike polynomials Z_{n,m} (rho, theta)

# Description
This package implements several methods to compute Zernike polynomials which can be summarised as follows:

  1) A naive implementation of the Zernike formulas, used to show the worst case scenario. It scales very poorly with the amount of polynomials being evaluated and their order {n, m}.
  2) A method developed by Chong which uses direct recurrence of Zernikes to speed up computation.
  3) A method based on the relationship between Jacobi polynomials J_{k}^{alfa, beta} and the Zernike polynomials. It takes advantage of 3-term recurrence formulas of Jacobi polynomials to speed up the computation. Not only is it much faster than the naive implementation but it is also more robust in terms of numerical stability.
  4) An extension of the Jacobi method to improve its efficiency


## Installation

To install ZERN, you can follow these simple steps

### 1. Download the files from GitHub

Download the repo directly from Github by running git clone:

```
git clone https://github.com/AlvaroMenduina/ZERN.git <your_local_dir>
```

Move to where you copied the repository files
```
cd <your_local_dir>
```

And run 

```
python setup.py sdist

pip install .
```

To test whether the installation was successful, you can try to import the module
```python
import zern.zern_core as zern
_test = zern.Zernike(mask=None)
```
