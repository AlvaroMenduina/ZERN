# ZERN - Zernike Polynomials in Python
Python package for the evaluation of Zernike polynomials $Z_{n,m} (\rho, \theta)$.

![Zernike](images/zernike.PNG)

## Description
This package implements several methods to compute Zernike polynomials which can be summarised as follows:

  1) A naive implementation of the Zernike formulas, used to show the worst case scenario. It scales very poorly with the amount of polynomials being evaluated and their order {n, m}.
  2) A method developed by Chong which uses direct recurrence of Zernikes to speed up computation.
  3) A method based on the relationship between Jacobi polynomials $J_{k}^{\alpha, \beta}$ and the Zernike polynomials $Z_{n,m} (\rho, \theta)$. It takes advantage of 3-term recurrence formulas of Jacobi polynomials to speed up the computation. This method more robust in terms of numerical stability than the standard approach, which starts to fail at higher radial order $n$.
  4) An extension of the Jacobi method to improve its efficiency



It supports arbitrary aperture masks.

## Installation

To install ZERN, you can follow these simple steps

### 1. Download the files from GitHub

Download the repo directly from Github by running git clone:

```
git clone https://github.com/AlvaroMenduina/ZERN.git <your_local_dir>
```

### 2. Install the package with pip
Move to where you copied the repository files
```
cd <your_local_dir>
```

And run 

```
python setup.py sdist

pip install .
```

### 3. Verifying it all worked
To test whether the installation was successful, you can try to import the module
```python
import zern.zern_core as zern
_test = zern.Zernike(mask=None)
```

## Testing coverage

```bash
================================ test session starts ================================ 
platform win32 -- Python 3.11.5, pytest-7.4.2, pluggy-1.3.0
rootdir: C:\Users\alvaro\Documents\Python Scripts\ZERN
plugins: anyio-4.0.0
collected 68 items                                                                                                                                                                                                

tests\test_zern.py ......................................................     [100%] 

================================ 68 passed in 3.54s ===============================
PS C:\Users\alvaro\Documents\Python Scripts\ZERN> coverage report --show-missing
Name                 Stmts   Miss  Cover   Missing
--------------------------------------------------
tests\test_zern.py     120      0   100%
zern\__init__.py         0      0   100%
zern\zern_core.py      116      6    95%   67, 86-88, 117, 262
--------------------------------------------------
TOTAL                  236      6    97%
```