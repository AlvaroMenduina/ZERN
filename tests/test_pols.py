
import zern.zern_core as zern
import numpy as np
import matplotlib.pyplot as plt
# import pytest

# @pytest.fixture
# def common_parameter():
#     N = 100
#     return N  # Example parameter value

def test_one():

    test_zernike = zern.ZernikeNaive(mask=None)
    N = 100
    rho = np.linspace(0, 1, N)
    n = 1
    m = 1
    r = test_zernike.R_nm(n, m, rho)
    solution = rho
    residual = all(np.isclose(r, solution, atol=1e-4))
    assert residual == True

# fig, ax = plt.subplots(1, 1)
# ax.plot(rho, r)
# ax.grid(True)
# plt.show()