
import zern.zern_core as zern
import numpy as np
import pytest

ATOL = 1e-6

@pytest.mark.parametrize("test_input, expected", [(2, 0), (0, 0), (1, 1), (3, 1)])
def test_parity(test_input, expected):
    assert zern.parity(test_input) == expected

@pytest.mark.parametrize("test_input, expected", [(1, 0), (2, 1), (3, 1), (4, 2)])
def test_limit_index(test_input, expected):
    """
    Test the limit radial order 'n' of polynomials needed to capture a certain number of Zernikes
    Z: 1 Zernike -> n=0 [piston, 0 order]
    Z: 3 Zernike -> n=1 [piston, +- tilt, 1st order]
    Z: 2 Zernike -> still n=1 [piston, +- tilt, 1st order]
    """
    assert zern.get_limit_index(test_input) == expected

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Jacobi Polynomials test
# J_{n}^{alfa, beta} (x)
# [x, n, alfa, beta] -> for the Zernike recursion: alfa = abs(m) | beta = 0

x = np.linspace(-1, 1, 100)
jac_polynomials = [(x, 0, 0, 0, np.ones_like(x)),
                   (x, 1, 1, 0, (1 + 3*x)/2),
                   (x, 2, 0, 0, 1 + 3*(x-1) + 6*((x-1)/2)**2)]

@pytest.mark.parametrize("x, n, alfa, beta, expected", jac_polynomials)
def test_zernike_jacobi(x, n, alfa, beta, expected):

    class_test = zern.Zernike(mask=None)
    result = class_test.get_jacobi_polymonial(x, n, alfa, beta)
    check = all(np.isclose(result, expected, atol=ATOL))
    assert check == True 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Zernike R_{n, m} Radial Polynomials test

rho = np.linspace(0, 1, 100)
rnm_polynomials = [(0, 0, rho, np.ones_like(rho)),
               (1, 1, rho, rho),(1, -1, rho, rho),
               (2, 0, rho, 2*rho**2 - 1),
               (2, 2, rho, rho**2),
               (3, 1, rho, 3*rho**3 - 2*rho),
               (3, -1, rho, 3*rho**3 - 2*rho),
               (3, 3, rho, rho**3),
               (4, 0, rho, 6*rho**4 - 6*rho**2 + 1),
               (4, 2, rho, 4*rho**4 - 3*rho**2)]

@pytest.mark.parametrize("n, m, rho, expected", rnm_polynomials)
def test_zernike_rnm(n, m, rho, expected):

    class_test = zern.Zernike(mask=None)
    result = class_test.R_nm(n, m, rho)
    check = all(np.isclose(result, expected, atol=ATOL))
    assert check == True 

@pytest.mark.parametrize("n, m, rho, expected", rnm_polynomials)
def test_zernike_rnm_jacobi(n, m, rho, expected):

    class_test = zern.Zernike(mask=None)
    result = class_test.R_nm_Jacobi(n, m, rho)
    check = all(np.isclose(result, expected, atol=ATOL))
    assert check == True 

@pytest.mark.parametrize("n, m, rho, expected", rnm_polynomials)
def test_zernike_rnm_jacobi_vs_standard(n, m, rho, expected):

    class_test = zern.Zernike(mask=None)
    result_std = class_test.R_nm(n, m, rho)
    result_jac = class_test.R_nm_Jacobi(n, m, rho)
    check = all(np.isclose(result_std, result_jac, atol=ATOL))
    assert check == True 

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Zernike R_{n, m}
rho = np.linspace(0, 1, 100)
znm_polynomials = [(0, 0, rho, 0, np.ones_like(rho)),
                   (1, 1, rho, 0, rho),
                   (1, 1, rho, np.pi, -rho),
                   (1, -1, rho, 0, np.zeros_like(rho)),
                   (1, -1, rho, np.pi, np.zeros_like(rho)),
                   (1, -1, rho, np.pi/2, rho),
                   (1, -1, rho, 3*np.pi/2, -rho),
                   (2, 0, rho, 0, 2*rho**2 - 1)]

@pytest.mark.parametrize("n, m, rho, theta, expected", znm_polynomials)
def test_zernike_znm_standard_nonorm(n, m, rho, theta, expected):

    class_test = zern.Zernike(mask=None)
    result = class_test.Z_nm(n, m, rho, theta, normalize_noll=False, mode="Standard")
    check = all(np.isclose(result, expected, atol=ATOL))
    assert check == True 

@pytest.mark.parametrize("n, m, rho, theta, expected", znm_polynomials)
def test_zernike_znm_jacobi_nonorm(n, m, rho, theta, expected):

    class_test = zern.Zernike(mask=None)
    result = class_test.Z_nm(n, m, rho, theta, normalize_noll=False, mode="Jacobi")
    check = all(np.isclose(result, expected, atol=ATOL))
    assert check == True 

def test_zernike_zn_unknown_mode():
    fake_mode = "NoMode"
    with pytest.raises(ValueError) as exc_info:
        class_test = zern.Zernike(mask=None)
        class_test.Z_nm(n=4, m=0, rho=np.zeros(10), theta=np.zeros(10), normalize_noll=False, mode=fake_mode)
    expected_message = f"Unknown mode: [{fake_mode}]. Choose either 'Standard' or 'Jacobi'"
    assert str(exc_info.value) == expected_message

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def test_create_model_matrix():
    """
    Test that the created 'model_matrix' is no longer empty after creation
    It has been populated with zernike Z_nm
    """

    class_test = zern.Zernike(mask=None)
    class_test.create_model_matrix(rho=np.zeros(10), theta=np.zeros(10), n_zernike=10, normalize_noll=False, mode="Jacobi")
    assert not np.all(class_test.model_matrix_flat == 0), "Array is still empty after modification"


def test_error_model_matrix():
    """
    Test that the call to 'get_zernike' before the creation of a model matrix raises an AttributeError
    """
    class_test = zern.Zernike(mask=None)
    with pytest.raises(AttributeError): # [NOTE]
        class_test.get_zernike(np.zeros(10))


def test_error_model_matrix_custom_exception():
    """
    Test that the call to 'get_zernike' before the creation of a model matrix raises an the correct message
    """
    with pytest.raises(AttributeError) as exc_info:
        class_test = zern.Zernike(mask=None)
        class_test.get_zernike(np.zeros(10))

    expected_message = "Model matrix does not yet exist, please run Zernike.__call__() first"
    assert str(exc_info.value) == expected_message


def test_incorrect_coef_size():
    with pytest.raises(ValueError) as exc_info:
        class_test = zern.Zernike(mask=None)
        class_test.create_model_matrix(rho=np.zeros(10), theta=np.zeros(10), n_zernike=10, normalize_noll=False, mode="Jacobi")
        class_test.get_zernike(coef=np.zeros(15))

# Inputs: rho, theta, n_zernike, coef, expected
def preprocess_data():
        
    N = 512
    x = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(xx, yy)
    aperture_mask = rho <= 1.0
    rho, theta = rho[aperture_mask], theta[aperture_mask]

    return rho, theta, aperture_mask

r, t, a = preprocess_data()
inputs = [(r, t, a, 10, np.zeros(10), None),
          (r, t, a, 10, np.zeros(5), None)]
@pytest.mark.parametrize("rho, theta, mask, n_zernike, coef, expected", inputs)
def test_padding_coef_padding(rho, theta, mask, n_zernike, coef, expected):
    """
    Check the padding branches work without breaking
    """

    class_test = zern.Zernike(mask=mask)
    class_test.create_model_matrix(rho, theta, n_zernike, normalize_noll=False, mode="Jacobi")
    _result = class_test.get_zernike(coef)
    assert _result.shape == mask.shape

inputs = [(r, t, a, 10, np.zeros(10), None)]
@pytest.mark.parametrize("rho, theta, mask, n_zernike, coef, expected", inputs)
def test_invert_model_matrix(rho, theta, mask, n_zernike, coef, expected):

    class_test = zern.Zernike(mask=mask)
    class_test.create_model_matrix(rho, theta, n_zernike, normalize_noll=False, mode="Jacobi")
    result = zern.invert_model_matrix(class_test.model_matrix_flat, mask)
    assert result.shape[:2] == mask.shape

inputs = [(r, t, a, 36, "Standard", 1.0),
          (r, t, a, 36, "Jacobi", 1.0)]
@pytest.mark.parametrize("rho, theta, mask, n_zernike, mode, expected", inputs)
def test_rms_noll(rho, theta, mask, n_zernike, mode, expected):
    """
    Test that the RMS of the polynomials is 1.0 for the Noll normalizations
    """
    normalization = True
    class_test = zern.Zernike(mask=mask)
    class_test.create_model_matrix(rho, theta, n_zernike, mode, normalize_noll=normalization)
    rms = [np.std(class_test.model_matrix_flat[:, k]) for k in range(class_test.N_total)]
    assert all(np.isclose(rms[1:], expected, atol=1e-2))

inputs = [(r, t, a, 36, "Jacobi", 1.0)]
@pytest.mark.parametrize("rho, theta, mask, n_zernike, mode, expected", inputs)
def test_ptv_noll(rho, theta, mask, n_zernike, mode, expected):
    """
    Test that the RMS of the polynomials is 1.0 for the Noll normalizations
    We need to be flexible on the tolerance because of the mask resolution
    and how sharply the Zernike go up to 1.0 at the edge
    """
    normalization = False
    class_test = zern.Zernike(mask=mask)
    class_test.create_model_matrix(rho, theta, n_zernike, mode, normalize_noll=normalization)
    ptv = [0.5*(np.max(class_test.model_matrix_flat[:, k]) - np.min(class_test.model_matrix_flat[:, k])) for k in range(class_test.N_total)]
    
    # Remove spherical, which has not 1.0 PV but 3/4
    ptv.pop(12)
    assert all(np.isclose(ptv[1:], expected, atol=1e-1))
