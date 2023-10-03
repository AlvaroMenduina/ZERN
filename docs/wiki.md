## [1] Zernike polynomials
The Zernike polynomials are a set of orthogonal polynomials in the unit circle. The low-order Zernike polynomials correspond to common optical aberrations such as astigmatism or coma and thus, constitute a useful basis for describing wavefront maps.

Zernike polynomials have both two $n, m$, the first one representing the *radial* order, the second one representing the *azimuth* order. They are composed of a radial polynomials $R_{n}^{m}(\rho)$ scaled by a trigonometric function to form *even* and *odd* Zernike polynomials:

$Z_{n}^{m}(\rho, \theta) = R_{n}^{m}(\rho) \cdot \cos(m\theta)$

$Z_{n}^{-m}(\rho, \theta) = R_{n}^{m}(\rho) \cdot \sin(m\theta)$

The Zernikes are usually defined with the following formula:

$R_{n}^{m}(\rho) = \sum_{k=0}^{\frac{n-m}{2}} \frac{(-1)^k(n-1)!}{k!(\frac{n+m}{2} - k)!(\frac{n-m}{2} - k)!} \rho^{n-2k}$