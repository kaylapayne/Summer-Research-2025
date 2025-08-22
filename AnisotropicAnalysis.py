import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Read the xi matrix from CSV
xi = pd.read_csv('xi_matrix.csv', index_col=0)

# Define Legendre polynomials
def P_l(mu, l):
    return np.polynomial.legendre.Legendre.basis(l)(mu)

# Compute the l-th multipole xi_l(s)
def compute_multipole(xi, s_values, l):
    xi_l = []
    for s in s_values:
        mu_vals = xi.columns.astype(float)  # convert string column names to float
        xi_vals = xi.loc[s].values          # xi(s, mu) at fixed s
        interp_func = interp1d(mu_vals, xi_vals, bounds_error=False, fill_value=0.0, kind='cubic')
        integrand = lambda mu: interp_func(mu) * P_l(mu, l)
        result, _ = quad(integrand, -1, 1)
        xi_l.append((2 * l + 1) / 2 * result)
    return np.array(xi_l)

# Define separation distances
s_values = xi.index.to_numpy()

# Compute multipoles
xi_0_raw = s_values**2 * compute_multipole(xi, s_values, 0)
xi_2_raw = s_values**2 * compute_multipole(xi, s_values, 2)
xi_4_raw = s_values**2 * compute_multipole(xi, s_values, 4)

# Apply Gaussian smoothing
xi_0 = xi_0_raw #gaussian_filter1d(xi_0_raw, sigma=2)
xi_2 = xi_2_raw #gaussian_filter1d(xi_2_raw, sigma=2)
xi_4 = xi_4_raw #gaussian_filter1d(xi_4_raw, sigma=2)

# Placeholder errors (10% of signal amplitude — replace with mocks if available)
err_0 = 0.1 * np.abs(xi_0)
err_2 = 0.1 * np.abs(xi_2)
err_4 = 0.1 * np.abs(xi_4)

# Plot the results
plt.figure(figsize=(10, 6))
#plt.errorbar(s_values, xi_0, yerr=err_0, fmt='o', capsize=2, label=r'$s^2\xi_0(s)$ (Monopole)', color='green')
#plt.errorbar(s_values, xi_2, yerr=err_2, fmt='x', capsize=2, label=r'$s^2\xi_2(s)$ (Quadrupole)', color='darkgreen')
#plt.errorbar(s_values, xi_4, yerr=err_4, fmt='d', capsize=2, label=r'$s^2\xi_4(s)$ (Hexadecapole)', color='lightgreen')

plt.plot(s_values, xi_0, label=r'$s^2\xi_0(s)$ (Monopole)', color='green')
plt.plot(s_values, xi_2, label=r'$s^2\xi_2(s)$ (Quadrupole)', color='darkgreen')
plt.plot(s_values, xi_4, label=r'$s^2\xi_4(s)$ (Hexadecapole)', color='lightgreen')

plt.xlabel(r'Separation $s$ ($h^{-1}$Mpc)')
plt.ylabel(r'$s^2 \xi_\ell$ ($h^{-2}$Mpc$^2$)')
plt.title('Multipole Components of the Two-Point Correlation Function')
plt.xlim(20, 200)
plt.ylim(-120, 180)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
