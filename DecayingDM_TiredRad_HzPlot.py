import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from astropy.cosmology import Planck18

# Constants
H0 = 67.74                         # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)     # convert km/s/Mpc to s^-1

# Density parameters today
omega_m0 = 0.3153
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_r0 - omega_m0

# Decaying matter parameters
omega_ddm0 = 0.4
Gamma = 1e-16

# Tired radiation parameters
omega_tr0 = 0.8 * omega_r0         # fraction of radiation as tired radiation
omega_r0_rest = omega_r0 - omega_tr0

# Masses for decay kinematics
mki = 1.0
mpsi = 1e-3
z_i = 1e4
epsilon = 1e-6

# Redshift grid
z_eval = np.logspace(np.log10(z_i), np.log10(1e-4), 600)

# Cosmology helpers
def E(z):
    return np.sqrt(
        omega_m0 * (1 + z)**3 +
        omega_r0 * (1 + z)**4 +
        omega_lambda0
    )

def H_internal(z):
    return H0_s * E(z)

def decay_factor(z):
    integral, _ = quad(lambda zp: 1 / ((1 + zp) * E(zp)), 0, z, limit=200, epsabs=1e-10)
    return np.exp(-Gamma / H0_s * integral)

def n_ki(z):
    return omega_ddm0 * (1 + z)**3 * decay_factor(z)

# Integrals
def pressure_ddm(z):
    if z >= z_i - epsilon:
        return 0
    def integrand(z_prime):
        if z_prime <= z:
            return 0
        p = np.sqrt(mki**2 / 4 - mpsi**2) * (1 + z) / (1 + z_prime)
        Hzi = H_internal(z_prime)
        numerator = Gamma * n_ki(z_prime) * p**3
        denominator = (1 + z_prime) * Hzi * 3 * (p**2 + mpsi**2)
        return numerator / denominator
    result, _ = quad(integrand, z, z_i, limit=200, epsabs=1e-10)
    return -result

def rho_tired_integral(z):
    if z >= z_i - epsilon:
        return 0
    def integrand(z_prime):
        if z_prime <= z:
            return 0
        p = np.sqrt(mki**2 / 4 - mpsi**2) * (1 + z) / (1 + z_prime)
        Hzi = H_internal(z_prime)
        numerator = Gamma * n_ki(z_prime) * p
        denominator = (1 + z_prime) * Hzi
        return numerator / denominator
    result, _ = quad(integrand, z, z_i, limit=200, epsabs=1e-10)
    return -result

def w_tired(z):
    rho = rho_tired_integral(z)
    if abs(rho) < 1e-30:
        return 0
    P = pressure_ddm(z)
    return P / rho

# Evaluate
w_vals = np.array([w_tired(z) for z in z_eval])
rho_tr = np.array([rho_tired_integral(z) for z in z_eval])
rho_r_rest = omega_r0_rest * (1 + z_eval)**4
rho_m = omega_m0 * (1 + z_eval)**3
H_combined = H0 * np.sqrt(rho_m + rho_r_rest + rho_tr + omega_lambda0)

# Planck18 H(z)
Hz_planck = Planck18.H(z_eval).to('km/s/Mpc').value

# Plot H(z)
plt.figure(figsize=(10, 6))
plt.loglog(1 + z_eval, Hz_planck, 'k--', label=r'$\Lambda$CDM (Planck18)')
plt.loglog(1 + z_eval, H_combined, 'r-', label='With Tired Radiation Component')
plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title('H(z) with Tired Radiation from Decaying DM vs $\Lambda$CDM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()

# Plot w(z)
plt.figure(figsize=(8, 5))
plt.plot(z_eval, w_vals, label=r'$w(z) = P/\rho$', color='darkred')
plt.axhline(1/3, ls='--', color='gray', label=r'$w = 1/3$ (radiation)')
plt.axhline(0, ls=':', color='black', label=r'$w = 0$ (matter)')
plt.xlabel(r'Redshift $z$', fontsize=14)
plt.ylabel(r'Equation of State $w(z)$', fontsize=14)
plt.title('Equation of State of Tired Radiation', fontsize=16)
plt.xscale('log')
plt.legend()
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()
