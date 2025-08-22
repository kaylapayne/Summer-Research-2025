import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from matplotlib.ticker import LogLocator, MultipleLocator

# Redshift range
one_plus_z = np.logspace(0, 8, 500)
z = one_plus_z - 1

# Constants and density parameters today
omega_m = 0.315
omega_r = 9.02e-5
omega_lambda = 0.685
omega_dm0 = 0.05
Gamma = 1e-0

# Base matter/radiation/decay modeling
rho_m = (omega_m - omega_dm0) * one_plus_z**3
rho_dm = omega_dm0 * one_plus_z**3 * np.exp(-Gamma * (1 / one_plus_z))
decayed = omega_dm0 * one_plus_z**3 * (1 - np.exp(-Gamma * (1 / one_plus_z)))

# Tired radiation behavior
mki = 1.0
mpsi = 1e-3
p_factor = np.sqrt(mki**2 / 4 - mpsi**2)
p_z = p_factor * one_plus_z / (1 + z[-1])
rho_tired = decayed * p_z / p_factor
P_tired = -decayed * p_z**3 / (3 * (p_z**2 + mpsi**2) * p_factor)
w_eff = P_tired / rho_tired
rho_tired_w = rho_tired

# Radiation including tired contribution
rho_r = omega_r * one_plus_z**4 + rho_tired_w

# Lambda and total
rho_lambda = np.full_like(one_plus_z, omega_lambda)
rho_m_total = rho_m + rho_dm
rho_total = rho_m_total + rho_r + rho_lambda

# --- Energy Density Plot ---
plt.figure(figsize=(10, 8))
plt.semilogx(one_plus_z, rho_m / rho_total, 'steelblue', ls='--', label=r'$\Omega_m$ (stable)')
plt.semilogx(one_plus_z, rho_dm / rho_total, 'c--', label=r'$\Omega_{\rm decay}$')
plt.semilogx(one_plus_z, rho_m_total / rho_total, 'b', label=r'$\Omega_{\rm m, total}$')
plt.semilogx(one_plus_z, rho_r / rho_total, 'r', label=r'$\Omega_r$ (with tired radiation)')
plt.semilogx(one_plus_z, rho_lambda / rho_total, color='gold', label=r'$\Omega_\Lambda$')
plt.xlabel(r'1 + $\mathcal{z}$', fontsize=18)
plt.ylabel('Fraction of Total Energy Density', fontsize=18)
plt.legend(fontsize=14, frameon=False)
plt.xscale('log')
plt.ylim(0, 1)
plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.title("Energy Density Evolution with Decaying Matter and Tired Radiation", fontsize=16)
plt.tight_layout()
plt.show()

# --- Hubble Parameter Plot ---
H0 = 67.74  # km/s/Mpc
H = H0 * np.sqrt(rho_total)
Hz_planck = Planck18.H(z).value

plt.figure(figsize=(9, 6))
plt.loglog(one_plus_z, H, label='Modified Model (with tired radiation)', color='blue')
plt.loglog(one_plus_z, Hz_planck, 'k--', label='Planck18 ΛCDM')
plt.xlabel(r'$1 + z$', fontsize=16)
plt.ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=16)
plt.title('Hubble Parameter Evolution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.tight_layout()
plt.show()
