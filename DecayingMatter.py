import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator

# Redshift range
one_plus_z = np.logspace(0, 8, 500)

# Original components
omega_m = 0.315
omega_r = 9.02e-5
omega_lambda = 0.685

# Decaying matter parameters
omega_dm0 = 0.05         # initial density of decaying matter
Gamma = 1e-0             # decay rate (adjust to control when it decays)

# Base evolution
rho_m = (omega_m - omega_dm0) * one_plus_z**3                                   # regular matter (non-decaying)
rho_dm = omega_dm0 * one_plus_z**3 * np.exp(-Gamma * (1/one_plus_z))            # decaying matter
decayed = omega_dm0 * one_plus_z**3 * (1 - np.exp(-Gamma * (1/one_plus_z)))     # amount turned into radiation
rho_r = omega_r * one_plus_z**4 + decayed                                       # original + gained from decay
rho_lambda = np.full_like(one_plus_z, omega_lambda)                             # const. dark energy
rho_m_total = rho_m + rho_dm                                                    # total matter (stable and decaying)

# Total
rho_total = rho_m + rho_dm + rho_r + rho_lambda

# Normalized
rho_m_norm = rho_m / rho_total
rho_dm_norm = rho_dm / rho_total
rho_r_norm = rho_r / rho_total
rho_lambda_norm = rho_lambda / rho_total
rho_m_total_norm = rho_m_total / rho_total

# Plotting
plt.rcParams['font.family'] = 'Palatino'

plt.figure(figsize=(10, 8))

# Stable matter
plt.semilogx(one_plus_z, rho_m_norm, color='steelblue', linestyle='--', label=r'$\Omega_m$ (stable)')

# Decaying matter
plt.semilogx(one_plus_z, rho_dm / rho_total, 'c--', label=r'$\Omega_{\rm decay}$ (decaying matter)')

# Total matter (stable + decaying)
plt.semilogx(one_plus_z, rho_m_total_norm, 'b', label=r'$\Omega_{\rm m, total}$ (stable + decay)')

# Radiation (with decay input)
plt.semilogx(one_plus_z, rho_r_norm, 'r', label=r'$\Omega_r$ (with decay input)')

# Cosmological constant
plt.semilogx(one_plus_z, rho_lambda_norm, color='gold', label=r'$\Omega_\Lambda$')

plt.xlabel(r'1 + $\mathcal{z}$', fontsize=18)
plt.ylabel('Fraction of Total Energy Density', fontsize=18)
plt.legend(fontsize=14, frameon=False, loc='upper right')

ax = plt.gca()
ax.set_xlim(left=1, right=1e8)
ax.set_ylim(0, 1)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 1))

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=15)

plt.grid(True, which='both', linestyle='--', lw=0.5)
plt.title("Energy Density Evolution with Decaying Matter", fontsize=16)
plt.show()
