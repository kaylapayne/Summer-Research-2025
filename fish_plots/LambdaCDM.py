import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator

'''
This is the original fish plot using the LCDM model, no decaying dark matter, just a comparison of
energy densities of matter, radiation, and dark energy over redshift history.
'''

one_plus_z = np.logspace(0, 8, 500)

omega_m = 0.315
omega_r = 9.02e-5
omega_lambda = 0.685

rho_m = omega_m * one_plus_z**3
rho_r = omega_r * one_plus_z**4
rho_lambda = np.full_like(one_plus_z, omega_lambda)

rho_total = rho_m + rho_r + rho_lambda

rho_m_norm = rho_m/rho_total
rho_r_norm = rho_r/rho_total
rho_lambda_norm = rho_lambda/rho_total

plt.rcParams['font.family'] = 'Palatino'                                        # Default is 'DejaVu Sans'

plt.figure(figsize=(10, 8))
plt.semilogx(one_plus_z, rho_m_norm, 'b', label=r'$\Omega_m(\mathcal{z})/\Omega(\mathcal{z})$')
plt.semilogx(one_plus_z, rho_r_norm, 'r', label=r'$\Omega_r(\mathcal{z})/\Omega(\mathcal{z})$')
plt.semilogx(one_plus_z, rho_lambda_norm, color='gold', label=r'$\Omega_\Lambda(\mathcal{z})/\Omega(\mathcal{z})$')
plt.xlabel(r'1 + $\mathcal{z}$', fontsize=18)
plt.ylabel('Fraction of Total Energy Density', fontsize=18)
plt.legend(fontsize=18, frameon=False, loc='upper right')

ax = plt.gca()

ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 1))


ax.set_xlim(left=1)
ax.set_ylim(bottom=0)
ax.set_ylim(top=1)
ax.set_xlim(right=10**8)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=5))
plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=10))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=15)

plt.grid(True, which='both', linestyle='--', lw=0.5)

plt.title("Energy Density Evolution without Decaying Matter", fontsize=16)

plt.show()
