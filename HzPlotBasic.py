import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy.cosmology import Planck18 as cosmo

# Constants
H0 = 70                         # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)   # convert km to m (x1000) and Mpc to m (3.086e22 m)

# Density parameters today
omega_m0 = 0.315
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_r0 - omega_m0

# Decaying matter parameters
omega_ddm0 = 0.4 # Initial fraction of decaying dark matter
Gamma = 1e-16 # Decay rate

# Define redshift range
z = np.logspace(0, 5, 500)          # From z = 1 to z = 100000
a = 1 / (1 + z)                     # Scale factor

# Initial conditions
rho_m0 = (omega_m0 - omega_ddm0)
rho_dm0 = omega_ddm0
rho_r0 = omega_r0

# System of equations: d(rho)/dz
def d_rho_dz(z, y):
    rho_x, rho_r = y
    rho_m = rho_m0 * (1 + z) ** 3
    rho_lambda = omega_lambda0
    rho_total = rho_m + rho_x + rho_r + rho_lambda
    H = H0_s * np.sqrt(rho_total)

    if H <= 0:
        return [0, 0]  # avoid invalid sqrt at late times

    drho_x_dz = (3 * rho_x + (Gamma / H) * rho_x) / (1 + z)
    drho_r_dz = (4 * rho_r - (Gamma / H) * rho_x) / (1 + z)
    return [drho_x_dz, drho_r_dz]


z_i = 1e4
y0 = [omega_ddm0 * (1 + z_i)**3, omega_r0 * (1 + z_i)**4]

z_span = (z_i, 1)
z_eval = np.logspace(np.log10(z_i), 0, 500)

sol = solve_ivp(
    d_rho_dz, z_span, y0,
    t_eval=z_eval,
    method='BDF',  # good for stiff problems
    rtol=1e-8,
    atol=1e-10
)

# Reverse outputs to match usual 1+z descending
z = sol.t
rho_x = sol.y[0]
rho_r = sol.y[1]
a_vals = 1 / (1 + z)
rho_m = rho_m0 * a_vals**-3

# Calculate H(z) for decaying DM and LCDM models
H_decay = H0 * np.sqrt(rho_m + rho_x + rho_r + omega_lambda0)
H_lcdm = H0 * np.sqrt(omega_m0 * a_vals**-3 + omega_r0 * a_vals**-4 + omega_lambda0)

# Get H(z) from astropy cosmology Planck18 at the same z points
H_astropy = cosmo.H(z).value

# Reverse arrays so that 1 + z is increasing (for plotting)
z_plot = z[::-1]
H_decay = H_decay[::-1]
H_lcdm = H_lcdm[::-1]
H_astropy = H_astropy[::-1]

plt.figure(figsize=(10, 6))
plt.loglog(1 + z_plot, H_lcdm, 'k--', label=r'$\Lambda$CDM (analytic)')
plt.loglog(1 + z_plot, H_decay, 'b-', label='Decaying DM (numerical)')
plt.loglog(1 + z_plot, H_astropy, 'g:', label='Astropy Planck18 $H(z)$')

plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title(r'Comparison of $H(z)$ with and without decaying DM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()
