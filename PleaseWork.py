import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
from scipy.integrate import solve_ivp

# Constants
H0 = 67.74                          # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)       # convert km to m (x1000) and Mpc to m (3.086e22 m)

# Density parameters today
omega_m0 = 0.3153
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_r0 - omega_m0

# Decaying matter parameters - mess with these
omega_ddm0 = 0.4        # Initial fraction of decaying dark matter
Gamma = 1e-16 #1e-3 * H0_s  # Make decay proportional to H0

# Define redshift range
z = np.logspace(0, 5, 500)          # From z = 1 to z = 100000
a = 1 / (1 + z)                     # Scale factor

# Initial conditions
rho_m0 = (omega_m0 - omega_ddm0)
rho_dm0 = omega_ddm0
rho_r0 = omega_r0
z_i = 1e4
y0 = [omega_ddm0 * (1 + z_i)**3, omega_r0 * (1 + z_i)**4, 0.0]

# System of equations: d(rho)/dt
def d_rho_dz(z, y):
    rho_x, rho_r, rho_tracked = y
    rho_m = rho_m0 * (1 + z) ** 3
    rho_lambda = omega_lambda0
    rho_total = rho_m + rho_x + rho_r + rho_lambda
    if rho_total <= 0:
        return [0, 0, 0]    # avoid invalid sqrt at late times
    H = H0_s * np.sqrt(rho_total)

    decay_term = (Gamma / H) * rho_x / (1 + z)
    drho_x_dz = (3 * rho_x - (Gamma / H) * rho_x) / (1 + z)
    drho_r_dz = (4 * rho_r + (Gamma / H) * rho_x) / (1 + z)
    dtracked_dz = decay_term
    return [drho_x_dz, drho_r_dz, dtracked_dz]

# Solve the system
z_span = (z_i - 1, 1)  # e.g., from 9999 → 1
z_eval = np.logspace(np.log10(z_i - 1), 0, 500)
sol = solve_ivp(
    d_rho_dz, z_span, y0,
    t_eval=z_eval,
    method='BDF',  # good for stiff problems
    rtol=1e-6,
    atol=1e-9
)

# Check if ODE solver worked
if not sol.success:
    print("ODE solver failed:", sol.message)
    exit()

# Extract solution
z = sol.t
z = np.array(z)
a_vals = 1 / (1 + z)
rho_x = sol.y[0]
rho_r = sol.y[1]
rho_tracked = sol.y[2]
rho_m = rho_m0 * a_vals**-3
H_lcdm = H0 * np.sqrt(omega_m0 * a_vals**-3 + omega_r0 * a_vals**-4 + omega_lambda0)
H_decay = H0 * np.sqrt(rho_m + rho_x + rho_r + omega_lambda0)

# Reverse arrays so that 1 + z is increasing
z_plot = z
a_vals = a_vals[::-1]

rho_x = np.maximum(rho_x, 1e-30)
rho_r = np.maximum(rho_r, 1e-30)

print("z range:", z.min(), z.max())
print("H_decay (min, max):", np.nanmin(H_decay), np.nanmax(H_decay))
print("H_lcdm (min, max):", np.nanmin(H_lcdm), np.nanmax(H_lcdm))
print("Any NaNs in H_decay?", np.any(np.isnan(H_decay)))
print("Any NaNs in H_lcdm?", np.any(np.isnan(H_lcdm)))

# Plot H(z)
plt.figure(figsize=(10, 6))
plt.loglog(1 + z_plot, H_lcdm, 'k--', label=r'$\Lambda$CDM')
plt.loglog(1 + z_plot, H_decay, 'b-', label='Decaying DM')
plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title(r'Comparison of $H(z)$ with and without decaying DM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()

# Plot tracked radiation from decay
plt.figure(figsize=(10, 6))
plt.plot(z, rho_tracked / (1 + z)**4, 'r-', label='Tracked radiation density (normalized)')
plt.xlabel(r'Redshift $z$', fontsize=14)
plt.ylabel(r'Tracked Radiation Energy Density', fontsize=14)
plt.title('Tracked Radiation Created by Decaying DM', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()