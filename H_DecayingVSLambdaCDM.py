import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
H0 = 70                         # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)   # convert km to m (x1000) and Mpc to m (3.086e22 m)

# Density parameters today
omega_m0 = 0.315
omega_baryon = 0.05
omega_dm_total = omega_m0 - omega_baryon
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_r0 - omega_m0

# Decaying matter parameters
omega_ddm0 = 0.04 * omega_dm_total    # Initial fraction of decaying dark matter - 4% of DM is DDM
Gyr = 3.154e16  # seconds in a gigayear
ddm_lifetime = 10 * Gyr  # 10 Gyr
Gamma = 1 / ddm_lifetime  # s⁻¹

# Define redshift range
z = np.logspace(0, 5, 500)          # From z = 1 to z = 100000
a = 1 / (1 + z)                     # Scale factor

# Initial conditions
rho_m0 = (omega_m0 - omega_ddm0)
rho_dm0 = omega_ddm0
rho_r0 = omega_r0

# System of equations: d(rho)/dt
def d_rho_dz(z, y):
    rho_x, rho_r = y
    rho_m = rho_m0 * (1 + z) ** 3
    rho_lambda = omega_lambda0
    rho_total = rho_m + rho_x + rho_r + rho_lambda
    H = H0_s * np.sqrt(rho_total)

    if H <= 0:
        return [0, 0]  # avoid invalid sqrt at late times

    drho_x_dz = (3 * rho_x - (Gamma / H) * rho_x) / (1 + z)
    drho_r_dz = (4 * rho_r + (Gamma / H) * rho_x) / (1 + z)
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
z = sol.t #[::-1]
rho_x = sol.y[0] #[::-1]
rho_r = sol.y[1] #[::-1]
a_vals = 1 / (1 + z)
rho_m = rho_m0 * a_vals**-3
#H_decay = H0 * np.sqrt(rho_m + rho_x + rho_r + omega_lambda0)
H_lcdm = H0 * np.sqrt(omega_m0 * a_vals**-3 + omega_r0 * a_vals**-4 + omega_lambda0)

H_decay = H0 * np.sqrt(rho_m + rho_x + rho_r + omega_lambda0)
#H_lcdm = H0 * np.sqrt(rho_m + rho_r + omega_lambda0)

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

# Compute fractional difference
delta_H_frac = (H_decay - H_lcdm) / H_lcdm

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# Plot H(z)
axes[0].loglog(1 + z_plot, H_lcdm, 'k--', label=r'$\Lambda$CDM')
axes[0].loglog(1 + z_plot, H_decay, 'b-', label='Decaying DM')
axes[0].set_xlabel(r'$1 + z$', fontsize=12)
axes[0].set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=12)
axes[0].set_title(r'Hubble Parameter $H(z)$', fontsize=14)
axes[0].legend()
axes[0].grid(True, which='both', ls='--', lw=0.5)

# Plot fractional difference
axes[1].semilogx(1 + z_plot, delta_H_frac, 'r-')
axes[1].axhline(0, color='gray', lw=0.8, ls='--')
axes[1].set_xlabel(r'$1 + z$', fontsize=12)
axes[1].set_ylabel(r'$\frac{\Delta H}{H_{\Lambda\mathrm{CDM}}}$', fontsize=14)
axes[1].set_title(r'Fractional Difference in $H(z)$', fontsize=14)
axes[1].grid(True, which='both', ls='--', lw=0.5)

plt.tight_layout()
plt.show()
