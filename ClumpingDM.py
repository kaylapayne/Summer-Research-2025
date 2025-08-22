import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
H0 = 70                         # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)   # H0 in 1/s

# Density parameters today
omega_m0 = 0.315
omega_r0 = 9.02e-5
omega_lambda0 = 1 - omega_m0 - omega_r0

# Split matter for DDM case
omega_ddm0 = 0.4               # Fraction of matter that's decaying
rho_m0 = omega_m0 - omega_ddm0
rho_dm0 = omega_ddm0
rho_r0 = omega_r0

# Decay constant
Gamma = 1e-16

# Clumping parameters
alpha = 1e-30
m_mediator = 1e-28
beta = 1e-8  # much smaller value to avoid imaginary H^2

# Redshift range
z_i = 1e4
z_eval = np.logspace(np.log10(z_i), 0, 500)
a = 1 / (1 + z_eval)

# ΛCDM H(z)
H_lcdm = H0 * np.sqrt(
    omega_m0 * a**-3 + omega_r0 * a**-4 + omega_lambda0
)

# Decaying DM: evolve rho_x and rho_r
y0 = [rho_dm0 * (1 + z_i)**3, rho_r0 * (1 + z_i)**4]

def d_rho_dz(z, y):
    rho_x, rho_r = y
    rho_m = rho_m0 * (1 + z)**3
    rho_lambda = omega_lambda0
    rho_total = rho_m + rho_x + rho_r + rho_lambda
    H = H0_s * np.sqrt(rho_total)
    if H <= 0:
        return [0, 0]
    drho_x_dz = (3 * rho_x - (Gamma / H) * rho_x) / (1 + z)
    drho_r_dz = (4 * rho_r + (Gamma / H) * rho_x) / (1 + z)
    return [drho_x_dz, drho_r_dz]

sol = solve_ivp(
    d_rho_dz, (z_i, 1), y0,
    t_eval=z_eval, method='BDF',
    rtol=1e-8, atol=1e-10
)

z = sol.t
a_vals = 1 / (1 + z)
rho_x = np.maximum(sol.y[0], 1e-30)
rho_r = np.maximum(sol.y[1], 1e-30)
rho_m_ddm = rho_m0 * a_vals**-3

H_ddm = H0 * np.sqrt(
    rho_m_ddm + rho_x + rho_r + omega_lambda0
)

# Clumping DM: same rho_m as ΛCDM, but add interaction term
rho_m_clump = omega_m0 * a**-3
rho_int = -beta * rho_m_clump**2

# Total energy density in clumping model
total_energy_density = rho_m_clump + omega_r0 * a**-4 + omega_lambda0 + rho_int

# Check for negatives
H2_clump = np.where(total_energy_density > 0, total_energy_density, np.nan)
H_clump = H0 * np.sqrt(H2_clump)

# Plotting
plt.figure(figsize=(10, 6))

plt.loglog(1 + z, H_ddm, 'slateblue', label='Decaying DM')
plt.loglog(1 + z, H_lcdm, 'k', dashes=(6, 6), label=r'$\Lambda$CDM')
plt.loglog(1 + z_eval, H_clump, 'r', dashes=(1, 2), label='Clumping DM')

plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title(r'Comparison of $H(z)$ in Different Cosmological Models', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(1 + z_eval, total_energy_density, label='Total Energy Density (Clumping)')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel(r'$1 + z$')
plt.ylabel('Total Energy Density')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.title('Check: Energy Density in Clumping Model')
plt.show()
