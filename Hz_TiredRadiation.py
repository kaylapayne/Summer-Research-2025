import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from astropy.cosmology import Planck18, WMAP9
import astropy.units as u

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
z_trans = 5000                      # transition redshift where tired radiation converts
Gamma_t = 1e-8                    # transition rate (s^-1)

# --- Decaying DM ODE ---
rho_m0 = omega_m0 - omega_ddm0
rho_dm0 = omega_ddm0
rho_r0 = omega_r0

def d_rho_dz_decaying(z, y):
    rho_x, rho_r = y
    rho_m = rho_m0 * (1 + z) ** 3
    rho_lambda = omega_lambda0
    rho_total = rho_m + rho_x + rho_r + rho_lambda
    H = H0_s * np.sqrt(rho_total)

    if H <= 0:
        return [0, 0]

    drho_x_dz = (3 * rho_x - (Gamma / H) * rho_x) / (1 + z)
    drho_r_dz = (4 * rho_r + (Gamma / H) * rho_x) / (1 + z)
    return [drho_x_dz, drho_r_dz]

z_i = 1e4
y0_decaying = [omega_ddm0 * (1 + z_i)**3, omega_r0 * (1 + z_i)**4]
z_span = (z_i, 0.01)
z_eval = np.logspace(np.log10(z_i), np.log10(0.01), 500)

sol_decaying = solve_ivp(
    d_rho_dz_decaying, z_span, y0_decaying,
    t_eval=z_eval,
    method='BDF',
    rtol=1e-8,
    atol=1e-10
)

# Extract decaying DM solution
z_decaying = sol_decaying.t
rho_x = sol_decaying.y[0]
rho_r = sol_decaying.y[1]
a_vals_decaying = 1 / (1 + z_decaying)
rho_m_decaying = rho_m0 * a_vals_decaying**-3
H_decay = H0 * np.sqrt(rho_m_decaying + rho_x + rho_r + omega_lambda0)

# --- Tired Radiation ODE ---

def d_rho_dz_tired_rad(z, y):
    rho_t, rho_m = y
    rho_r_rest = omega_r0_rest * (1 + z)**4
    rho_lambda = omega_lambda0
    rho_total = rho_r_rest + rho_t + rho_m + rho_lambda
    H = H0_s * np.sqrt(rho_total)
    theta = 1 if z < z_trans else 0  # transition turns on below z_trans

    drho_t_dz = (4 * rho_t + (Gamma_t / H) * rho_t * theta) / (1 + z)
    drho_m_dz = (3 * rho_m - (Gamma_t / H) * rho_t * theta) / (1 + z)

    return [drho_t_dz, drho_m_dz]

# Initial conditions for tired radiation at high redshift
y0_tired = [omega_tr0 * (1 + z_i)**4, omega_m0 * (1 + z_i)**3]

sol_tired = solve_ivp(
    d_rho_dz_tired_rad, z_span, y0_tired,
    t_eval=z_eval,
    method='BDF',
    rtol=1e-8,
    atol=1e-10
)

z_tired = sol_tired.t
rho_t = sol_tired.y[0]
rho_m_tired = sol_tired.y[1]
a_vals_tired = 1 / (1 + z_tired)
rho_r_rest = omega_r0_rest * (1 + z_tired)**4
H_tired = H0 * np.sqrt(rho_r_rest + rho_t + rho_m_tired + omega_lambda0)

# --- Standard LCDM ---
a_vals = 1 / (1 + z_decaying)
H_lcdm = H0 * np.sqrt(omega_m0 * a_vals**-3 + omega_r0 * a_vals**-4 + omega_lambda0)

# Planck18 and WMAP9 from astropy
Hz_planck = Planck18.H(z_decaying).to('km/s/Mpc').value
Hz_wmap = WMAP9.H(z_decaying).to('km/s/Mpc').value

# Plot everything
plt.figure(figsize=(10, 6))
plt.loglog(1 + z_decaying, H_lcdm, 'k--', label=r'$\Lambda$CDM')
plt.loglog(1 + z_decaying, H_decay, 'b-', label='Decaying DM')
plt.loglog(1 + z_tired, H_tired, 'm-.', label='Tired Radiation')
#plt.loglog(1 + z_decaying, Hz_planck, 'r:', label='Planck18 ΛCDM')
#plt.loglog(1 + z_decaying, Hz_wmap, 'g:', label='WMAP9 ΛCDM')

plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title(r'Comparison of $H(z)$ Models: Decaying DM & Tired Radiation', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.tight_layout()
plt.show()
