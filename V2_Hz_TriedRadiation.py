import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
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
Gamma_t = 1e-8                      # transition rate (s^-1)

# Masses for decay kinematics
mki = 1.0
mpsi = 1e-3
z_i = 1e4
epsilon = 1e-6                     # small buffer to avoid edge integration problems

# Redshift grid (extended lower bound)
z_span = (z_i, 1e-4)
z_eval = np.logspace(np.log10(z_i), np.log10(1e-4), 600)

# E(z) for use in decay factor and H(z) integrals
def E(z):
    return np.sqrt(
        omega_m0 * (1 + z)**3 +
        omega_r0 * (1 + z)**4 +
        omega_lambda0
    )

# H(z) in s^-1
def H_internal(z):
    return H0_s * E(z)

# Exponential decay suppression of number density
def decay_factor(z):
    integral, _ = quad(lambda zp: 1 / ((1 + zp) * E(zp)), 0, z, limit=200, epsabs=1e-10)
    return np.exp(-Gamma / H0_s * integral)

# Number density with decay
def n_ki(z):
    return omega_ddm0 * (1 + z)**3 * decay_factor(z)

# Pressure integral for tired radiation
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

# Energy density integral for tired radiation
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

# w(z) = P / rho
def w_tired(z):
    rho = rho_tired_integral(z)
    if abs(rho) < 1e-30:
        return 0
    P = pressure_ddm(z)
    return P / rho

# Evaluate w(z), rho(z), and H(z)
w_vals = np.array([w_tired(z) for z in z_eval])
rho_t = np.array([rho_tired_integral(z) for z in z_eval])
rho_m_tired = omega_m0 * (1 + z_eval)**3
rho_r_rest = omega_r0_rest * (1 + z_eval)**4
H_tired = H0 * np.sqrt(rho_r_rest + rho_t + rho_m_tired + omega_lambda0)

# Decaying DM ODE
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

y0_decaying = [omega_ddm0 * (1 + z_i)**3, omega_r0 * (1 + z_i)**4]
sol_decaying = solve_ivp(d_rho_dz_decaying, z_span, y0_decaying, t_eval=z_eval, method='BDF', rtol=1e-8, atol=1e-10)
z_decaying = sol_decaying.t
rho_x = sol_decaying.y[0]
rho_r = sol_decaying.y[1]
a_vals_decaying = 1 / (1 + z_decaying)
rho_m_decaying = rho_m0 * a_vals_decaying**-3
H_decay = H0 * np.sqrt(rho_m_decaying + rho_x + rho_r + omega_lambda0)

# LCDM baseline
a_vals = 1 / (1 + z_decaying)
H_lcdm = H0 * np.sqrt(omega_m0 * a_vals**-3 + omega_r0 * a_vals**-4 + omega_lambda0)
Hz_planck = Planck18.H(z_decaying).to('km/s/Mpc').value
Hz_wmap = WMAP9.H(z_decaying).to('km/s/Mpc').value

# Plot H(z)
plt.figure(figsize=(10, 6))
plt.loglog(1 + z_decaying, H_lcdm, 'k--', label=r'$\Lambda$CDM')
plt.loglog(1 + z_decaying, H_decay, 'b-', label='Decaying DM')
plt.loglog(1 + z_eval, H_tired, 'm-.', label='Tired Radiation (from integral)')
plt.xlabel(r'$1 + z$', fontsize=14)
plt.ylabel(r'Hubble Parameter $H(z)$ [km/s/Mpc]', fontsize=14)
plt.title(r'Comparison of $H(z)$ Models', fontsize=16)
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
