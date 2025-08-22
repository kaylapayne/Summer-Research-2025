import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM

# === Load Data ===
df = pd.read_csv('bin_correlation_data.csv')  # Use your CSV filename
s = df['Bin'].values               # separation in Mpc/h
xi = df['Correlation'].values     # correlation function

# === Define Fit Range (around expected BAO bump) ===
fit_mask = (s > 80) & (s < 140)
s_fit = s[fit_mask]
xi_fit = xi[fit_mask]

# === Define Gaussian + Linear Background ===
def bao_gaussian(s, A, s_peak, sigma, m, b):
    return A * np.exp(-0.5 * ((s - s_peak) / sigma)**2) + m * s + b

# === Fit ===
popt, pcov = curve_fit(bao_gaussian, s_fit, xi_fit, p0=[0.01, 105, 10, 0, 0])
A, s_peak, sigma, m, b = popt
s_peak_err = np.sqrt(np.diag(pcov))[1]  # uncertainty on s_peak

print(f"Radial BAO Peak: s_peak = {s_peak:.2f} ± {s_peak_err:.2f} Mpc/h")

# === Estimate H(z) ===
# Set redshift (mean of your selection range, e.g., z=0.3)
z_mean = 0.3
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
H_fid = cosmo.H(z_mean).value  # km/s/Mpc

# Sound horizon from Planck (in Mpc)
r_d = 147.1

# Estimate H(z)
H_est = r_d * H_fid / s_peak
H_est_err = r_d * H_fid / s_peak**2 * s_peak_err

print(f"Estimated H(z={z_mean:.2f}) = {H_est:.2f} ± {H_est_err:.2f} km/s/Mpc")

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(s, xi, label=r'$\xi(s)$')
plt.plot(s_fit, bao_gaussian(s_fit, *popt), 'r--', label='Gaussian Fit')
plt.axvline(s_peak, color='k', linestyle='--', label=f'$s_{{peak}}$ = {s_peak:.1f} Mpc/h')
plt.fill_between(s_fit,
                 bao_gaussian(s_fit, *popt) - sigma,
                 bao_gaussian(s_fit, *popt) + sigma,
                 color='red', alpha=0.1, label='~1σ width')
plt.xlabel("Separation $s$ [Mpc/h]")
plt.ylabel(r"$\xi(s)$")
plt.title("BAO Peak Fit and $H(z)$ Estimate")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
