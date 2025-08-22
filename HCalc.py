import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from Corrfunc.theory import DDsmu
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
c = 299792.458  # Speed of light in km/s
r_d = 147.0     # Sound horizon at drag epoch (Mpc)

# STEP 1: Load and subsample DESI data
print("Loading DESI data...")
filename = 'zall-pix-guadalupe.fits'
with fits.open(filename) as hdul:
    data = hdul[1].data
    ra = data['TARGET_RA']
    dec = data['TARGET_DEC']
    z = data['Z']

# Optional: random subsample for speed
print("Subsampling...")
N = 50000
idx = np.random.choice(len(z), N, replace=False)
ra = ra[idx]
dec = dec[idx]
z = z[idx]

# STEP 2: Convert (RA, Dec, z) to Cartesian coordinates
print("Converting coordinates...")
r_com = cosmo.comoving_distance(z).to(u.Mpc).value
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)

x = r_com * np.cos(dec_rad) * np.cos(ra_rad)
y = r_com * np.cos(dec_rad) * np.sin(ra_rad)
z_xyz = r_com * np.sin(dec_rad)

# STEP 3: Compute 2PCF using Corrfunc
print("Computing 2PCF...")
nbins = 40
s_max = 200
mu_bins = 30
r_edges = np.linspace(0, s_max, nbins + 1)

DD = DDsmu(autocorr=1, nthreads=4,
           binfile=r_edges,
           mu_max=1.0, nmu_bins=mu_bins,
           X1=x, Y1=y, Z1=z_xyz,
           periodic=False)

# STEP 4: Collapse 2D (s, μ) counts into 1D (s)
print("Collapsing to 1D radial counts...")
# Project npairs from 2D (s, mu) into 1D (s) bins
xi_counts = np.zeros(nbins)
xi_norm = np.zeros(nbins)

for i, entry in enumerate(DD):
    s_bin = i // mu_bins  # integer division: which radial bin
    xi_counts[s_bin] += entry['npairs']
    xi_norm[s_bin] += 1

# Average over mu bins
xi = xi_counts / xi_norm

# Midpoints of s bins for plotting and fitting
s_vals = 0.5 * (r_edges[:-1] + r_edges[1:])

# STEP 5: Fit a BAO peak model
def bao_model(pi, A, B, C, pi_bao, sigma):
    return A + B * pi**-2 + C * np.exp(-(pi - pi_bao)**2 / (2 * sigma**2))

initial_guess = [1.0, 1.0, 1.0, 100.0, 10.0]

print("Fitting BAO peak...")
params, covariance = curve_fit(bao_model, s_vals, xi, p0=initial_guess)
pi_bao = params[3]
H_z = c / pi_bao

# STEP 6: Plot results
print(f'BAO peak at π = {pi_bao:.2f} Mpc → H(z) ≈ {H_z:.2f} km/s/Mpc')

plt.figure(figsize=(8, 5))
plt.plot(s_vals, xi, label='Projected 2PCF', color='white')
plt.plot(s_vals, bao_model(s_vals, *params), '--', label='BAO Fit', color='cyan')
plt.xlabel('Separation π [Mpc]', color='white')
plt.ylabel('ξ(π)', color='white')
plt.title(f'BAO Peak Fit → H(z) ≈ {H_z:.2f} km/s/Mpc', color='white')
plt.legend()
plt.grid(True, color='gray', linestyle=':')
plt.gca().set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
plt.tick_params(colors='white')
plt.tight_layout()
plt.show()

