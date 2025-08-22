import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from Corrfunc.theory import DD
import matplotlib.pyplot as plt

# Load DESI data
filename = "zall-pix-guadalupe.fits"
with fits.open(filename) as hdul:
    data = hdul[1].data
    ra = data['TARGET_RA']
    dec = data['TARGET_DEC']
    z = data['Z']
    weights = data['WEIGHT_FKP'] if 'WEIGHT_FKP' in data.columns.names else np.ones_like(z)

# Convert to comoving distance (Mpc/h)
r_com = cosmo.comoving_distance(z).to(u.Mpc).value

# Convert RA, DEC, z to Cartesian (xyz)
ra_rad = np.radians(ra)
dec_rad = np.radians(dec)
x = r_com * np.cos(dec_rad) * np.cos(ra_rad)
y = r_com * np.cos(dec_rad) * np.sin(ra_rad)
z_xyz = r_com * np.sin(dec_rad)

# Subsample if needed (for speed)
max_points = 500000
idx = np.random.choice(len(x), size=min(max_points, len(x)), replace=False)
x, y, z_xyz = x[idx], y[idx], z_xyz[idx]

# Create random catalog (uniform in RA/DEC/Z box)
n_random = len(x)
ra_rand = np.random.uniform(ra.min(), ra.max(), n_random)
dec_rand = np.random.uniform(dec.min(), dec.max(), n_random)
z_rand = np.random.uniform(z.min(), z.max(), n_random)

r_rand = cosmo.comoving_distance(z_rand).to(u.Mpc).value
ra_rad_rand = np.radians(ra_rand)
dec_rad_rand = np.radians(dec_rand)
x_rand = r_rand * np.cos(dec_rad_rand) * np.cos(ra_rad_rand)
y_rand = r_rand * np.cos(dec_rad_rand) * np.sin(ra_rad_rand)
z_rand_xyz = r_rand * np.sin(dec_rad_rand)

# Define bins for pair separation
r_edges = np.linspace(0, 200, 41)  # 5 Mpc/h bins
bin_centers = 0.5 * (r_edges[1:] + r_edges[:-1])

# Compute DD, DR, RR using Corrfunc
DD_counts = DD(autocorr=1, nthreads=4, binfile=r_edges,
               X1=x, Y1=y, Z1=z_xyz, boxsize=None, periodic=False)

RR_counts = DD(autocorr=1, nthreads=4, binfile=r_edges,
               X1=x_rand, Y1=y_rand, Z1=z_rand_xyz, boxsize=None, periodic=False)

DR_counts = DD(autocorr=0, nthreads=4, binfile=r_edges,
               X1=x, Y1=y, Z1=z_xyz, X2=x_rand, Y2=y_rand, Z2=z_rand_xyz, boxsize=None, periodic=False)

# Extract pair counts
DD = np.array([d['npairs'] for d in DD_counts], dtype=float)
RR = np.array([r['npairs'] for r in RR_counts], dtype=float)
DR = np.array([dr['npairs'] for dr in DR_counts], dtype=float)

# Normalize
nD = len(x)
nR = len(x_rand)
norm_DD = DD / (nD * (nD - 1) / 2)
norm_RR = RR / (nR * (nR - 1) / 2)
norm_DR = DR / (nD * nR)

# Landy–Szalay estimator
xi = (norm_DD - 2 * norm_DR + norm_RR) / norm_RR

# Plot result
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, xi, marker='o', label='DESI ξ(r)')
plt.xlabel('Separation $r$ [Mpc/h]')
plt.ylabel('ξ(r)')
plt.title('Two-Point Correlation Function (DESI, Corrfunc only)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
