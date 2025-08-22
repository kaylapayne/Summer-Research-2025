import time
import Corrfunc
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Corrfunc.theory.DDrppi import DDrppi
import astropy.cosmology

# Analysis Configuration
# data_file = 'LRG_NGC_clustering.dat.fits'
# rand_file = 'LRG_NGC_1_clustering.ran.fits'
data_file = 'BGS_BRIGHT_NGC_clustering.dat.fits'
rand_file = 'BGS_BRIGHT_NGC_0_clustering.ran.fits'
subsample_frac_data, subsample_frac_rand = 0.3, 0.7
rp_bins = np.linspace(0.1, 200, 101)  # up to 200 Mpc/h (transverse)
z_min, z_max = 0.1, 0.4

print(f"Corrfunc version: {Corrfunc.__version__}")

# Bins pi (line-of-sight and transverse separation)
pimax = 200.0                         # up to 200 Mpc/h (line-of-sight)

# Load the DESI DR1 data and associated random data for analysis
data = fitsio.read(data_file)
rand = fitsio.read(rand_file)
print(data_file, data.dtype.names)
print(rand_file, rand.dtype.names)

print("Full Dataset Sizes:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Set the redshift limits to between 0.1 and 0.3, and
# filter the data and random datasets accordingly
data_mask = (data['Z'] > z_min) & (data['Z'] < z_max)
rand_mask = (rand['Z'] > z_min) & (rand['Z'] < z_max)
data = data[data_mask]
rand = rand[rand_mask]

print(f"Subsampling for: z > {z_min:.2f}, z < {z_max:.2f})")
print("  Data N:", len(data))
print("  Random N:", len(rand))


# Subsampling function for further random subsampling
# to reduce datasets to manageable sizes.
def subsample_catalog(catalog, frac):
    n = len(catalog)
    idx = np.random.choice(n, int(frac * n), replace=False)
    return catalog[idx]


# Do the random subsampling for both data and rand
data = subsample_catalog(data, subsample_frac_data)
rand = subsample_catalog(rand, subsample_frac_rand)

print("Random subsampling to:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Convert Redshifts to Comoving Distances using the astropy library
# cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
r_data = astropy.cosmology.Planck15.comoving_distance(data['Z']).value
r_rand = astropy.cosmology.Planck15.comoving_distance(rand['Z']).value
w_data = (data['WEIGHT'] *
          #data['WEIGHT_ZFAIL'] *
          #data['WEIGHT_COMP'] *
          #data['WEIGHT_SYS'] *
          data['WEIGHT_FKP'])
w_rand = (rand['WEIGHT'] *
          #rand['WEIGHT_ZFAIL'] *
          #rand['WEIGHT_COMP'] *
          #rand['WEIGHT_SYS'] *
          rand['WEIGHT_FKP'])

print(f"r_min = {np.min(r_data):.1f} Mpc/h (z={z_min:.2f})")
print(f"r_max = {np.max(r_data):.1f} Mpc/h (z={z_max:.2f})")


def to_native_endian(arr):
    return arr.astype(np.float64).view(np.dtype('float64').newbyteorder('='))


w_data = to_native_endian(w_data)
w_rand = to_native_endian(w_rand)


# Function to convert spherical (ra,dec,r) to Cartesian coordinates
def spherical_to_cartesian(ra, dec, r):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return np.vstack((x, y, z)).T


# now convert the ra/decl/r into xyz positions
pos_data = spherical_to_cartesian(data['RA'], data['DEC'], r_data)
pos_rand = spherical_to_cartesian(rand['RA'], rand['DEC'], r_rand)

# This is the main part of the script, using the DD function of the
# Corrfunc library to build histogram results of the distances between
# data-data, data-random, and random-random datasets
t1 = time.time()
print(f"Starting DD counts at: {time.ctime(t1)},", end=" ")
# Data-data (auto-correlation)
results_DD = DDrppi(autocorr=1, nthreads=8, pimax=pimax, binfile=rp_bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    weights1=w_data, weight_type='pair_product',
                    periodic=False)

t2 = time.time()
print(f"duration: {(t2 - t1):.1f}")
print(f"Starting DR counts at: {time.ctime(t2)},", end=" ")
# Data-random (cross-correlation)
results_DR = DDrppi(autocorr=0, nthreads=8, pimax=pimax, binfile=rp_bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    X2=pos_rand[:, 0], Y2=pos_rand[:, 1], Z2=pos_rand[:, 2],
                    weights1=w_data, weights2=w_rand,
                    weight_type='pair_product',
                    periodic=False)

t3 = time.time()
print(f"duration: {(t3 - t2):.1f}")
print(f"Starting RR counts at: {time.ctime(t3)},", end=" ")
# Random-random (auto-correlation)
results_RR = DDrppi(autocorr=1, nthreads=8, pimax=pimax, binfile=rp_bins,
                    X1=pos_rand[:, 0], Y1=pos_rand[:, 1], Z1=pos_rand[:, 2],
                    weights1=w_rand,
                    weight_type='pair_product',
                    periodic=False)

t4 = time.time()
print(f"duration: {(t4 - t3):.1f}")
print(f"Completed all counts at: = {time.ctime(t4)}, total duration: {(t4 - t1):.1f}")

# This is the part of the script that normalizes the results
# from the DD calls
DD_counts = np.array([r['npairs'] * r['weightavg'] for r in results_DD])
DR_counts = np.array([r['npairs'] * r['weightavg'] for r in results_DR])
RR_counts = np.array([r['npairs'] * r['weightavg'] for r in results_RR])

norm_DD = np.sum(w_data) ** 2
norm_DR = np.sum(w_data) * np.sum(w_rand)
norm_RR = np.sum(w_rand) ** 2

DD_norm = DD_counts / norm_DD
DR_norm = DR_counts / norm_DR
RR_norm = RR_counts / norm_RR

xi_rp_pi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm

n_rp = len(rp_bins) - 1
n_pi = int(pimax)
xi_matrix = xi_rp_pi.reshape((n_rp, n_pi))

rp_centers = 0.5 * (rp_bins[:-1] + rp_bins[1:])
df = pd.DataFrame(xi_matrix, index=rp_centers)
df.to_csv('xi_rp_pi_matrix.csv')

xi_matrix = pd.read_csv('xi_rp_pi_matrix.csv', index_col=0)

# Prepare axes
rp = xi_matrix.index.values.astype(float)      # r_p bin centers
pi = xi_matrix.columns.values.astype(float)    # pi bin centers

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(xi_matrix.values, origin='lower', aspect='auto',
           extent=(pi.min(), pi.max(), rp.min(), rp.max()),
           cmap='RdBu_r', vmin=-0.05, vmax=0.1)  # Adjust vmin/vmax as needed

plt.colorbar(label=r'$\xi(r_p, \pi)$')
plt.xlabel(r'$\pi$ [Mpc/h]')
plt.ylabel(r'$r_p$ [Mpc/h]')
plt.title(r'2D Correlation Function $\xi(r_p, \pi)$')
plt.tight_layout()
plt.show()