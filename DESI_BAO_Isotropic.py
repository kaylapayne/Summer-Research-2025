import time
import Corrfunc
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Corrfunc.theory.DD import DD
from astropy.cosmology import Planck15 as cosmo

plt.rcParams['text.usetex'] = True

# Analysis Configuration
# data_file = 'LRG_NGC_clustering.dat.fits'
# rand_file = 'LRG_NGC_1_clustering.ran.fits'
data_file = 'BGS_BRIGHT_NGC_clustering.dat.fits'
rand_file = 'BGS_BRIGHT_NGC_0_clustering.ran.fits'
subsample_frac_data, subsample_frac_rand = 0.5, 1
z_min, z_max = 0.1, 0.4
edge_start, edge_end, edge_num = 20, 200, 91

print(f"Corrfunc version: {Corrfunc.__version__}")

# Create and save a bin file with bins for accumulating distances
# this is used by the Corrfunc library calls
bin_file = "bins.txt"
print(f"Bins: start={edge_start}, end={edge_end}, bin size={(edge_end - edge_start) / (edge_num - 1):.1f}")
edges = np.linspace(edge_start, edge_end, edge_num)
with open(bin_file, "w") as f:
    for i in range(len(edges) - 1):
        f.write(f"{edges[i]:.8f} {edges[i + 1]:.8f}\n")

# Load the DESI DR1 data and associated random data for analysis
data = fitsio.read(data_file)
rand = fitsio.read(rand_file)
print(data_file, data.dtype.names)
print(rand_file, rand.dtype.names)

print("Full Dataset Sizes:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Perform redshift cuts
data_mask = (data['Z'] > z_min) & (data['Z'] < z_max)
rand_mask = (rand['Z'] > z_min) & (rand['Z'] < z_max)
data = data[data_mask]
rand = rand[rand_mask]

print(f"Data cuts for: z > {z_min:.2f}, z < {z_max:.2f})")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Subsampling function for further random cuts
# to reduce datasets to manageable sizes.
def subsample_catalog(catalog, frac):
    n = len(catalog)
    idx = np.random.choice(n, int(frac * n), replace=False)
    return catalog[idx]


# Do the random subsampling for both data and rand
data = subsample_catalog(data, subsample_frac_data)
rand = subsample_catalog(rand, subsample_frac_rand)

print("Random subsampling cuts to:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Convert Redshifts to Comoving Distances using the astropy library
# cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
r_data = cosmo.comoving_distance(data['Z']).value
r_rand = cosmo.comoving_distance(rand['Z']).value

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
results_DD = DD(autocorr=1, nthreads=8, binfile=bin_file,
                X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                weights1=w_data, weight_type='pair_product',
                periodic=False)

t2 = time.time()
print(f"duration: {(t2 - t1):.1f}")
print(f"Starting DR counts at: {time.ctime(t2)},", end=" ")
# Data-random (cross-correlation)
results_DR = DD(autocorr=0, nthreads=8, binfile=bin_file,
                X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                X2=pos_rand[:, 0], Y2=pos_rand[:, 1], Z2=pos_rand[:, 2],
                weights1=w_data,
                weights2=w_rand,
                weight_type='pair_product',
                periodic=False)

t3 = time.time()
print(f"duration: {(t3 - t2):.1f}")
print(f"Starting RR counts at: {time.ctime(t3)},", end=" ")
# Random-random (auto-correlation)
results_RR = DD(autocorr=1, nthreads=8, binfile=bin_file,
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

xi_ls = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm

bin_centers = 0.5 * (edges[:-1] + edges[1:])

xi_ls_s2 = xi_ls * bin_centers**2

# Create a DataFrame for saving
df = pd.DataFrame({
    'Bin': bin_centers,
    'Correlation': xi_ls_s2
})
# Save to CSV
df.to_csv('bin_correlation_data.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(bin_centers, xi_ls_s2, label='ξ(s)')
plt.axvline(105, color='red', linestyle='--', label='BAO scale (~105 Mpc/h)')
plt.xlabel("Separation s [Mpc/h]")
plt.ylabel=r'Correlation $\mathbf{s^2 \xi(s)}$'
#plt.ylim(-100, 100)  # zoom in on realistic BAO scale
plt.grid(True)
plt.legend()
plt.title(f"Two-point Correlation Function: {data_file}")
plt.show()