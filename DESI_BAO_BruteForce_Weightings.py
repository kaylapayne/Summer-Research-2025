import time
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
subsample_frac_data = 0.1    # Subsample 10% of data
subsample_frac_rand = 0.3    # Subsample 10% of randoms
n_jobs = 20                   # Number of CPUs for parallel execution
bins = np.linspace(1, 200, 400)  # 1 Mpc/h resolution
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# === Load Data ===
data = fitsio.read('BGS_ANY_NGC_clustering.dat.fits')
rand = fitsio.read('BGS_ANY_NGC_1_clustering.ran.fits')

# Select z ~ 0.1 to 0.5
zmin, zmax = 0.1, 0.5
data_mask = (data['Z'] > zmin) & (data['Z'] < zmax)
rand_mask = (rand['Z'] > zmin) & (rand['Z'] < zmax)

data = data[data_mask]
rand = rand[rand_mask]

# === Subsampling ===
def subsample_catalog(catalog, frac):
    n = len(catalog)
    idx = np.random.choice(n, int(frac * n), replace=False)
    return catalog[idx]

print("Data N:", len(data))
print("Random N:", len(rand))

data = subsample_catalog(data, subsample_frac_data)
rand = subsample_catalog(rand, subsample_frac_rand)

print("Data N:", len(data))
print("Random N:", len(rand))

# === Convert Redshifts to Comoving Distances ===
r_data = cosmo.comoving_distance(data['Z']).value
r_rand = cosmo.comoving_distance(rand['Z']).value

# === Spherical to Cartesian ===
def spherical_to_cartesian(ra, dec, r):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z = r * np.sin(dec_rad)
    return np.vstack((x, y, z)).T


def pair_counts_chunk(pos1_chunk, w1_chunk, pos2, w2, bins, start_idx=0, symmetric=False):
    tree2 = cKDTree(pos2)
    hist = np.zeros(len(bins) - 1)

    for i, (p1, w1) in enumerate(zip(pos1_chunk, w1_chunk)):
        global_i = start_idx + i
        idxs = tree2.query_ball_point(p1, r=bins[-1])
        if symmetric:
            # Avoid double counting by only taking pairs where index j > i (global)
            idxs = [j for j in idxs if j > global_i]
        if len(idxs) > 0:
            dists = np.linalg.norm(pos2[idxs] - p1, axis=1)
            weights = w1 * w2[idxs]
            hist += np.histogram(dists, bins=bins, weights=weights)[0]

    return hist

def parallel_pair_counts(pos1, w1, pos2, w2, bins, symmetric=False, n_jobs=4):
    chunks = np.array_split(pos1, n_jobs)
    w_chunks = np.array_split(w1, n_jobs)
    results = []
    offsets = np.cumsum([0] + [len(chunk) for chunk in chunks[:-1]])

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(pair_counts_chunk, chunk, w_chunk, pos2, w2, bins, offset, symmetric)
                   for chunk, w_chunk, offset in zip(chunks, w_chunks, offsets)]

        for future in as_completed(futures):
            results.append(future.result())

    total_hist = sum(results)
    return total_hist


pos_data = spherical_to_cartesian(data['RA'], data['DEC'], r_data)
pos_rand = spherical_to_cartesian(rand['RA'], rand['DEC'], r_rand)
w_data = data['WEIGHT']  # replace with your actual weights array for data
w_rand = rand['WEIGHT']  # replace with weights for random catalog, typically all 1

print("Computing pair counts with weights...")

print("Starting DD counts at:", time.ctime(time.time()))
DD = parallel_pair_counts(pos_data, w_data, pos_data, w_data, bins, symmetric=True, n_jobs=15)
print("Starting DR counts at:", time.ctime(time.time()))
DR = parallel_pair_counts(pos_data, w_data, pos_rand, w_rand, bins, symmetric=False, n_jobs=15)
print("Starting RR counts at:", time.ctime(time.time()))
RR = parallel_pair_counts(pos_rand, w_rand, pos_rand, w_rand, bins, symmetric=True, n_jobs=15)
print("Completed all counts at:", time.ctime(time.time()))

# === Normalization with weights ===
sum_wd = np.sum(w_data)
sum_wr = np.sum(w_rand)
sum_wd2 = np.sum(w_data ** 2)
sum_wr2 = np.sum(w_rand ** 2)

norm_DD = (sum_wd ** 2 - sum_wd2) / 2
norm_RR = (sum_wr ** 2 - sum_wr2) / 2
norm_DR = sum_wd * sum_wr

DD_norm = DD / norm_DD
RR_norm = RR / norm_RR
DR_norm = DR / norm_DR

# === Landy–Szalay Estimator ===
xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, xi, label=r'$\xi(s)$', drawstyle='steps-mid')
plt.axhline(0, color='gray', ls='--')
plt.xlabel("Separation $s$ [Mpc/h]")
plt.ylabel(r'$\xi(s)$')
plt.title("BAO Correlation Function")
plt.xlim(0, 200)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
