import json
import os
import shutil
import time
import zipfile
from datetime import datetime
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Corrfunc.theory import DD
from Corrfunc.theory.DDrppi import DDrppi
import astropy.cosmology
from strenum import StrEnum
import scipy.integrate


# Enum to indicate which mode this script should run in.
# mode is set in settings below.
class RunMode(StrEnum):
    ISOTROPIC = 'ISOTROPIC'
    ANISOTROPIC = 'ANISOTROPIC'


# --------------- ANALYSIS SETTINGS -----------------
# data_file = 'LRG_NGC_clustering.dat.fits'
# rand_file = 'LRG_NGC_1_clustering.ran.fits'
#data_file = 'LRG_N_clustering.dat.fits'
#rand_file = 'LRG_N_0_clustering.ran.fits'
data_file = 'ELG_LOPnotqso_NGC_clustering.dat.fits'
rand_file = 'ELG_LOPnotqso_NGC_0_clustering.ran.fits'
subsample_frac_data, subsample_frac_rand = 1, 1
edge_start, edge_end, edge_num = 20, 200, 46
z_min, z_max = 1.4, 1.6
weight_components = ['WEIGHT', 'WEIGHT_FKP']
run_mode = RunMode.ISOTROPIC
cpu_cores = 10
# --------------- ANALYSIS SETTINGS -----------------

# pimax is used to bin pi (line-of-sight and transverse separation, set to 1 for iso)
pimax = 200.0 if run_mode==RunMode.ANISOTROPIC else 1.0  # up to 200 Mpc/h (line-of-sight)
h = 0.674

print( f'Run Mode: {run_mode}')

# Load the DESI DR1 data and associated random data for analysis
data = fitsio.read(f'data/{data_file}')
rand = fitsio.read(f'data/{rand_file}')
print(data_file, data.dtype.names)
print(rand_file, rand.dtype.names)

print("Full Dataset Sizes:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# perform redshift (z) cuts per z_min, z_max configs
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


# Do the random subsampling cuts both data and rand
data = subsample_catalog(data, subsample_frac_data)
rand = subsample_catalog(rand, subsample_frac_rand)

print("Random subsampling to:")
print("  Data N:", len(data))
print("  Random N:", len(rand))

# Convert Redshifts to co-moving distances using the astropy library
# also multiply by h, since Corrfunc expects units of Mpc/h
r_data = astropy.cosmology.Planck15.comoving_distance(data['Z']).value * h
r_rand = astropy.cosmology.Planck15.comoving_distance(rand['Z']).value * h

# build up the weightings based on assigned weight components in config
w_data = np.ones_like(data['WEIGHT'])
w_rand = np.ones_like(rand['WEIGHT'])
for weight_component in weight_components:
    w_data *= data[weight_component]
    w_rand *= rand[weight_component]

print(f"r_min = {np.min(r_data):.1f} Mpc/h (z={z_min:.2f})")
print(f"r_max = {np.max(r_data):.1f} Mpc/h (z={z_max:.2f})")


# adjust byte ordering for data doing directly to the Corrfunc library
# (e.g. weighting data).  Don't need to convert other data as it's
# converted to position from source RA,ASC,Z values from fits files.
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


# convert the ra/decl/r into xyz positions
pos_data = spherical_to_cartesian(data['RA'], data['DEC'], r_data)
pos_rand = spherical_to_cartesian(rand['RA'], rand['DEC'], r_rand)

# use the edge settings from config to build an array of bins for Corrfunc
bins = np.linspace(edge_start, edge_end, edge_num)


# This is the main part of the script, using the DD(isotropic) or DDrppi(anisotropic)
# functions of the Corrfunc library to build histogram results of the distances between
# data-data, data-random, and random-random datasets

# Data-data (auto-correlation): use DDrppi for anisotropic, and DD for isotropic
t1 = time.time()
print(f"Starting DD counts at: {time.ctime(t1)},", end=" ")
results_DD = DDrppi(autocorr=1, nthreads=cpu_cores, pimax=pimax, binfile=bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    weights1=w_data, weight_type='pair_product',
                    periodic=False) if run_mode==RunMode.ANISOTROPIC else \
                 DD(autocorr=1, nthreads=cpu_cores, binfile=bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    weights1=w_data, weight_type='pair_product',
                    periodic=False)

# Data-random (cross-correlation): use DDrppi for anisotropic, and DD for isotropic
t2 = time.time()
print(f"duration: {(t2 - t1):.1f}")
print(f"Starting DR counts at: {time.ctime(t2)},", end=" ")
results_DR = DDrppi(autocorr=0, nthreads=cpu_cores, pimax=pimax, binfile=bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    X2=pos_rand[:, 0], Y2=pos_rand[:, 1], Z2=pos_rand[:, 2],
                    weights1=w_data, weights2=w_rand,
                    weight_type='pair_product',
                    periodic=False) if run_mode==RunMode.ANISOTROPIC else \
                 DD(autocorr=0, nthreads=cpu_cores, binfile=bins,
                    X1=pos_data[:, 0], Y1=pos_data[:, 1], Z1=pos_data[:, 2],
                    X2=pos_rand[:, 0], Y2=pos_rand[:, 1], Z2=pos_rand[:, 2],
                    weights1=w_data,
                    weights2=w_rand,
                    weight_type='pair_product',
                    periodic=False)

# Random-random (auto-correlation): use DDrppi for anisotropic, and DD for isotropic
t3 = time.time()
print(f"duration: {(t3 - t2):.1f}")
print(f"Starting RR counts at: {time.ctime(t3)},", end=" ")
results_RR = DDrppi(autocorr=1, nthreads=cpu_cores, pimax=pimax, binfile=bins,
                    X1=pos_rand[:, 0], Y1=pos_rand[:, 1], Z1=pos_rand[:, 2],
                    weights1=w_rand,
                    weight_type='pair_product',
                    periodic=False) if run_mode==RunMode.ANISOTROPIC else \
                 DD(autocorr=1, nthreads=cpu_cores, binfile=bins,
                    X1=pos_rand[:, 0], Y1=pos_rand[:, 1], Z1=pos_rand[:, 2],
                    weights1=w_rand,
                    weight_type='pair_product',
                    periodic=False)

t4 = time.time()
print(f"duration: {(t4 - t3):.1f}")
print(f"Completed all counts at: = {time.ctime(t4)}, total duration: {(t4 - t1):.1f}")

# normalize the results from the DD, DDrppi calls:
DD_counts = np.array([r['npairs'] * r['weightavg'] for r in results_DD])
DR_counts = np.array([r['npairs'] * r['weightavg'] for r in results_DR])
RR_counts = np.array([r['npairs'] * r['weightavg'] for r in results_RR])

norm_DD = np.sum(w_data) ** 2
norm_DR = np.sum(w_data) * np.sum(w_rand)
norm_RR = np.sum(w_rand) ** 2

DD_norm = DD_counts / norm_DD
DR_norm = DR_counts / norm_DR
RR_norm = RR_counts / norm_RR

# Apply the Landy-Szalay estimator to build normalized results:
xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm

# Reshape data into xi matrix and write to file for later
# analysis and plotting.
n_rp = len(bins) - 1
n_pi = int(pimax)
xi_matrix = xi.reshape((n_rp, n_pi))
bin_centers = 0.5 * (bins[:-1] + bins[1:])


def get_folder_name():
    # Get the current date and time to create a folder name for the results
    now = datetime.now()
    iso_aniso="ISOTROPIC" if run_mode == RunMode.ISOTROPIC else "ANISOTROPIC"
    folder = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}__{z_min}-{z_max}__{len(data)}'
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{data_file}", exist_ok=True)
    os.makedirs(f"results/{data_file}/{iso_aniso}", exist_ok=True)
    os.makedirs(f"results/{data_file}/{iso_aniso}/{folder}", exist_ok=True)
    return f"results/{data_file}/{iso_aniso}/{folder}"


def zip_folder(folder_path, zip_name):
    if not any(os.scandir(folder_path)):
        print("The folder is empty.")
    else:
        print("The folder contains files.")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                print(f"Adding {file_path} as {arcname}")  # Debugging line
                zipf.write(file_path, arcname)
    print(f"Folder '{folder_path}' has been zipped as '{zip_name}'.")


# writes results into subfolders of the results folder.  The results
# folder will be created if not already there.  Results are further
# organized into data_file named sub-folders for easier future access
def write_results_to_file(folder_name, xi_matrix, centers ):

    # create a settings key/value data structure to write to json to record settings.
    settings = {
        'file_data': data_file,
        'file_rand': rand_file,
        'z_min': z_min,
        'z_max': z_max,
        'bin_size': (edge_end-edge_start)/(edge_num-1),
        'weight_components': ', '.join(weight_components),
        'size_data': len(data),
        'size_rand': len(rand),
        'subsample_frac_data': subsample_frac_data,
        'subsample_frac_rand': subsample_frac_rand,
        'duration_dd': (t2 - t1),
        'duration_dr': (t3 - t2),
        'duration_rr': (t4 - t3),
        'duration_total': (t4 - t1),
        'run_mode': run_mode
    }

    # write the settings to the results folder as a json file to the results folder
    with open(f'{folder_name}/settings.json', 'w') as f:
        json.dump(settings, f, indent=4, sort_keys=True)

    # write the dataframe as a csv to the results folder
    df = pd.DataFrame(xi_matrix, index=centers)
    df.to_csv(f'{folder_name}/xi_matrix.csv')


# function to zip up all results in results directory
def zip_results():
    # zip up the results folder
    folder_to_zip = folder_name
    zip_file_name = f'results/{data_file}/results.zip'
    zip_folder(folder_to_zip, zip_file_name)
    shutil.move(f'results/{data_file}/results.zip', folder_name)


# save and reload results for initial plotting
folder_name = get_folder_name()
write_results_to_file(folder_name, xi_matrix, bin_centers)

# now do basic plotting (mostly to just indicate processing is done
if run_mode==RunMode.ISOTROPIC:
    xi_s2 = xi * bin_centers ** 2
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, xi_s2, label='ξ(s)')
    plt.axvline(105, color='red', linestyle='--', label='BAO scale (~105 Mpc/h)')
    plt.xlabel("Separation s [Mpc/h]")
    plt.ylabel( r'Correlation $s^2 \xi(s)$')
    plt.grid(True)
    plt.legend()
    plt.title(f"Two-point Correlation Function: {data_file}")
    plt.savefig(f'{folder_name}/plot.png', dpi=300, bbox_inches='tight')
    zip_results()
    plt.show()

elif run_mode==RunMode.ANISOTROPIC:
    xi_matrix = pd.read_csv(f'{folder_name}/xi_matrix.csv', index_col=0)

    # Setup rp and pi axes for plotting
    rp = xi_matrix.index.values.astype(float)  # r_p bin centers
    pi = xi_matrix.columns.values.astype(float)  # pi bin centers

    # Plot heatmap of xi_matrix data
    plt.figure(figsize=(8, 6))
    plt.imshow(xi_matrix.values, origin='lower', aspect='auto',
               extent=(pi.min(), pi.max(), rp.min(), rp.max()),
               cmap='RdBu_r', vmin=-0.05, vmax=0.1)  # Adjust vmin/vmax as needed

    plt.colorbar(label=r'$\xi(r_p, \pi)$')
    plt.xlabel(r'$\pi$ [Mpc/h]')
    plt.ylabel(r'$r_p$ [Mpc/h]')
    plt.title(r'2D Correlation Function $\xi(r_p, \pi)$')
    plt.tight_layout()
    plt.savefig(f'{folder_name}/plot.png', dpi=300, bbox_inches='tight')
    zip_results()
    plt.show()