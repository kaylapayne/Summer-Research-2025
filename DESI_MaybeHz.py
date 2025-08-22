import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from Corrfunc.theory import DD
from scipy.optimize import curve_fit

# Load DESI data
filename = "zall-pix-guadalupe.fits"
with fits.open(filename) as hdul:
    data = hdul[1].data
    ra_all = data['TARGET_RA']
    dec_all = data['TARGET_DEC']
    z_all = data['Z']

# Redshift bins
z_bins = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1)]

# Fiducial sound horizon
r_d_fid = 147.0  # Mpc

# Convert r_edges to Mpc/h
r_edges = np.linspace(0, 200, 41)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

# Get h from Planck18 cosmology
h = cosmo.H0.value / 100
r_d_fid_h = r_d_fid * h  # Convert sound horizon to Mpc/h

# Convert r_edges and r_centers to Mpc/h
r_edges_h = r_edges * h
r_centers_h = r_centers * h

bao_peaks = []
Hz_vals = []
z_mid_vals = []

# Gaussian + baseline model for fitting BAO peak
def gaussian_model(r, A, r0, sigma, C):
    return A * np.exp(-(r - r0) ** 2 / (2 * sigma ** 2)) + C

for z_min, z_max in z_bins:
    print(f"Processing bin {z_min} < z < {z_max}")

    mask = (z_all > z_min) & (z_all < z_max)
    if np.sum(mask) < 1000:
        print(f"Skipping bin {z_min} < z < {z_max} due to insufficient data points.")
        continue

    ra = ra_all[mask]
    dec = dec_all[mask]
    z = z_all[mask]

    # Convert to comoving distance in Mpc/h
    r_com = cosmo.comoving_distance(z).to(u.Mpc).value * h
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # Convert spherical to Cartesian
    x = r_com * np.cos(dec_rad) * np.cos(ra_rad)
    y = r_com * np.cos(dec_rad) * np.sin(ra_rad)
    z_xyz = r_com * np.sin(dec_rad)

    # Subsample data for performance
    max_points = 50000
    idx = np.random.choice(len(x), min(max_points, len(x)), replace=False)
    x, y, z_xyz = x[idx], y[idx], z_xyz[idx]

    # Generate random catalog with 5x number of data points
    n_rand = 5 * len(x)

    # Sample random redshifts from data's redshift distribution
    z_rand = np.random.choice(z, size=n_rand, replace=True)
    ra_rand = np.random.uniform(ra.min(), ra.max(), n_rand)
    dec_rand = np.random.uniform(dec.min(), dec.max(), n_rand)

    r_rand = cosmo.comoving_distance(z_rand).to(u.Mpc).value * h
    ra_rad_rand = np.radians(ra_rand)
    dec_rad_rand = np.radians(dec_rand)

    x_rand = r_rand * np.cos(dec_rad_rand) * np.cos(ra_rad_rand)
    y_rand = r_rand * np.cos(dec_rad_rand) * np.sin(ra_rad_rand)
    z_rand_xyz = r_rand * np.sin(dec_rad_rand)

    # Calculate pair counts with Corrfunc
    DD_counts = DD(1, 4, r_edges_h, x, y, z_xyz, periodic=False)
    RR_counts = DD(1, 4, r_edges_h, x_rand, y_rand, z_rand_xyz, periodic=False)
    DR_counts = DD(0, 4, r_edges_h, x, y, z_xyz, X2=x_rand, Y2=y_rand, Z2=z_rand_xyz, periodic=False)

    DD_arr = np.array([row['npairs'] for row in DD_counts], dtype=float)
    RR_arr = np.array([row['npairs'] for row in RR_counts], dtype=float)
    DR_arr = np.array([row['npairs'] for row in DR_counts], dtype=float)

    # Normalize pair counts
    norm_DD = DD_arr / (len(x) * (len(x) - 1) / 2)
    norm_RR = RR_arr / (len(x_rand) * (len(x_rand) - 1) / 2)
    norm_DR = DR_arr / (len(x) * len(x_rand))

    # Landy-Szalay estimator for correlation function
    xi = (norm_DD - 2 * norm_DR + norm_RR) / norm_RR

    fit_mask = (r_centers_h > 60) & (r_centers_h < 160)

    try:
        xi_fit = xi[fit_mask]
        r_fit = r_centers_h[fit_mask]

        # Robust default parameters (within bounds)
        A_init = 0.02
        r0_init = 100.0  # Mpc/h — within [85, 115]
        sigma_init = 10.0
        C_init = 0.001

        popt, pcov = curve_fit(
            gaussian_model,
            r_fit,
            xi_fit,
            p0=[A_init, r0_init, sigma_init, C_init],
            bounds=([0, 85, 1, -0.5], [1, 115, 50, 0.5])
        )

        r_bao = popt[1]

        z_mid = 0.5 * (z_min + z_max)
        H_fid = cosmo.H(z_mid).value
        H_est = H_fid * r_d_fid_h / r_bao

        bao_peaks.append(r_bao)
        z_mid_vals.append(z_mid)
        Hz_vals.append(H_est)

        print(f"Bin {z_min}-{z_max}: BAO peak at {r_bao:.2f} Mpc/h, estimated H(z) = {H_est:.2f} km/s/Mpc")

        # Plot fit
        plt.figure()
        plt.plot(r_centers_h, xi, label='ξ(r)')
        plt.plot(r_fit, gaussian_model(r_fit, *popt), 'r--', label='Gaussian Fit')
        plt.axvline(r_bao, color='k', linestyle=':', label=f'Peak = {r_bao:.2f}')
        plt.xlabel('r [Mpc/h]')
        plt.ylabel('ξ(r)')
        plt.title(f'BAO Fit: {z_min} < z < {z_max}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Fit failed in bin {z_min}-{z_max}: {e}")

        continue

    # Plot correlation function and fit for this bin
    plt.plot(r_centers_h, xi, label=fr'{z_min} < z < {z_max}')
    plt.plot(r_centers_h[fit_mask], gaussian_model(r_centers_h[fit_mask], *popt), 'k--', alpha=0.5)

plt.xlabel(r'Separation $r$ [Mpc/h]')
plt.ylabel(r'$\xi(r)$')
plt.title('2PCF and BAO Peak Fits in Redshift Bins')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot H(z) comparison with Planck18 cosmology
z_planck = np.linspace(0.1, 2.0, 200)
H_planck = cosmo.H(z_planck).value  # km/s/Mpc

if Hz_vals:
    plt.figure(figsize=(8, 5))
    plt.plot(z_planck, H_planck, label='Planck18', color='black')
    plt.plot(z_mid_vals, Hz_vals, 'ro', label='Estimated H(z) from BAO')
    plt.xlabel('Redshift $z$')
    plt.ylabel(r'H(z) [km/s/Mpc]')
    plt.title('Estimated H(z) from BAO Peaks vs Planck18')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid BAO fits; skipping H(z) plot.")
