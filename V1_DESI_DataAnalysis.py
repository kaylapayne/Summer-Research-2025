import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from scipy.integrate import quad
import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter


# --- Dataset folders by survey type ---
dataset_sources = {
    "BGS": {
        "base_path": Path("results") / "BGS_BRIGHT_NGC_clustering.dat.fits" / "ISOTROPIC",
        "folders": [
            "2025-06-15_13-55-54__0.25-0.4__340380",
            "2025-06-15_13-30-09__0.1-0.25__299540"
        ]
    },
    "LRG": {
        "base_path": Path("results") / "LRG_N_clustering.dat.fits" / "ISOTROPIC",
        "folders": [
            "2025-06-16_17-24-23__0.4-0.8__48313",
            "2025-06-16_17-25-38__0.4-1.1__80651",
            "2025-06-16_17-34-29__0.8-1.1__32338"
        ]
    },
    "ELG": {
        "base_path": Path("results") / "ELG_LOPnotqso_NGC_clustering.dat.fits" / "ISOTROPIC",
        "folders": [
            "2025-06-26_15-54-09__0.8-1.0__500994",
            "2025-06-26_16-00-26__1.0-1.2__514903",
            "2025-06-26_16-03-32__1.2-1.4__466386",
            "2025-06-26_16-05-11__1.4-1.6__339039"
        ]
    }
}

# --- Weird sound horizon calcs ---

# Sound horizon from Planck18
#r_d = cosmo.comoving_sound_horizon(cosmo.zdrag).value  # ~147.09 Mpc
#r_d = 147.09  # Mpc (Planck 2018 value for r_d at z_drag)
#r_d_fid = 105  # Assuming you built your template with s_fid = 105 Mpc/h

# Speed of light
c = 299792.458  # km/s

# Compute z_drag using Eisenstein & Hu 1998 approximation (consistent with Planck)
Ω_m = cosmo.Om0
Ω_b = cosmo.Ob0
h = cosmo.h

# Eisenstein & Hu drag redshift approximation
def z_drag_EH98(Ω_m, Ω_b, h):
    b1 = 0.313 * (Ω_m * h**2)**(-0.419) * (1 + 0.607 * (Ω_m * h**2)**0.674)
    b2 = 0.238 * (Ω_m * h**2)**0.223
    return 1291 * (Ω_m * h**2)**0.251 / (1 + 0.659 * (Ω_m * h**2)**0.828) * (1 + b1 * (Ω_b * h**2)**b2)

z_drag = z_drag_EH98(Ω_m, Ω_b, h)

# Sound speed at early times
def R(z):
    return 3 * Ω_b / (4 * cosmo.Onu0) * (1 / (1 + z))

def integrand(z):
    return c / cosmo.H(z).value / np.sqrt(3 * (1 + R(z)))

# Numerically integrate from z_drag to ∞ to get comoving sound horizon
r_d, err = quad(integrand, z_drag, 1e5)
r_d_mpc_h = r_d * h  # Convert Mpc → Mpc/h

# Now r_d is in Mpc
print(f"r_d ≈ {r_d_mpc_h:.2f} Mpc at z_drag ≈ {z_drag:.2f}")

r_d_fid = 105  # The sound horizon used in your BAO template in Mpc/h

# --- End of weird sound horizon calcs ---

# Plot containers
hz_points = []
hz_labels = []

# Prepare for subplot layout
total_datasets = sum(len(info["folders"]) for info in dataset_sources.values())
cols = 3
rows = (total_datasets + cols - 1) // cols  # ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
axes = axes.flatten()  # easier indexing

# Iterate through both BGS and LRG
plot_index = 0  # track subplot position
for source_label, source_info in dataset_sources.items():
    base_path = source_info["base_path"]
    for folder in source_info["folders"]:
        folder_path = base_path / folder
        csv_file = folder_path / "xi_matrix.csv"
        settings_file = folder_path / "settings.json"

        # Load metadata
        with open(settings_file, 'r') as f:
            info = json.load(f)
        z_min, z_max = info["z_min"], info["z_max"]
        z_eff = 0.5 * (z_min + z_max)
        subsample = info.get("subsample_frac_data", 1.0)

        # Load xi(s)
        df = pd.read_csv(csv_file)
        df.columns = ["s", "xi"]
        s = df["s"].values
        xi = df["xi"].values
        xi_plot = s**2 * xi

        # Gaussian fit
        def bao_peak(s, A, s0, sigma, C):
            return A * np.exp(-(s - s0)**2 / (2 * sigma**2)) + C

        mask = (s > 80) & (s < 130)
        popt, pcov = curve_fit(bao_peak, s[mask], xi_plot[mask], p0=[10, 105, 10, 0])
        A_fit, s_peak, sigma_fit, C_fit = popt
        s_peak_err = np.sqrt(np.diag(pcov))[1]

        # Estimate H(z)
        s_fid = 105
        H_fid = cosmo.H(z_eff).value
        #Hz = H_fid * (s_fid / s_peak)
        #Hz_err = H_fid * (s_fid / s_peak**2) * s_peak_err
        Hz = H_fid * (r_d_fid / r_d_mpc_h) * (s_fid / s_peak)
        Hz_err = H_fid * (r_d_fid / r_d_mpc_h) * (s_fid / s_peak ** 2) * s_peak_err

        # Store for H(z) plot
        hz_points.append((z_eff, Hz, Hz_err))
        label = f"{source_label} $z={z_min:.2f}-{z_max:.2f}$, {subsample*100:.0f}%"
        hz_labels.append(label)

        # Plot s²ξ(s) and Gaussian fit on individual subplot
        ax = axes[plot_index]
        ax.plot(s, xi_plot, label="Data", color='mediumslateblue')
        ax.plot(s[mask], bao_peak(s[mask], *popt), linestyle='--', label="Gaussian Fit", color='orangered')
        ax.set_title(label)
        ax.grid(True)
        plot_index += 1

# Hide any unused subplots
for ax in axes[plot_index:]:
    ax.axis("off")

# Finalize the figure layout
fig.suptitle("BAO Fits Across Redshift Bins (BGS + LRG)", fontsize=16)
fig.supxlabel(r"$s$ [Mpc/h]")
fig.supylabel(r"$s^2 \xi(s)$")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot H(z) vs 1+z
z_model = np.linspace(0.01, 2.0, 200)
Hz_LCDM = cosmo.H(z_model).value

# Sort all entries by z_eff for ordered legend
sorted_data = sorted(zip(hz_points, hz_labels), key=lambda x: x[0][0])
hz_points_sorted, hz_labels_sorted = zip(*sorted_data)

plt.figure(figsize=(8, 5))
plt.plot(1 + z_model, Hz_LCDM, 'gray', linestyle='--', label='ΛCDM (Planck18)')

# With error
#for (z_eff, Hz, Hz_err), label in zip(hz_points_sorted, hz_labels_sorted):
#    plt.errorbar(1 + z_eff, Hz, yerr=Hz_err, fmt='o', label=label)

# Normalize redshift for colormap
z_vals = np.array([z for (z, _, _) in hz_points_sorted])
norm = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
cmap = cm.tab20b #cm.plasma cm.cool

# Without error bars
for (z_eff, Hz, Hz_err), label in zip(hz_points_sorted, hz_labels_sorted):
    color = cmap(norm(z_eff))
    plt.errorbar(1 + z_eff, Hz, fmt='o', color=color, label=label)

plt.xlabel(r'$1 + z$')
plt.ylabel(r'$H(z)$ [km/s/Mpc]')
plt.title("Extracted H(z) from BAO vs ΛCDM (BGS + LRG)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot but adding fit line ---

# Printing values in order
for (z_eff, Hz, Hz_err), label in zip(hz_points_sorted, hz_labels_sorted):
    print(f"{label}: H(z={z_eff:.2f}) = {Hz:.2f} ± {Hz_err:.2f} km/s/Mpc")

# Define H(z) model: flat ΛCDM, omitting radiation because you can at low redshift
#def Hz_LCDM_model(z, H0, Om):
    #return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

# Defining H(z) including radiation to be consistant and so that if i ever scale to higher redshifts it doesn't disagree
def Hz_LCDM_model(z, H0, Om):
    Or = (cosmo.Ogamma0 + cosmo.Onu0)
    return H0 * np.sqrt(Or * (1 + z)**4 + Om * (1 + z)**3 + (1 - Om - Or))

# Prepare data
z_data = np.array([z for (z, _, _) in hz_points_sorted])
Hz_data = np.array([Hz for (_, Hz, _) in hz_points_sorted])
Hz_err_data = np.array([Hz_err for (_, _, Hz_err) in hz_points_sorted])

# Fit to data with uncertainties
popt, pcov = curve_fit(Hz_LCDM_model, z_data, Hz_data, sigma=Hz_err_data, absolute_sigma=True)
H0_fit, Om_fit = popt
H0_err, Om_err = np.sqrt(np.diag(pcov))

# Print best-fit values
print(f"\nFitted flat ΛCDM parameters:")
print(f"H0 = {H0_fit:.2f} ± {H0_err:.2f} km/s/Mpc")
print(f"Ωm = {Om_fit:.3f} ± {Om_err:.3f}")

# Plot result
z_plot = np.linspace(z_data.min()*0.9, z_data.max()*1.1, 300)
Hz_fit = Hz_LCDM_model(z_plot, *popt)

plt.figure(figsize=(8, 5))
plt.plot(1 + z_plot, Hz_fit, color='royalblue', label=f'Fit: $H_0={H0_fit:.1f}$, $\\Omega_m={Om_fit:.3f}$')
plt.plot(1 + z_model, Hz_LCDM, 'gray', linestyle='--', label='Planck18 ΛCDM')

# Data points
for (z_eff, Hz, Hz_err), label in zip(hz_points_sorted, hz_labels_sorted):
    color = cmap(norm(z_eff))
    plt.errorbar(1 + z_eff, Hz, yerr=Hz_err, fmt='o', color=color, label=label)

plt.xlabel(r'$1 + z$')
plt.ylabel(r'$H(z)$ [km/s/Mpc]')
plt.title("BAO-Inferred H(z) with ΛCDM Fit")
plt.grid(True)
plt.legend(fontsize='small', loc='upper left')
plt.tight_layout()
plt.show()

# --- Plot H(z)*r_d so that it is model independent ---

# Compute H(z) * r_d for each point
Hz_rd_data = Hz_data * r_d_mpc_h
Hz_rd_err_data = Hz_err_data * r_d_mpc_h

# Also compute model prediction H(z)*r_d
Hz_rd_model = Hz_LCDM * r_d_mpc_h
Hz_rd_fit = Hz_LCDM_model(z_plot, *popt) * r_d_mpc_h

plt.figure(figsize=(8, 5))
plt.plot(1 + z_plot, Hz_rd_fit, color='royalblue', label='Fitted $H(z) \\cdot r_d$')
plt.plot(1 + z_model, Hz_rd_model, 'gray', linestyle='--', label='Planck18 $H(z) \\cdot r_d$')

# Plot H(z)*r_d data points
for (z_eff, _, _), Hz_rd, Hz_rd_err, label in zip(hz_points_sorted, Hz_rd_data, Hz_rd_err_data, hz_labels_sorted):
    color = cmap(norm(z_eff))
    plt.errorbar(1 + z_eff, Hz_rd, yerr=Hz_rd_err, fmt='o', color=color, label=label)

plt.xlabel(r'$1 + z$')
plt.ylabel(r'$H(z) \cdot r_d$ [km/s]')
plt.title("BAO-Inferred $H(z) \cdot r_d$ (Model-Independent)")
plt.grid(True)
plt.legend(fontsize='small', loc='upper left')

# Set y-axis to scientific notation
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 3))  # Force scientific for large numbers
plt.gca().yaxis.set_major_formatter(formatter)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))  # Alternative way if needed

plt.tight_layout()
plt.show()
