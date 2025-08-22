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
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


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

# --- H(z)*r_d PLOTTING STUFF ---

# for plot H(z) vs 1+z
z_model = np.linspace(0.01, 2.0, 200)
Hz_LCDM = cosmo.H(z_model).value

# Sort all entries by z_eff for ordered legend
sorted_data = sorted(zip(hz_points, hz_labels), key=lambda x: x[0][0])
hz_points_sorted, hz_labels_sorted = zip(*sorted_data)

# Normalize redshift for colormap
z_vals = np.array([z for (z, _, _) in hz_points_sorted])
norm = mcolors.Normalize(vmin=z_vals.min(), vmax=z_vals.max())
cmap = cm.tab20b #cm.plasma cm.cool

# --- Plot but adding fit line ---

# Printing values in order
for (z_eff, Hz, Hz_err), label in zip(hz_points_sorted, hz_labels_sorted):
    print(f"{label}: H(z={z_eff:.2f}) = {Hz:.2f} ± {Hz_err:.2f} km/s/Mpc")

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

# Plot H(z)*r_d so that it is model independent
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



# === LETS COMPARE SOME DDM cause why not ;P ===

# --- Constants ---
H0 = 70                             # Hubble constant today in km/s/Mpc
H0_s = H0 * 1000 / (3.086e22)       # Convert to 1/s
Gyr = 3.154e16                      # Seconds in a gigayear

# --- Cosmological parameters ---
omega_m0 = 0.315                            # Energy density percentage of matter now
omega_baryon = 0.05                         # Energy density percentage of matter we see (baryons) now
omega_dm_total = omega_m0 - omega_baryon    # Energy density percentage of dark matter now
omega_r0 = 9.02e-5                          # Energy density percentage of radiation now
omega_lambda0 = 1 - omega_r0 - omega_m0     # Energy density percentage of dark energy now
omega_lambda_scales = [[1.12,1.11,1.075],[1.11,1.10,1.068],[1,1,1]] #[[1.015,1.014,1.0091],[1.0132,1.0125,1.008],[1,1,1]] #[[1.014,1.0099,1.0037],[1.0125,1.0089,1.0032],[1,1,1]] #[[1.0099,1.0083,1.0072],[1.0089,1.0075,1.0064],[1,1,1]] #[[0.469, 0.870, 0.934],[0.605, 0.890, 0.942],[0.672, 0.902, 0.947]]    # For scaling DE fraction for each decay lifetime trial
percent_ddm = 0.33                          # Vincent initially suggested 4%
omega_ddm0 = percent_ddm * omega_dm_total   # Energy density percentage of decaying dark matter now
omega_stable_m0 = omega_m0 - omega_ddm0     # Energy density percentage of stable matter (DM and baryons) now

# --- Settings ---
z_i = 1e5                                       # Initial redshift we start integrating - aka earliest point this model covers (1e4 being in well back into the radiation dominated era)
z_eval = np.logspace(np.log10(z_i), -3, 600)    # Logspace of 600 values from 0.001 to z_i basically np.logspace(1, -3, 600), from 10 down to 0.001
z_eval = np.append(z_eval, 0)                   # add zero
z_eval = np.sort(z_eval)[::-1]                  # descending order for solver
a_eval = 1 / (1 + z_eval)                       # 'a' is the scale factor – a=1/(1+z)
H_lcdm = H0 * np.sqrt(                          # Calculating the Hubble parameter for each value in a_eval
    omega_r0 * a_eval**-4 +                     # Contribution from radiation scaling as a**-4
    omega_m0 * a_eval**-3 +                     # Contribution from matter scaling as a**-3
                                                # Contribution from curvature scaling as a**-2 is ignored assuming a flat universe
    omega_lambda0                               # Contribution from dark energy – constant contribution
)
H_lcdm_rd = H_lcdm*r_d_mpc_h                    # From Friendmann equation
Hz_LCDM_rd = cosmo.H(z_eval).value*r_d_mpc_h    # From Plank18

cols = [0, 1]
lifetimes = [0.05, 0.5, 5]                      # DDM lifetimes in Gyr – as suggested for comparison by Vincent
z_decay_starts = [100, 6]                              # Redshift at which the decay starts – I have added 3 here to compare
colors = ['darkgoldenrod', 'firebrick', 'darkorchid']       # Colours for plotting the 3 different lifetimes later
labels = [f'Lifetime = {tau} Gyr' for tau in lifetimes]     # Generates labels for legend

# --- Plot setup ---
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)    # Setting up 6 subplots to compare, 3 H(z) plots, 3 fractional difference between given DDM model and LCDM
fig.subplots_adjust(hspace=0.3, wspace=0.2)                     # Adjusting spacing of subplots

for col, z_decay_start, omega_lambda_scales_vector in zip(cols, z_decay_starts, omega_lambda_scales):    # Looping over the different decay start times defined in z
    for tau, scale, color, label in zip(lifetimes, omega_lambda_scales_vector, colors, labels):         # Looping over the four lists in parallel, this is a bit fancy, probably not entirely necessary
        Gamma = 1 / (tau * Gyr)                                                                         # Defining Gamma given the lifetime converted into seconds
        omega_lambda = omega_lambda0 * scale                                    # scaled Lambda for this model

        def d_density_params_dz(z, y):
            omega_ddm, omega_r = y                                              # Unpack density parameters for DDM and radiation
            omega_stable_m = omega_stable_m0 * (1 + z) ** 3                     # Stable matter density parameter at redshift z (scales as (1+z)**3)
            #omega_lambda = omega_lambda0                                        # Dark energy density parameter (assumed constant)
            omega_total = omega_stable_m + omega_ddm + omega_r + omega_lambda   # Total dimensionless density parameter at z
            H = H0_s * np.sqrt(omega_total)                                     # Hubble parameter H(z) in s⁻¹ units (scaled by H0_s)
            if H <= 0:                                                          # Avoid unphysical sqrt of negative or zero
                print(f"[WARNING] omega_total = {omega_total:.3e} at z = {z:.2f} → H = {H:.3e}")
                return [0, 0]
            Gamma_eff = Gamma if z <= z_decay_start else 0                      # Effective decay rate switches on only when z <= decay start redshift

            # Evolution equations for DDM and radiation density parameters - I calculated these in my notes - don't be confused by the sign shift, that comes from converting from equations wrt t to wrt z
            d_omega_ddm_dz = (3 * omega_ddm + (Gamma_eff / H) * omega_ddm) / (1 + z)
            d_omega_r_dz = (4 * omega_r - (Gamma_eff / H) * omega_ddm) / (1 + z)

            return [d_omega_ddm_dz, d_omega_r_dz]                   # Returns the derivatives of decaying dark matter and radiation density parameters with respect to redshift z

        y0 = [omega_ddm0 * (1 + z_i)**3, omega_r0 * (1 + z_i)**4]   # Setting initial conditions for DDM and rad, scaling by z_i – you get the initial amounts of these species at the beginning of our simulation

        sol = solve_ivp(            # Solve initial value problem for a system of ODEs
            d_density_params_dz,    # This is the function we're solving — it returns the derivatives
            (z_i, 0),               # This is the interval of integration in redshift
            y0,                     # Initial condition vector at z = z_i
            t_eval=z_eval,          # Tells the solver at which redshift points we want the solution returned
            method='BDF',           # This chooses the numerical integrator - BDF = Backward Differentiation Formula, which is good for stiff ODE systems, which this is (because of the sharp decay terms from Γ)
            rtol=1e-8,              # Error tolerance relative to solution size
            atol=1e-10              # Error tolerance absolute threshold
        )

        z_sol = sol.t                               # Extracts the redshift values at which the solution was evaluated, for plotting and evaluation
        omega_ddm = np.maximum(sol.y[0], 1e-30)     # Acts as numerical floor - ensures DDM density stays positive and avoids numerical issues (e.g. sqrt of negative)
        omega_r = np.maximum(sol.y[1], 1e-30)       # Ensure radiation density stays positive and numerically stable
        a_sol = 1 / (1 + z_sol)                     # Converts redshift array from solver to scale factor values
        omega_m = omega_stable_m0 * a_sol ** -3     # Evolve stable matter density with redshift (scales as a**-3)
        H_decay = H0 * np.sqrt(omega_m +            # Computing the Hubble parameter in a universe with DDM
                               omega_ddm +
                               omega_r +
                               omega_lambda)
        H_decay_rd = H_decay*r_d_mpc_h
        delta_H_rd_frac = (H_decay_rd - H_lcdm_rd) / H_lcdm_rd  # Computes fractional difference between H(z) in the DDM model and the standard ΛCDM model
        this_label = f"{label}, DE scale: {scale}"
        axes[0, col].plot(1 + z_sol, H_decay_rd, color=color, label=this_label)  # Top row: H(z), swapping from loglog to plot for better readability at a small range
        axes[1, col].plot(1 + z_sol, delta_H_rd_frac, color=color)         # Bottom row: ΔH/H_LCDM, swapping from semilogx to plot for better readability at a small range

        print(f"z_start = {z_decay_start}, Final rho_r = {omega_r[-1]:.3e}, Final rho_x = {omega_ddm[-1]:.3e}")

    # Add LambdaCDM reference
    axes[0, col].plot(1 + z_eval, H_lcdm_rd, 'gray', linestyle='--', label=r'$\Lambda$CDM $\cdot r_d$ (from Friedmann equation)')
    axes[0, col].plot(1 + z_eval, Hz_LCDM_rd, 'red', linestyle='--', label=r'$\Lambda$CDM $\cdot r_d$ (from Plank18 data)')
    axes[0, col].plot(1 + z_plot, Hz_rd_fit, color='royalblue', linestyle='-.', label=r'Fitted $H(z) \cdot r_d$')

    # Plot H(z) * r_d data points with error bars
#    for (z_eff, _, _), Hz_rd, Hz_rd_err, label in zip(hz_points_sorted, Hz_rd_data, Hz_rd_err_data, hz_labels_sorted):
#        color = cmap(norm(z_eff))
#        axes[0, col].errorbar(
#            1 + z_eff, Hz_rd, yerr=Hz_rd_err,
#            fmt='o', color=color, label=label, markersize=4, capsize=2
#        )

    # Titles and legend
    axes[0, col].set_title(f'DM Decay Starts at $z = {z_decay_start}$', fontsize=13)
    # axes[0, col].legend(fontsize=10, loc='upper left')
    axes[0, col].legend(fontsize=8, loc='upper left')

    # Plot data points with error bars on left
    if col == 0:
        data_handles = []
        data_labels = []
        for (z_eff, _, _), Hz_rd, Hz_rd_err, label in zip(hz_points_sorted, Hz_rd_data, Hz_rd_err_data,
                                                          hz_labels_sorted):
            color = cmap(norm(z_eff))
            h = axes[0, col].errorbar(
                1 + z_eff, Hz_rd, yerr=Hz_rd_err,
                fmt='o', color=color, markersize=4, capsize=2
            )
            data_handles.append(h)
            data_labels.append(label)

    # Plot data points with error bars on right
    if col == 1:
        data_handles = []
        data_labels = []
        for (z_eff, _, _), Hz_rd, Hz_rd_err, label in zip(hz_points_sorted, Hz_rd_data, Hz_rd_err_data,
                                                          hz_labels_sorted):
            color = cmap(norm(z_eff))
            h = axes[0, col].errorbar(
                1 + z_eff, Hz_rd, yerr=Hz_rd_err,
                fmt='o', color=color, label=label, markersize=4, capsize=2
            )
            data_handles.append(h)
            data_labels.append(label)

        # Add separate legend for data points
        from matplotlib.legend import Legend

        axes[0, col].add_artist(
            Legend(
                axes[0, col], data_handles, data_labels,
                loc='lower right', fontsize=8, title='Data Points', ncol=1
            )
        )

    axes[1, col].axhline(0, color='gray', lw=0.8, ls='--')
    #axes[0, col].plot(1 + z_model, Hz_rd_model, 'gray', linestyle='--', label='Planck18 $H(z) \\cdot r_d$')

    # Plot fractional difference of data points wrt LCDM
    for (z_eff, _, _), Hz_rd, Hz_rd_err in zip(hz_points_sorted, Hz_rd_data, Hz_rd_err_data):
        # Compute LCDM H(z) at z_eff
#        H_LCDM_at_z = cosmo.H(z_eff).value
#        H_LCDM_rd = H_LCDM_at_z * r_d_mpc_h
        # Interpolate your precomputed H_lcdm_rd at z_eval
        Hz_LCDM_rd_func = interp1d(z_eval, H_lcdm_rd, kind='cubic', bounds_error=False, fill_value="extrapolate")
        # Evaluate at the z_eff points
        H_LCDM_rd = Hz_LCDM_rd_func(z_eff)

        delta_frac = (Hz_rd - H_LCDM_rd) / H_LCDM_rd
        delta_frac_err = Hz_rd_err / H_LCDM_rd

        color = cmap(norm(z_eff))
        axes[1, col].errorbar(
            1 + z_eff, delta_frac, yerr=delta_frac_err,
            fmt='o', color=color, markersize=4, capsize=2
        )

        # --- Best-fit ΛCDM line for fractional difference plot ---
        # Evaluate best-fit model at z_eval
        Hz_fit_line = Hz_LCDM_model(z_eval, H0_fit, Om_fit) * r_d_mpc_h
        Hz_planck_line = cosmo.H(z_eval).value * r_d_mpc_h

        # Fractional difference: (fit - Planck) / Planck
        delta_Hz_fit_frac = (Hz_fit_line - Hz_planck_line) / Hz_planck_line

        axes[1, col].plot(
            1 + z_eval, delta_Hz_fit_frac,
            label='Best-fit ΛCDM (from BAO)', color='royalblue', linewidth=1.5, linestyle='-.'
        )

    # Axis labels
    if col == 0:
        axes[0, col].set_ylabel(r'$H(z) \cdot r_d$ [km/s/Mpc]', fontsize=12)
        axes[1, col].set_ylabel(
            r'Fractional Difference $\frac{H_{\mathrm{decay}} - H_{\Lambda \mathrm{CDM}}}{H_{\Lambda \mathrm{CDM}}} \cdot r_d$',
            fontsize=12
        )
    axes[1, col].set_xlabel(r'$1 + z$', fontsize=12)

    # Grid
    axes[0, col].grid(True, which='both', ls='--', lw=0.5)
    axes[1, col].grid(True, which='both', ls='--', lw=0.5)

for ax in axes[0]:
    ax.set_xlim(right=7.5)
    ax.set_xlim(left=0.95)
    ax.set_ylim(top=80000)
    ax.set_ylim(bottom=65)
for ax in axes[1]:
    ax.set_xlim(right=7.5)
    ax.set_xlim(left=0.95)
    ax.set_ylim(top=0.05)
    #ax.set_ylim(bottom=-0.012)

plt.suptitle(
    rf'Initial DDM = {percent_ddm*100:.1f}%,   $H_0 = {H0:.2f}\ \mathrm{{km/s/Mpc}}$ for Friedmann Models',
    fontsize=12
)


plt.tight_layout()
plt.show()

# === END OF DDM CALCS ===
