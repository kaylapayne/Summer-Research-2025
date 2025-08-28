import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.integrate import solve_ivp

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
omega_lambda_scales = [[1.015,1.014,1.0091],[1.0132,1.0125,1.008],[1,1,1]] #[[1.014,1.0099,1.0037],[1.0125,1.0089,1.0032],[1,1,1]] #[[1.0099,1.0083,1.0072],[1.0089,1.0075,1.0064],[1,1,1]] #[[0.469, 0.870, 0.934],[0.605, 0.890, 0.942],[0.672, 0.902, 0.947]]    # For scaling DE fraction for each decay lifetime trial
omega_ddm0 = 0.04 * omega_dm_total          # Energy density percentage of decaying dark matter now (at 4% of DM)
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

cols = [0, 1]
lifetimes = [0.05, 0.5, 5]      #[0.5, 4, 20]                             # DDM lifetimes in Gyr – as suggested for comparison by Vincent
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
        delta_H_frac = (H_decay - H_lcdm) / H_lcdm  # Computes fractional difference between H(z) in the DDM model and the standard ΛCDM model
        this_label = f"{label}, DE scale: {scale}"
        axes[0, col].plot(1 + z_sol, H_decay, color=color, label=this_label)  # Top row: H(z), swapping from loglog to plot for better readability at a small range
        axes[1, col].plot(1 + z_sol, delta_H_frac, color=color)         # Bottom row: ΔH/H_LCDM, swapping from semilogx to plot for better readability at a small range

        print(f"z_start = {z_decay_start}, Final rho_r = {omega_r[-1]:.3e}, Final rho_x = {omega_ddm[-1]:.3e}")

    # Add LambdaCDM reference
    axes[0, col].plot(1 + z_eval, H_lcdm, 'k--', label=r'$\Lambda$CDM')
    axes[1, col].axhline(0, color='gray', lw=0.8, ls='--')

    # Adding Hubble constant values with error bars for comparison
    # SH0ES H0 (distance ladder)
#    axes[0, col].errorbar(1, 73.0, yerr=1.0, fmt='o', color='cornflowerblue', label='SH0ES (Local)', markersize=4)

    # Planck H0 (CMB-inferred)
#    axes[0, col].errorbar(1, 67.4, yerr=0.5, fmt='o', color='darkseagreen', label='Planck (CMB)', markersize=4)

    # Titles and legend
    axes[0, col].set_title(f'DM Decay Starts at $z = {z_decay_start}$', fontsize=13)
    axes[0, col].legend(fontsize=10, loc='upper left')

    # Axis labels
    if col == 0:
        axes[0, col].set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=12)
        axes[1, col].set_ylabel(r'Fractional Difference $\frac{H_{\mathrm{decay}} - H_{\Lambda \mathrm{CDM}}}{H_{\Lambda \mathrm{CDM}}}$', fontsize=12)

    axes[1, col].set_xlabel(r'$1 + z$', fontsize=12)

    # Grid
    axes[0, col].grid(True, which='both', ls='--', lw=0.5)
    axes[1, col].grid(True, which='both', ls='--', lw=0.5)

for ax in axes[0]:
    ax.set_xlim(right=7.5)
    ax.set_xlim(left=0.95)
    ax.set_ylim(top=800)
    ax.set_ylim(bottom=65)
for ax in axes[1]:
    ax.set_xlim(right=7.5)
    ax.set_xlim(left=0.95)
    ax.set_ylim(top=0.002)
    #ax.set_ylim(bottom=-0.012)

#axes[0, 0].set_xlim(0.95, 7)
#axes[0, 0].set_ylim(65, 800)
#axes[1, 0].set_xlim(0.95, 7)
#axes[1, 0].set_ylim(top=0.002)
#axes[0, 1].set_xlim(0.95, 7)
#axes[0, 1].set_ylim(65, 800)
#axes[1, 1].set_xlim(0.95, 7)
#axes[1, 1].set_ylim(top=0.002)

plt.tight_layout()

#plt.savefig("ddm_plot.pdf", transparent=True)

plt.show()
