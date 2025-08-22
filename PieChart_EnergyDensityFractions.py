import matplotlib.pyplot as plt
import numpy as np

# Energy density fractions (approximate Planck 2018)
labels = ['Dark Energy', 'Dark Matter', 'Baryonic Matter', 'Radiation']

# --- Cosmological parameters ---
omega_m0 = 0.315  # Total matter
omega_baryon = 0.05  # Baryonic (visible) matter
omega_dm_total = omega_m0 - omega_baryon  # Dark matter = total - baryonic
omega_r0 = 9.02e-5  # Radiation
omega_lambda0 = 1 - omega_r0 - omega_m0  # Dark energy = 1 - matter - radiation

sizes = [omega_lambda0, omega_dm_total, omega_baryon, omega_r0]

# Custom colors
colors = ['lightcoral', 'lightsalmon', 'sandybrown', '#ffcccc']

# Explode only radiation
explode = (0.0, 0.0, 0.0, 0.15)

# Create the figure
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts = ax.pie(
    sizes,
    explode=explode,
    labels=None,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'black'}
)

# Add percentage labels outside the pie
total = sum(sizes)
for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))
    percentage = f"{100 * sizes[i] / total:.2f}%"

    # Move the Radiation label farther to the left
    if labels[i] == 'Radiation':
        ax.text(-0.17, 1.15 * y, percentage, ha='center', va='center', fontsize=10)
    else:
        ax.text(1.2 * x, 1.1 * y, percentage, ha='center', va='center', fontsize=10)

# Add legend
ax.legend(wedges, labels,
          title="Components",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# Title
plt.title("Energy Density Fractions Today", fontsize=16)

# Layout
plt.tight_layout()

# Save as transparent PDF
fig.savefig("energy_density_fractions.pdf", transparent=True)

# Optional: show figure
plt.show()
