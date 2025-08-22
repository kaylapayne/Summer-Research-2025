from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

data = fits.open("zall-pix-guadalupe.fits")[1].data
mask = (data["ZWARN"] == 0) & (data["Z"] > 0)
ra = data["TARGET_RA"][mask]
dec = data["TARGET_DEC"][mask]
z = data["Z"][mask]

# Redshift binning parameters
z_start = 0.1
z_end = 1.0
bin_width = 0.005
step = 0.001
z_bins = np.arange(z_start, z_end - bin_width, step)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter([], [], s=0.05)
ztext = ax.text(0.98, 0.95, '', transform=ax.transAxes,
                fontsize=12, ha='right', va='top')
def init():
    ax.set_xlim(ra.min(), ra.max())
    ax.set_ylim(dec.min(), dec.max())
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ztext.set_text("")
    ax.set_title("DESI Galaxy Sky Positions by Redshift Slice")
    return sc,

def update(i):
    zmin = z_bins[i]
    zmax = zmin + bin_width
    zmid = (zmin + zmax) / 2
    half_width = (zmax - zmin) / 2

    # Gaussian-like fade function based on distance from zmid
    sigma = half_width * 0.8  # controls softness of fading
    weights = np.exp(-0.5 * ((z - zmid) / sigma)**2)

    # Mask and apply weights to points near this bin
    mask_bin = (z > zmin - half_width) & (z < zmax + half_width)
    ra_bin = ra[mask_bin]
    dec_bin = dec[mask_bin]
    alpha_bin = weights[mask_bin]

    # Set colored scatter with fading alpha
    sc.set_offsets(np.column_stack((ra_bin, dec_bin)))
    sc.set_alpha(None)  # needed to use per-point alpha
    sc.set_facecolors(np.column_stack((
        np.full_like(alpha_bin, 0.2),  # R
        np.full_like(alpha_bin, 0.4),  # G
        np.full_like(alpha_bin, 1.0),  # B
        alpha_bin                     # A (transparency)
    )))

    ax.set_title(f"Galaxies at ~ z = {zmid:.3f}")
    ztext.set_text(f"Redshift bin:\n{zmin:.3f}–{zmax:.3f}")
    return sc, ztext

ani = animation.FuncAnimation(
    fig, update, frames=len(z_bins),
    init_func=init, blit=True, interval=50
)

plt.show()
