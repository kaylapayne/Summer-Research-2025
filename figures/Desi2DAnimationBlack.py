from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.font_manager import FontProperties
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from matplotlib.patches import Circle

# Load font from file
#custom_font = FontProperties(fname='mission-control.otf')

#data = fits.open("data/zall-pix-guadalupe.fits")[1].data
print("Opening the FITS data...")
data = fits.open("../data/zall-pix-guadalupe.fits")[1].data
mask = (data["ZWARN"] == 0) & (data["Z"] > 0)
count = len(data)
ra = data["TARGET_RA"][mask]
dec = data["TARGET_DEC"][mask]
z = data["Z"][mask]

print("Binning based on redshift...")
# Redshift binning parameters
z_start = 0.02
z_end = 1.00
bin_width = 0.005
step = 0.0001
z_bins = np.arange(z_start, z_end - bin_width, step)

print("Setting up plots...")
# Set up the plot
dpi = 200
fig, ax = plt.subplots(figsize=(1600 / dpi, 1000 / dpi), dpi=dpi, facecolor='white')
#fig, ax = plt.subplots(figsize=(3840 / dpi, 2160 / dpi), dpi=dpi, facecolor='black')
ax.set_facecolor('white')
sc = ax.scatter([], [], s=0.3,  edgecolors='none', facecolor='black')

# Scale bar axes just above status bar
ax_scale = fig.add_axes([0.1, 0.04, 0.8, 0.02])  # [left, bottom, width, height]
ax_scale.set_xlim(0, 1)
ax_scale.set_ylim(0, 1)
ax_scale.axis('off')  # Hide axes

# BAO scale length in Mpc (sound horizon)
r_d = 147.0

def bao_angular_radius_deg(zmid, r_d=r_d):
    """Calculate angular BAO scale in degrees at redshift zmid."""
    D_c = cosmo.comoving_distance(zmid).value  # Mpc
    theta_rad = r_d / D_c
    return np.degrees(theta_rad)


def init():
    global ztext_left, ztext_center, ztext_right, bao_line, bao_caps
    ax.set_xlim(ra.min(), ra.max())
    ax.set_ylim(dec.min(), dec.max())
    ax.set_xlabel("RA [deg]", fontsize=7)
    ax.set_ylabel("DEC [deg]", fontsize=7)
    ax.set_title("DESI Galaxy Sky Positions: Redshift Slice")
    ax_status = fig.add_axes([0, 0.0, 1, 0.03], facecolor='#505050', zorder=10)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.patch.set_alpha(0.1)


    # Left-aligned text (e.g. redshift)
    ztext_left = ax_status.text(0.01, 0.5, '', ha='left', va='center',
                                fontsize=8, color='#505050')
#                                fontproperties=custom_font)

    # Left-aligned text (e.g. redshift)
    ztext_center = ax_status.text(0.5, 0.5, '', ha='center', va='center',
                                fontsize=8, color='#505050')
#                                fontproperties=custom_font)

    # Right-aligned text (e.g. comoving distance)
    ztext_right = ax_status.text(0.99, 0.5, '', ha='right', va='center',
                                 fontsize=8, color='#505050')
#                                 fontproperties=custom_font)

    # # Initialize BAO scale bar and caps (line + vertical caps)
    # bao_line, = ax_scale.plot([], [], color='#303030', lw=2)
    # bao_caps, = ax_scale.plot([], [], color='#303030', lw=2)

    return sc, ztext_left, ztext_center, ztext_right



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
    current_count = len(ra_bin)

    sc.set_offsets(np.column_stack((ra_bin, dec_bin)))
    sc.set_alpha(None)  # needed to use per-point alpha
    sc.set_facecolors("black")

    distance_mpc = cosmo.comoving_distance(zmid).to(u.Mpc).value
    bao_deg = bao_angular_radius_deg(zmid)

    print(f"Galaxies at ~ z = {zmid:.4f}")
    ztext_left.set_text("DESI Dataset (zall-pix-guadalupe.fits)")
    ztext_center.set_text(f"Displaying {current_count:.0f} / {count:.0f} Galaxies")
    ztext_right.set_text(f"Redshift: {zmid:.4f} ({distance_mpc:.0f} MPC)")

    # # Update BAO scale bar
    # bao_deg = bao_angular_radius_deg(zmid)
    # ra_range = ra.max() - ra.min()
    # bar_len_frac = bao_deg / ra_range  # fraction of x-axis
    #
    # bar_start = 0.5 - bar_len_frac / 2
    # bar_end = 0.5 + bar_len_frac / 2
    # y_bar = 0.5  # centered vertically in scale bar ax
    # cap_height = 0.6  # relative height of vertical caps
    #
    # # Horizontal bar line
    # bao_line.set_data([bar_start, bar_end], [y_bar, y_bar])
    #
    # # Vertical caps at ends (two vertical lines)
    # bao_caps_x = [bar_start, bar_start, np.nan, bar_end, bar_end]
    # bao_caps_y = [y_bar - cap_height / 2, y_bar + cap_height / 2,
    #               np.nan,
    #               y_bar - cap_height / 2, y_bar + cap_height / 2]
    # bao_caps.set_data(bao_caps_x, bao_caps_y)

    return sc, ztext_left, ztext_center, ztext_right


ani = animation.FuncAnimation(
    fig, update, frames=len(z_bins),
    init_func=init, blit=True, interval=1
)

#ani.save('desi_animation.mp4', fps=30, dpi=dpi, bitrate=10000 )
#print("done")

plt.show()
