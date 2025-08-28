from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import pyvista as pv

print("Opening FITS file...")
# Load redshift catalog
zcat = fits.open("data/zall-pix-guadalupe.fits")[1].data

print("Extracting z, ra, dec data...")
mask = (zcat["ZWARN"] == 0) & (zcat["Z"] > 0.01)  # good redshifts
z = zcat["Z"][mask]
ra = zcat["TARGET_RA"][mask]
dec = zcat["TARGET_DEC"][mask]

# Convert to comoving distances
r = cosmo.comoving_distance(z).value
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=r*u.Mpc)
print("Converting to cartesian coordinates...")
x, y, z_cart = coords.cartesian.x.value, coords.cartesian.y.value, coords.cartesian.z.value

print("Preparing data for plotting...")
cloud = pv.PolyData(np.column_stack((x, y, z_cart)))
cloud["redshift"] = z

plotter = pv.Plotter()
plotter.set_background("black")
plotter.add_points(
    cloud,
    render_points_as_spheres=True,
    point_size=2,
    scalars="redshift",
    cmap="plasma_r",
#    cmap="cubehelix",
#    cmap="magma",
#    cmap="turbo",
    scalar_bar_args={
        "title": "Redshift",
        "title_font_size": 36,
        "label_font_size": 24,
        "color": "white",
        "fmt": "%.2f",
        "vertical": True,
        "position_x": 0.88,  # adjust as needed for layout
        "position_y": 0.15,
        "width": 0.04,  # optional: make it slimmer
        "height": 0.6  # optional: taller bar
    }
)
plotter.show()
