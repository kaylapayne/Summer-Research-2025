from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import pyvista as pv
import matplotlib.pyplot as plt

print("Opening FITS file...")
# Load redshift catalog
zcat = fits.open("zall-pix-guadalupe.fits")[1].data

print("Extracting z, ra, dec data...")
mask = (zcat["ZWARN"] == 0) & (zcat["Z"] > 0.01) # good redshifts
z = zcat["Z"][mask]
ra = zcat["TARGET_RA"][mask]
dec = zcat["TARGET_DEC"][mask]

# Convert to comoving distances
r = cosmo.comoving_distance(z).value
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=r*u.Mpc)
print("Converting to cartesian coordinates...")
x, y, z = coords.cartesian.x.value, coords.cartesian.y.value, coords.cartesian.z.value

print("Preparing data for plotting...")
cloud = pv.PolyData(np.column_stack((x, y, z)))
cloud["redshift"] = z

plotter = pv.Plotter()
plotter.add_points(cloud, render_points_as_spheres=True, point_size=2, scalars="redshift", cmap="viridis")
plotter.show()
