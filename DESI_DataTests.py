import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy import units as u
from Corrfunc.theory import DD
from scipy.optimize import curve_fit

# Load DESI data
def readDESIfitsData(fileName):
    with fits.open(fileName) as hdul:
        data = hdul[1].data
        ra = data['RA']
        dec = data['DEC']
        z = data['Z']
    return(ra, dec, z)

def getBinData(ra, dec, z, z_min, z_max, n_max):
    mask = (z > z_min) & (z < z_max)
    ra_bin = ra[mask]
    dec_bin = dec[mask]
    z_bin = z[mask]
    n = min(len(ra_bin), n_max)
    mask_subsample = np.random.choice(len(ra), n, replace=False)
    ra_bin = ra_bin[mask_subsample]
    dec_bin = dec_bin[mask_subsample]
    z_bin = z_bin[mask_subsample]
    return(ra_bin, dec_bin, z_bin)

def getDistances(coords):
    sep_matrix = np.zeros((len(coords), len(coords)))
    # Compute separations manually
    for i in range(len(coords)):
        sep_matrix[i, :] = coords[i].separation(coords).deg

    # now convert the upper triangular matrix (not including main diag) to 1d array.
    # This will give 1 value for each pair of points
    i_upper = np.triu_indices_from(sep_matrix, k=1)
    separation_data = sep_matrix[i_upper]
    return(separation_data)

ra_dat, dec_dat, z_dat = readDESIfitsData("BGS_ANY_NGC_clustering.dat.fits")
ra_ran, dec_ran, z_ran = readDESIfitsData("BGS_ANY_NGC_1_clustering.ran.fits")

ra_dat, dec_dat, z_dat = getBinData(ra_dat, dec_dat, z_dat, 0.1, 0.11, 5000)
ra_ran, dec_ran, z_ran = getBinData(ra_ran, dec_ran, z_ran, 0.1, 0.11, 5000)

coords_dat = SkyCoord(ra=ra_dat * u.deg, dec=dec_dat * u.deg, frame='icrs')
coords_ran = SkyCoord(ra=ra_ran * u.deg, dec=dec_ran * u.deg, frame='icrs')

distances_dat = getDistances(coords_dat)
distances_ran = getDistances(coords_ran)


