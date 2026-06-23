import numpy as np
import astropy
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

RAs = [
    Angle('09h59m21.68s'),
    Angle('09h59m11.46s'),
    Angle('09h59m09.13s'),
    Angle('09h58m45.34s'),
    Angle('09h59m12.35s'),
    Angle('09h59m06.73s'),
    Angle('09h59m23.62s'),
    Angle('09h59m17.62s'),
    Angle('09h59m06.33s'),
    Angle('09h59m01.40s'),
    Angle('09h59m09.76s'),
    Angle('09h58m46.20s')
]

DECs = [
    Angle('02d14m53.02s'),
    Angle('02d18m10.42s'),
    Angle('02d18m22.38s'),
    Angle('02d18m28.87s'),
    Angle('02d18m28.86s'),
    Angle('02d22m45.93s'),
    Angle('02d23m32.73s'),
    Angle('02d23m41.24s'),
    Angle('02d26m30.48s'),
    Angle('02d28m02.28s'),
    Angle('02d28m32.95s'),
    Angle('02d28m45.76s'),
]

zs = np.array([
    6.882,
    6.701,
    6.702,
    6.732,
    6.73,
    6.748,
    6.814,
    6.759,
    6.847,
    6.750,
    6.81,
    6.67
])

def get_ENDSTA_gals():
    """
    Function returns data from Endsley&Stark 2021.
    :return:
    x_data: numpy array,
        X-positions based on RA of galaxies.
    y_data: numpy array,
        Y-position based on DEC of galaxies.
    z_data: numpy array,
        Z-position based on redshifts of galaxies.
    """
    ra_c = Angle('09h59m10s')
    dec_c = Angle('02d25m20s')
    z_c = 6.76
    dist_c = Cosmo.comoving_distance(z_c)

    # flat-sky approximation
    x_data = np.zeros((len(RAs)))
    y_data = np.zeros((len(RAs)))
    z_data = np.zeros((len(RAs)))
    for i, (r, d, z) in enumerate(zip(RAs, DECs, zs)):
        cosx = np.sin(
            d.to(u.deg)
        ) * np.sin(
            dec_c.to(u.deg)
        ) + np.cos(
            d.to(u.deg)
        ) * np.cos(dec_c.to(u.deg))
        cosy = np.sin(dec_c.to(u.deg))**2 + np.cos(
            dec_c.to(u.deg)
        )**2 * np.cos(r.to(u.deg)-ra_c.to(u.deg))
        dist = Cosmo.comoving_distance(z)
        if d > dec_c:
            x_data[i] = (np.sqrt(
                dist**2 + dist_c**2 - 2 * dist * dist_c * cosx)).value
        else:
            x_data[i] = - (np.sqrt(
                dist ** 2 + dist_c ** 2 - 2 * dist * dist_c * cosx)).value
        if r > ra_c:
            y_data[i] = (np.sqrt(
                dist**2 + dist_c**2 - 2 * dist * dist_c * cosy)).value
        else:
            y_data[i] = - (np.sqrt(
                dist ** 2 + dist_c ** 2 - 2 * dist * dist_c * cosy)).value

        z_data[i] = (dist-dist_c).value

    return x_data, y_data, z_data
