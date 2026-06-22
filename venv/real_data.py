"""
Ingestion of real observational catalogs for bubble inference.

Handles the RA/Dec -> comoving-Mpc coordinate transform, data-driven prior
derivation, and parsing of a fixed-width EW catalog with upper limits.
"""
import dataclasses

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as Cosmo
import astropy.units as u

# Byte ranges (1-indexed, inclusive) from the catalog's CDS byte-by-byte
# description, converted to 0-indexed half-open colspecs for read_fwf.
_COLSPECS = [(0, 17), (18, 31), (32, 36), (37, 48), (49, 59),
             (60, 66), (67, 73), (74, 83), (84, 91), (92, 99)]
_COLNAMES = ['id', 'name', 'pid', 'ra', 'dec', 'zspec', 'muv',
             'ew', 'ewerr', 'ew_type']


@dataclasses.dataclass
class CatalogData:
    id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    redshift: np.ndarray
    muv: np.ndarray
    ew: np.ndarray
    ew_err: np.ndarray
    is_upper_limit: np.ndarray


def load_catalog(path: str, z_min: float = 5.0, muv_max: float = 0.0) -> CatalogData:
    """Load a fixed-width EW catalog, dropping unusable/unphysical rows.

    A row is dropped if it has no EW measurement at all (`ew` is nan), or if
    it fails a sanity check (`zspec <= z_min`, or `muv` unphysical: >= muv_max
    or an unmeasured sentinel like -99). Among kept rows, `ewerr <= 0` (e.g.
    the catalog's -1.00 convention) flags an EW **upper limit** rather than a
    detection with a real error bar.
    """
    df = pd.read_fwf(path, colspecs=_COLSPECS, names=_COLNAMES)

    n_total = len(df)
    nan_mask = df['ew'].isna().to_numpy()
    sane_mask = ((df['zspec'] > z_min)
                 & (df['muv'] < muv_max)
                 & (df['muv'] > -90)).to_numpy()
    keep_mask = ~nan_mask & sane_mask

    print(f"[load_catalog] {n_total} rows total: "
          f"{nan_mask.sum()} dropped (no EW measurement), "
          f"{(~nan_mask & ~sane_mask).sum()} dropped (sanity filter: "
          f"zspec<={z_min} or muv unphysical), "
          f"{keep_mask.sum()} kept.", flush=True)

    kept = df[keep_mask]
    ewerr = kept['ewerr'].to_numpy()
    is_upper_limit = ewerr <= 0

    print(f"[load_catalog] of {keep_mask.sum()} kept: "
          f"{is_upper_limit.sum()} upper limits, "
          f"{(~is_upper_limit).sum()} detections.", flush=True)

    return CatalogData(
        id=kept['id'].to_numpy(),
        ra=kept['ra'].to_numpy(),
        dec=kept['dec'].to_numpy(),
        redshift=kept['zspec'].to_numpy(),
        muv=kept['muv'].to_numpy(),
        ew=kept['ew'].to_numpy(),
        ew_err=ewerr,
        is_upper_limit=is_upper_limit,
    )


def radec_to_comoving(ra, dec, redshift):
    """Flat-sky RA/Dec/z -> comoving-Mpc Cartesian, centered on the sample.

    Transverse (x, y) use each galaxy's own comoving distance, which is the
    exact (not small-angle-hacked) transverse comoving distance in a flat
    cosmology. z is the centered line-of-sight comoving offset, matching the
    `z_gal` convention already used throughout the model (a comoving-Mpc LOS
    offset, positive = farther from the observer / higher redshift).
    """
    ra0 = float(np.mean(ra))
    dec0 = float(np.mean(dec))
    z0 = float(np.mean(redshift))

    d_c = Cosmo.comoving_distance(redshift).to(u.Mpc).value
    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value

    x = d_c * np.radians(ra - ra0) * np.cos(np.radians(dec0))
    y = d_c * np.radians(dec - dec0)
    z = d_c - d_c0

    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()
    x = x - x_mean
    y = y - y_mean
    z = z - z_mean

    return x, y, z, ra0, dec0, z0, x_mean, y_mean, z_mean


def comoving_to_radec(x, y, z, ra0, dec0, z0, x_mean=0.0, y_mean=0.0, z_mean=0.0):
    """Inverse of `radec_to_comoving`: centered comoving Mpc -> (RA, Dec, redshift).

    For a point (or array of points) in the same centered frame `radec_to_comoving`
    produces (e.g. a fitted bubble center, or points on its surface for plotting).
    `x_mean, y_mean, z_mean` are the centering offsets `radec_to_comoving` returned
    for the galaxy sample that frame was built from -- pass 0 (default) only if
    those weren't kept and the sample is tight enough that the approximation is
    acceptable for a quick plot.
    """
    from astropy.cosmology import z_at_value

    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value
    d_c_target = (z + z_mean) + d_c0

    redshift = np.array([
        z_at_value(Cosmo.comoving_distance, d_c_i * u.Mpc)
        for d_c_i in d_c_target
    ])

    ra  = ra0 + np.degrees((x + x_mean) / (d_c_target * np.cos(np.radians(dec0))))
    dec = dec0 + np.degrees((y + y_mean) / d_c_target)

    return ra, dec, redshift


def data_driven_priors(x, y, z, pad_factor: float = 1.3, r_min: float = 0.5):
    """Prior box for (x, y, z, r_bub) sized from the actual galaxy positions."""
    half_extent = pad_factor * max(
        np.abs(x).max(), np.abs(y).max(), np.abs(z).max()
    )
    prior_lo = np.array([-half_extent, -half_extent, -half_extent, r_min])
    prior_hi = np.array([half_extent, half_extent, half_extent, half_extent])
    return prior_lo, prior_hi