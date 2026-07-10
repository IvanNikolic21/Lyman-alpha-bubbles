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


# CDS byte-by-byte layout for the two-file catalog (`tb_lya.txt` +
# `sample_nirspec_properties.txt`), 0-indexed half-open colspecs for read_fwf.
_LYA_COLSPECS = [(0, 14), (15, 19), (20, 25), (26, 32), (33, 38), (39, 44), (45, 46),
                 (47, 53), (54, 59), (60, 65), (66, 67),
                 (68, 76), (77, 85), (86, 94), (95, 96),
                 (97, 105), (106, 114), (115, 123), (124, 125),
                 (126, 132), (133, 139), (140, 146),
                 (147, 153), (154, 160), (161, 167),
                 (168, 179), (180, 181)]
_LYA_COLNAMES = ['id', 'pid', 'zspec', 'ew_grat', 'ew_grat_lo', 'ew_grat_hi', 'ew_grat_lim',
                 'ew_prism', 'ew_prism_lo', 'ew_prism_hi', 'ew_prism_lim',
                 'fesc_grat', 'fesc_grat_lo', 'fesc_grat_hi', 'fesc_grat_lim',
                 'fesc_prism', 'fesc_prism_lo', 'fesc_prism_hi', 'fesc_prism_lim',
                 'fwhm', 'fwhm_lo', 'fwhm_hi', 'dv', 'dv_lo', 'dv_hi',
                 'reference', 'active']
_LYA_SKIPROWS = 125   # header + byte-by-byte description ends before the data block

_PROP_COLSPECS = [(0, 13), (14, 18), (19, 30), (31, 41), (42, 48), (49, 56), (57, 64),
                  (65, 70), (71, 76), (77, 83), (84, 89), (90, 95),
                  (96, 102), (103, 109), (110, 116), (117, 118)]
_PROP_COLNAMES = ['id', 'pid', 'ra', 'dec', 'zspec', 'magF150W', 'muv', 'muv_lo', 'muv_hi',
                  'logm', 'logm_lo', 'logm_hi', 'sSFR', 'sSFR_lo', 'sSFR_hi', 'active']
_PROP_SKIPROWS = 28

_EW_SENTINEL = -99.0   # "no measurement in this channel" (distinct from a reported limit)


def load_catalog_v2(lya_path: str, properties_path: str, z_min: float = 5.0,
                    muv_max: float = 0.0, prefer: str = 'grating') -> CatalogData:
    """Load the two-file CDS catalog (`tb_lya.txt` Lya EW table +
    `sample_nirspec_properties.txt` position/Muv table) and merge into a
    `CatalogData`, replacing the old single-file `table.dat` / `load_catalog`.

    The properties table's `active` flag deduplicates galaxies independently
    observed under both a SPURS/DIVER id and a JADES/GTO id (verified: e.g.
    `DIVER-1018968` and `JADES-1166` sit at identical RA/Dec) -- only
    `active == 1` rows are kept as the canonical galaxy list, matched by id
    (case-insensitive) into the Lya table for EW.

    `prefer` picks grating (R~1000) vs prism (R~100) EW when a galaxy has
    both -- grating is used when present and falls back to prism only when
    grating has no measurement at all (`ew_grat == -99`, distinct from a
    reported non-detection). `prefer='prism'` inverts the fallback order.
    """
    if prefer not in ('grating', 'prism'):
        raise ValueError(f"prefer must be 'grating' or 'prism', got {prefer!r}")

    df_lya  = pd.read_fwf(lya_path, colspecs=_LYA_COLSPECS, names=_LYA_COLNAMES,
                          skiprows=_LYA_SKIPROWS)
    df_prop = pd.read_fwf(properties_path, colspecs=_PROP_COLSPECS, names=_PROP_COLNAMES,
                          skiprows=_PROP_SKIPROWS)

    df_lya['id_norm']  = df_lya['id'].str.upper().str.strip()
    df_prop['id_norm'] = df_prop['id'].str.upper().str.strip()

    prop_active = df_prop[df_prop['active'] == 1]
    merged = prop_active.merge(df_lya, on='id_norm', how='inner', suffixes=('', '_lya'))
    n_unmatched = len(prop_active) - len(merged)
    if n_unmatched:
        print(f"[load_catalog_v2] {n_unmatched} active properties rows had no "
              f"matching Lya-table id and were dropped.", flush=True)

    has_grat  = merged['ew_grat'].to_numpy() != _EW_SENTINEL
    has_prism = merged['ew_prism'].to_numpy() != _EW_SENTINEL
    use_grat  = has_grat if prefer == 'grating' else ~has_prism
    has_any   = has_grat | has_prism

    ew     = np.where(use_grat, merged['ew_grat'], merged['ew_prism'])
    ew_lo  = np.where(use_grat, merged['ew_grat_lo'], merged['ew_prism_lo'])
    ew_hi  = np.where(use_grat, merged['ew_grat_hi'], merged['ew_prism_hi'])
    ew_lim = np.where(use_grat, merged['ew_grat_lim'], merged['ew_prism_lim']).astype(bool)

    ew_err = np.where((ew_lo != _EW_SENTINEL) & (ew_hi != _EW_SENTINEL),
                      (ew_lo + ew_hi) / 2.0, 1.0)   # placeholder for UL rows, unused downstream

    zspec = merged['zspec'].to_numpy()
    muv   = merged['muv'].to_numpy()

    n_total  = len(merged)
    sane_mask = (zspec > z_min) & (muv < muv_max) & (muv > -90)
    keep_mask = has_any & sane_mask

    print(f"[load_catalog_v2] {n_total} active rows total: "
          f"{(~has_any).sum()} dropped (no EW in either channel), "
          f"{(has_any & ~sane_mask).sum()} dropped (sanity filter: "
          f"zspec<={z_min} or muv unphysical), "
          f"{keep_mask.sum()} kept.", flush=True)
    print(f"[load_catalog_v2] EW source: {(use_grat & keep_mask).sum()} grating, "
          f"{(~use_grat & keep_mask).sum()} prism.", flush=True)

    is_upper_limit = ew_lim[keep_mask]
    print(f"[load_catalog_v2] of {keep_mask.sum()} kept: "
          f"{is_upper_limit.sum()} upper limits, "
          f"{(~is_upper_limit).sum()} detections.", flush=True)

    return CatalogData(
        id=merged['id'].to_numpy()[keep_mask],
        ra=merged['ra'].to_numpy()[keep_mask],
        dec=merged['dec'].to_numpy()[keep_mask],
        redshift=zspec[keep_mask],
        muv=muv[keep_mask],
        ew=ew[keep_mask],
        ew_err=ew_err[keep_mask],
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


def data_driven_priors(x, y, z, pad_factor: float = 1.3, r_min: float = 0.5,
                       r_max: float = None):
    """Prior box for (x, y, z, r_bub) sized from the actual galaxy positions.

    The (x, y, z) center bounds use each axis's own extent, not a shared
    `max(...)` across axes -- a deep pencil-beam survey's line-of-sight (z)
    depth is typically far larger than its transverse (x, y) footprint, and
    inflating the transverse prior to match it wastes prior volume on bubble
    centers far outside the surveyed sky area.

    `r_bub`'s ceiling returned here is the **single-bubble (n_bubs=1) budget**:
    a model with one bubble must be allowed to grow up to the full LOS extent
    so it has a genuine chance to span widely-separated clusters of detections
    -- capping it at the (much smaller) transverse extent would structurally
    guarantee it loses a Bayes-factor comparison to multi-bubble models
    regardless of what the data actually support. Multi-bubble models divide
    this budget by their bubble count (`r_max / n_bubs` in `real_data_run.py`'s
    prior transforms) so the comparison reflects an actual modeling tradeoff,
    not an artifact of an arbitrarily tighter prior. Pass `r_max` directly to
    override this default (e.g. for an explicit physically-motivated cap).
    """
    x_half = pad_factor * np.abs(x).max()
    y_half = pad_factor * np.abs(y).max()
    z_half = pad_factor * np.abs(z).max()

    if r_max is None:
        r_max = z_half

    prior_lo = np.array([-x_half, -y_half, -z_half, r_min])
    prior_hi = np.array([x_half, y_half, z_half, r_max])
    return prior_lo, prior_hi