"""
resample_pencil_pool.py -- re-sample an existing pencil-beam pool (make_pencil_pool.py) on a different sightline
grid, keeping EVERY mock the same universe.

A mock is fully described by the parameters stored in <source_dir>/mocks/*.npz: lightcone realization, source model
(m_min), ionizing efficiency (log_zeta7, alpha), and the placement of the survey footprint (rot, offset). From these
the ionization state anywhere in its lightcone follows (zeta(z) F >= 1, as in make_pencil_pool.py). Here each
galaxy's sightline is read off again, at the bin centres of a new grid (los_grid.py), so two pools made from the same
source differ ONLY in how finely the sightlines are sampled.

The sightline length (galaxy -> z = 5.3) is computed exactly as the likelihood does (lightcone_field.bin_z_edges):
comoving distance of the galaxy minus that of z = 5.3, so the bins of the pool and of the likelihood are the same
stretches of sky. (make_pencil_pool.py used the lightcone geometry's z = 5.3 offset, 0.25 cMpc different.)

Output, same layout as make_pencil_pool.py, plus the grid spec:
  <out_dir>/mocks/train_batch_XXXXX.npz : masks (n, 43, n_bins) uint8, 1 = ionized, column 0 at the z = 5.3 end;
                                          the source's mock parameters copied; realized recomputed; los_grid
  <out_dir>/pool/train_batch_XXXXX.npz  : theta = 1 - masks[:, :, ::-1] (column 0 nearest the galaxy), n_gal, n_los,
                                          los_grid
(xhi_local / xhi_global_local of the source are not carried over: they depend on the sightline points.)

--check_legacy instead re-creates the SOURCE's own 75 points ('legacy:75', with the source's sightline length) and
checks that the masks come out identical to the stored ones: a test that the re-sampling reproduces the pool.

Usage (JWST_inference env)
-----
python resample_pencil_pool.py --source_dir ~/Documents/Lya_bubbles/esf_pencils_forest \\
    --out_dir ~/Documents/Lya_bubbles/esf_pencils_forest_los/uniform_300 --los_grid uniform:300
python resample_pencil_pool.py --source_dir ~/Documents/Lya_bubbles/esf_pencils_forest --check_legacy
"""
import os
import glob
import time
import argparse

import numpy as np

import esf_lightcone as esf
import make_pencil_pool as mpp
import exact_pixel_likelihood as epl
import los_grid

COPIED_KEYS = ('x_ion', 'x_hi_7', 'z_end', 'tau_e', 'alpha', 'log_zeta7', 'm_min', 'realization', 'rot', 'offset',
               'delta5_gal')


def sightline_lengths_as_likelihood():
    """Galaxy -> z = 5.3 comoving length of every sightline [cMpc], exactly as lightcone_field.bin_z_edges."""
    cat = epl.load_real_catalog(os.path.join(epl.REPO, 'tb_lya.txt'),
                                os.path.join(epl.REPO, 'sample_nirspec_properties.txt'), 5.0, -18.0, 'grating', 7.3)
    d_c0 = epl.Cosmo.comoving_distance(cat['z0']).to(epl.u.Mpc).value
    d_c_end = epl.Cosmo.comoving_distance(epl.lf.Z_END_DEFAULT).to(epl.u.Mpc).value
    return (d_c0 + cat['z_gal']) - d_c_end, cat['z_gal']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_dir', required=True, help='pool made by make_pencil_pool.py (uses its mocks/)')
    ap.add_argument('--out_dir', help='where the re-sampled mocks/ and pool/ go')
    ap.add_argument('--los_grid', help="sightline grid spec, see los_grid.py, e.g. 'uniform:300'")
    ap.add_argument('--lc_dir', default=os.path.expanduser('~/Documents/Lya_bubbles/esf_lightcones'))
    ap.add_argument('--check_legacy', action='store_true', help='reproduce the source pool and compare')
    a = ap.parse_args()

    geometry = np.load(os.path.join(a.lc_dir, 'geometry.npz'))
    x_gal, y_gal, z_gal_lightcone = geometry['x_gal'], geometry['y_gal'], geometry['z_gal']
    n_gal = len(x_gal)

    if a.check_legacy:
        spec = 'legacy:75'
        lengths = z_gal_lightcone - float(geometry['z_end_offset'])    # the source pool's own sightline length
    else:
        if not (a.out_dir and a.los_grid):
            raise ValueError('--out_dir and --los_grid are needed (or --check_legacy)')
        spec = a.los_grid
        lengths, z_gal_catalog = sightline_lengths_as_likelihood()
        assert np.allclose(z_gal_catalog, z_gal_lightcone), 'catalog and lightcone geometry disagree'
        for sub in ('mocks', 'pool'):
            os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)

    n_bins = los_grid.n_bins(spec)
    # sightline sample positions in the lightcone's line-of-sight coordinate, nearest the galaxy first
    distances = np.stack([los_grid.sample_distances(spec, lengths[g]) for g in range(n_gal)])     # (n_gal, n_bins)
    ell_points = z_gal_lightcone[:, None] - distances
    print(f'[resample] grid {spec}: {n_bins} bins; first sample {distances[:, 0].min():.2f}-'
          f'{distances[:, 0].max():.2f} cMpc in front of the galaxies; widths '
          f'{np.diff(los_grid.bin_edges(spec, lengths.min())).min():.2f}-'
          f'{np.diff(los_grid.bin_edges(spec, lengths.max())).max():.2f} cMpc', flush=True)

    mock_files = sorted(glob.glob(os.path.join(a.source_dir, 'mocks', 'train_batch_*.npz')))
    n_checked = 0
    for path in mock_files:
        name = os.path.basename(path)
        if not a.check_legacy:
            out_pool = os.path.join(a.out_dir, 'pool', name)
            if os.path.exists(out_pool):
                continue                                         # resumable
        t0 = time.perf_counter()
        source = np.load(path)
        realization = int(source['realization'][0])
        assert np.all(source['realization'] == realization)
        lightcone = np.load(os.path.join(a.lc_dir, f'lc_{realization:03d}.npz'))
        log_density, ell, z_los = lightcone['logF'], lightcone['ell'], lightcone['z_los']
        cell_size = float(lightcone['dx'])
        n_transverse = log_density.shape[1]

        # same cell lookup as make_pencil_pool.py: nearest line-of-sight cell, floor in the transverse directions
        los_index = np.clip(np.round((ell_points - ell[0]) / cell_size).astype(int), 0, len(ell) - 1)
        z_cell = z_los[los_index]                                # (n_gal, n_bins)
        mass_index = np.searchsorted(esf.M_MIN, source['m_min'])

        n = len(source['m_min'])
        ionized_from_galaxy = np.zeros((n, n_gal, n_bins), np.uint8)
        for i in range(n):
            x_rot, y_rot = mpp.footprint_transform(x_gal, y_gal, int(source['rot'][i]))
            ix = np.floor((x_rot + source['offset'][i, 0]) / cell_size).astype(int) % n_transverse
            iy = np.floor((y_rot + source['offset'][i, 1]) / cell_size).astype(int) % n_transverse
            field = log_density[mass_index[i]][ix[:, None], iy[:, None], los_index].astype(np.float32)
            ionized_from_galaxy[i] = (field + source['alpha'][i] * np.log10((1 + z_cell) / 8)
                                      >= -source['log_zeta7'][i])
        masks = ionized_from_galaxy[:, :, ::-1]                  # column 0 at the z = 5.3 end, as the source pool

        if a.check_legacy:
            identical = np.array_equal(masks, source['masks'])
            print(f'[check] {name}: masks identical to the source: {identical} '
                  f'(differing entries: {np.count_nonzero(masks != source["masks"])} of {masks.size})', flush=True)
            n_checked += 1
            if n_checked == 2:                                   # two files are enough for the test
                break
            continue

        mock_out = {key: source[key] for key in COPIED_KEYS}
        np.savez(os.path.join(a.out_dir, 'mocks', name), masks=masks, realized=masks.mean(axis=(1, 2)),
                 los_grid=spec, **mock_out)
        theta = (1 - masks[:, :, ::-1]).reshape(n, -1).astype(np.uint8)
        tmp = out_pool[:-len('.npz')] + '.tmp.npz'
        np.savez(tmp, theta=theta, n_gal=n_gal, n_los=n_bins, los_grid=spec)
        os.replace(tmp, out_pool)                                # pool file appears only when complete
        print(f'[resample] {name}: {n} mocks in {time.perf_counter() - t0:.0f} s, '
              f'neutral fraction of the sightlines {1 - masks.mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
