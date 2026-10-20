"""
compare_exact_nn_v2.py -- exact-likelihood vs network posterior on the SAME pool, per intrinsic EW model, with the
emphasis on z > 6.8 (the target of the v2 prior: both should agree there), plus the hyperparameter posteriors and
the evidence of each EW model.

Inputs per EW model
-------------------
  exact : exact_pixel_likelihood_chunked.py combine output (weights w, maps, hyper_* prior parameters, evidence)
  nn    : sbi_pixel_field.py infer output, pixel_infer.npz (weights, log_ratio, marginal_map)
The network pool must be the exact pool, or its val part: with combine --splits train val (1.2M maps) and
infer --pool_split val (200k), the val maps are the last rows of the exact pool, in the same order. Then
  - quantities defined map by map (correlation, weight overlap, the network's hyperparameter weights) use the val
    rows, with the exact weights restricted to them and renormalised (= the exact posterior on the val pool);
  - the exact maps and hyperparameter posteriors use the whole pool (the more precise estimate of the same
    posterior; the network map is its estimate on the val pool).
Compare the network only with exact files WITHOUT --tau_prior (the network has no tau prior).

What is compared
----------------
maps     P(neutral) of every galaxy x sightline bin, exact vs network, and their difference. Bins are placed at their
         redshifts (lyabubbles.lightcone_field.bin_z_edges, as in the likelihood), so x = redshift.
numbers  over the bins with centre z > --z_cut (default 6.8) and over all bins: mean and max |dP|, mean P(neutral);
         ESS of both; corr(L_exact, log r_nn) over the pool; weight overlap sum_i min(w_exact_i, w_nn_i)
         (1 = identical posteriors, 0 = disjoint).
hyper    weighted histograms of the mock-prior parameters (x_hi_7, z_end, zeta, f_host, tau_e for v2) under the prior,
         the exact posterior and the network posterior; median and 68% interval of each.
evidence ln Z of each EW model from the exact likelihood, and ln Z differences relative to the first model given.

Usage
-----
python compare_exact_nn_v2.py out=compare_v2 exponential=exact_v2_exponential_val.npz:pixel_infer_exp.npz \\
       [tang_muv=...:...] [z_cut=6.8]
Output: <out>_maps.png, <out>_hyper.png, and a printed table.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

import exact_pixel_likelihood as epl
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

INK, MUTED = '#0b0b0b', '#8a8986'
COLORS = ['#2a78d6', '#1a9e77', '#d81b60']                   # fixed order: model 1, 2, 3 (plot_exact_hyperparams)
HYPER = [('x_hi_7', r'$x_{\rm HI}(z=7)$'), ('z_end', r'$z_{\rm end}$'), ('zeta', r'$\zeta$ (gradual $\to$ rapid)'),
         ('f_host', r'$f_{\rm host}$'), ('tau_e', r'$\tau_e$'),
         ('x_ion', r'$x_{\rm ion}$'), ('k_factor', r'$k$ factor')]   # the last two: v1 pools


def wquantiles(v, w, qs):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return np.interp(np.asarray(qs) * c[-1], c, v[o])


def network_rows(ex, n_nn):
    """Indices of the exact pool rows the network was evaluated on: all rows, or the val rows (the last ones)."""
    n = len(ex['w'])
    if n_nn == n:
        return np.arange(n)
    files, counts = ex['pool_files'], ex['pool_counts']
    is_val = np.repeat([str(f).startswith('val_') for f in files], counts)
    if is_val.sum() == n_nn and is_val[-n_nn:].all():
        return np.arange(n - n_nn, n)
    raise ValueError(f"network pool has {n_nn} maps; the exact pool has {n} of which {is_val.sum()} val -- "
                     f"run infer on the same pool (or on --pool_split val of it)")


def evidence(ex):
    """ln Z and its error; recomputed from L for combine files written before log_evidence was stored."""
    if 'log_evidence' in ex.files:
        return float(ex['log_evidence']), float(ex['log_evidence_err'])
    lp = ex['log_prior_tau']
    n = len(ex['L'])
    return (float(logsumexp(ex['L'] + lp) - logsumexp(lp)),
            float(np.sqrt(max(1 / float(ex['ess_full']) - 1 / n, 0.0))))


def bin_edges_z(n_los, cat):
    """(n_gal, n_los + 1) redshift edges of every galaxy's sightline bins, bin 0 at the galaxy (theta order)."""
    d_c0 = Cosmo.comoving_distance(cat['z0']).to(u.Mpc).value
    out = []
    for g in range(len(cat['z_gal'])):
        z_b, z_e = epl.lf.bin_z_edges(n_los, cat['z_gal'][g], d_c0, z_end=epl.lf.Z_END_DEFAULT)
        out.append(np.r_[z_b[0], z_e])
    return np.array(out)


def main():
    opts, runs = {'out': 'compare_v2', 'z_cut': '6.8'}, []
    for a in sys.argv[1:]:
        k, v = a.split('=', 1)
        if k in opts:
            opts[k] = v
        else:
            ex, _, nn = v.partition(':')
            runs.append((k, os.path.expanduser(ex), os.path.expanduser(nn) if nn else None))
    if not runs:
        sys.exit(__doc__)
    z_cut = float(opts['z_cut'])
    cat = epl.load_real_catalog(os.path.join(epl.REPO, 'tb_lya.txt'),
                                os.path.join(epl.REPO, 'sample_nirspec_properties.txt'), 5.0, -18.0, 'grating', 7.3)

    loaded = []
    for name, ex_path, nn_path in runs:
        ex = np.load(ex_path)
        nn = np.load(nn_path) if nn_path else None
        rows = network_rows(ex, len(nn['weights'])) if nn is not None else None
        if nn is not None and np.isfinite(ex['tau_prior']).all():
            print(f"WARNING {name}: the exact file has a tau prior, the network does not -- not a like-for-like "
                  f"comparison")
        loaded.append((name, ex, nn, rows))
    n_gal, n_los = int(loaded[0][1]['n_gal']), int(loaded[0][1]['n_los'])
    edges = bin_edges_z(n_los, cat)                          # (n_gal, n_los + 1), decreasing along each row
    zc = 0.5 * (edges[:, 1:] + edges[:, :-1])
    hi_z = zc > z_cut

    # ---- numbers ----
    print(f"\n{'model':16s} {'ESS ex':>8s} {'ESS nn':>8s} {'corr':>6s} {'overlap':>8s} | z>{z_cut}: "
          f"{'P ex':>5s} {'P nn':>5s} {'mean|dP|':>8s} {'max|dP|':>7s} | all: {'mean|dP|':>8s} {'max|dP|':>7s}")
    for name, ex, nn, rows in loaded:
        if nn is None:
            print(f"{name:16s} {float(ex['ess_full']):8.1f}   (no network file)")
            continue
        we = ex['w'][rows] / ex['w'][rows].sum()             # exact posterior on the network's pool
        wn = nn['weights'].astype(float)
        Lr = ex['L'][rows]
        good = np.isfinite(Lr)
        corr = np.corrcoef(Lr[good], nn['log_ratio'][good])[0, 1]
        dP = nn['marginal_map'] - ex['p_full']
        if len(rows) < len(ex['w']):
            print(f"{'':16s} (exact ESS on the full pool; on the network's {len(rows)} maps: {1 / np.sum(we ** 2):.1f}; "
                  f"corr over the {good.mean():.1%} maps with finite L)")
        print(f"{name:16s} {float(ex['ess_full']):8.1f} {float(nn['ess']):8.1f} {corr:6.3f} "
              f"{np.minimum(we, wn).sum():8.3f} |        {ex['p_full'][hi_z].mean():5.3f} "
              f"{nn['marginal_map'][hi_z].mean():5.3f} {np.abs(dP[hi_z]).mean():8.3f} {np.abs(dP[hi_z]).max():7.3f} "
              f"|      {np.abs(dP).mean():8.3f} {np.abs(dP).max():7.3f}")

    # ---- evidence ----
    lz0 = evidence(loaded[0][1])[0]
    print(f"\n{'model':16s} {'ln Z':>10s} {'+-':>6s} {'ln Z - ln Z(' + loaded[0][0] + ')':>24s}")
    for name, ex, _, _ in loaded:
        lz, err = evidence(ex)
        print(f"{name:16s} {lz:10.3f} {err:6.3f} {lz - lz0:24.3f}")
    if any(np.isfinite(ex['tau_prior']).all() for _, ex, _, _ in loaded):
        print("(tau prior applied in at least one file: compare evidences only between files with the same prior)")

    # ---- hyperparameters ----
    keys = [(k, lab) for k, lab in HYPER if f'hyper_{k}' in loaded[0][1].files]
    if keys:
        print(f"\n{'parameter':10s} {'model':16s} {'prior':>22s} {'exact':>22s} {'network':>22s}   (median [16, 84])")
        fig, axes = plt.subplots(len(loaded), len(keys), figsize=(3.0 * len(keys), 2.4 * len(loaded)),
                                 squeeze=False)
        for r, (name, ex, nn, rows) in enumerate(loaded):
            prior_w = np.exp(ex['log_prior_tau'] - ex['log_prior_tau'].max())
            col = COLORS[r % len(COLORS)]
            for c, (k, lab) in enumerate(keys):
                v = ex[f'hyper_{k}'].astype(float)
                ax = axes[r, c]
                bins = np.linspace(v.min(), v.max(), 41)
                ax.hist(v, bins=bins, weights=prior_w, density=True, histtype='stepfilled', color=MUTED,
                        alpha=0.35, label='prior')
                ax.hist(v, bins=bins, weights=ex['w'], density=True, histtype='step', lw=1.8, color=col,
                        label='exact')
                cells = [wquantiles(v, prior_w, [0.5, 0.16, 0.84]), wquantiles(v, ex['w'], [0.5, 0.16, 0.84])]
                if nn is not None:
                    ax.hist(v[rows], bins=bins, weights=nn['weights'].astype(float), density=True,
                            histtype='step', lw=1.8, ls='--', color=col, label='network')
                    cells.append(wquantiles(v[rows], nn['weights'].astype(float), [0.5, 0.16, 0.84]))
                print(f"{k:10s} {name:16s} " + ' '.join(f"{q[0]:7.3f} [{q[1]:6.3f},{q[2]:6.3f}]" for q in cells))
                ax.set_xlabel(lab, color=INK)
                ax.set_yticks([])
                for s in ('top', 'right', 'left'):
                    ax.spines[s].set_visible(False)
                if c == 0:
                    ax.set_ylabel(name, color=INK)
                if r == 0 and c == 0:
                    ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{opts['out']}_hyper.png", dpi=150)
        print(f"saved {opts['out']}_hyper.png")

    # ---- maps: exact, network, network - exact; x = redshift ----
    n_col = 3 if any(nn is not None for _, _, nn, _ in loaded) else 1
    fig, axes = plt.subplots(len(loaded), n_col, figsize=(5.0 * n_col, 0.12 * n_gal * len(loaded) + 1.5),
                             squeeze=False, sharex=True)
    for r, (name, ex, nn, _) in enumerate(loaded):
        panels = [(ex['p_full'], 'Blues', 0, 1, f'{name}: exact P(neutral)')]
        if nn is not None:
            panels += [(nn['marginal_map'], 'Blues', 0, 1, f'{name}: network P(neutral)'),
                       (nn['marginal_map'] - ex['p_full'], 'RdBu_r', -0.5, 0.5, 'network - exact')]
        for c in range(len(panels), n_col):                 # no network file for this model: no empty axes
            axes[r, c].axis('off')
        for c, (val, cmap, lo, hi, title) in enumerate(panels):
            ax = axes[r, c]
            for g in range(n_gal):                           # one strip per galaxy, bins at their redshifts
                mesh = ax.pcolormesh(edges[g, ::-1], [g, g + 1], val[g, ::-1][None, :], cmap=cmap, vmin=lo, vmax=hi)
            ax.plot(cat['z_gal_red'], np.arange(n_gal) + 0.5, '.', color=INK, ms=3)   # the galaxies
            ax.axvline(z_cut, color=INK, lw=0.8, ls=':')
            ax.set_ylim(n_gal, 0)
            ax.set_title(title, fontsize=9, color=INK)
            ax.set_ylabel('galaxy (catalog row)', color=INK)
            if r == len(loaded) - 1:
                ax.set_xlabel('redshift', color=INK)
            fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(f"{opts['out']}_maps.png", dpi=150)
    print(f"saved {opts['out']}_maps.png")


if __name__ == '__main__':
    main()
