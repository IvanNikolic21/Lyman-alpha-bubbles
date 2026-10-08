"""
Simulation-based calibration (SBC) for the pixel-field SBI (NRE + pool importance resampling).

A rank test on the full ~3225-dim binary theta is intractable, but SBC holds for ANY scalar
function f(theta): if the inference is calibrated, the rank of f(theta_true) among f(posterior
samples) is uniform over held-out instances.

WHAT THE DATA CAN CONSTRAIN. Each galaxy contributes essentially ONE number (its Lya transmission),
and the damping wing is dominated by the neutral gas closest to the galaxy (tau ~ int x_HI dr / r^2).
Pixels far from every galaxy are NOT constrained by Lya even with perfect data; whatever the
posterior says there comes from correlations built into the simulator (global ionization level,
box-walk geometry). Note: calibration is NOT made pessimistic by such pixels -- a posterior equal to
the prior is perfectly calibrated -- but global / far statistics then test the SIMULATOR'S prior
structure, not information extracted from Lya, and averaging them with near-source pixels dilutes
any real miscalibration. So statistics are split into two groups and reported separately:

 Lya-informed (what the data actually measure):
  dw_g       per-galaxy damping-wing-weighted neutral fraction, w_k ~ 1/(r_k + r_v)^2 over the
             sightline bins (r_k = bin-centre distance from the galaxy, r_v ~ 2 cMpc ~ 200 km/s offset);
             SBC ranks pooled over all galaxies x queries  -> the per-galaxy "one number" test
  dw_mean    sample mean of dw_g over galaxies
  near       neutral fraction in the first N_NEAR bins (~50 cMpc)
  pair_agree agreement of near-source states for close galaxy pairs (cross-sightline structure)
  TARP       coverage test on the near-source block (first N_NEAR bins of every sightline) only

 Depth sweep ("how far from the galaxies can we trust / learn anything?"):
  tarp_cum   TARP on the block of the first N bins of every sightline, for N in N_SWEEP
  tarp_shell TARP on the shell of bins between consecutive N_SWEEP entries (isolates each distance)
  per-bin    ranks / posterior std of the across-galaxy neutral fraction in each bin k, and the
             importance-weighted posterior P(neutral) map of every query (post_p) with its truth
             (true_grid) -> per-bin Brier skill vs the prior, computed in plot_pixel_sbc_depth.py.
  Calibration alone does NOT mean "precise": a prior-like posterior passes TARP too. Precision at a
  given distance = calibrated AND informative (posterior std < prior std, Brier skill > 0).

 Prior-structure (tests the simulator's coupling, NOT Lya information):
  glob, slab_0..3   global / per-slab neutral fractions (slab 1-3 lie far from the galaxies)


Posterior samples are drawn EXACTLY as `infer` does for the real data: the query's x is scored with
the trained ratio estimator against a DISJOINT pool, and samples are drawn by importance resampling
(SIR) with the normalized weights -- so this tests the whole posterior construction, not just the
network. A small pool / low ESS shows up as too-narrow posteriors (U-shaped ranks).

Ranks of discrete statistics are tie-broken uniformly at random (standard for SBC with ties).

Cluster-only (torch/sbi), forward passes only.

Usage
-----
python sbi_pixel_sbc.py --n_los 75 \\
    --ratio sbi_runs/pixel_v2/ratio_estimator.pt \\
    --pool_dir sbi_runs/pixel_v2/sim --pool_split val \\
    --n_queries 1000 --n_post 200 \\
    --lya_catalog tb_lya.txt --properties_catalog sample_nirspec_properties.txt \\
    --output_dir sbi_runs/pixel_v2

Then plot locally with plot_pixel_sbc.py (numpy/matplotlib only).
"""
import os
import time
import argparse

import numpy as np

import real_data_run as rdr
import sbi_pixel_field as spf   # load_sims(), _add_catalog_args()

N_NEAR = 5
SLAB_EDGES = (0, 19, 38, 57, 75)
N_SWEEP = (1, 2, 3, 5, 8, 12, 19, 28, 38, 57, 75)   # cumulative TARP blocks; shells between entries
R_V_CMPC = 2.0      # ~200 km/s Lya velocity offset at z~7 in comoving distance (softens 1/r^2 at r -> 0)


def dw_weights(bin_width_cmpc, n_los):
    """Damping-wing weights per galaxy and bin: w_k ~ 1/(r_k + R_V)^2, normalized per galaxy.
    bin_width_cmpc: (n_gal,) bin width of each galaxy's sightline (equal-comoving bins)."""
    r = (np.arange(n_los)[None, :] + 0.5) * bin_width_cmpc[:, None]
    w = 1.0 / (r + R_V_CMPC) ** 2
    return w / w.sum(axis=1, keepdims=True)                               # (n_gal, n_los)


def dw_per_galaxy(theta_grid, W):
    return (theta_grid * W[None]).sum(axis=2)                             # (n, n_gal)


def statistics(theta_grid, pairs, W):
    """theta_grid: (n, n_gal, n_los) binary (1 = neutral). Returns (n, n_stat) array
    (order = STAT_NAMES). Per-galaxy dw_g is returned separately by dw_per_galaxy()."""
    near = theta_grid[:, :, :N_NEAR].mean(axis=2)                      # (n, n_gal)
    cols = [dw_per_galaxy(theta_grid, W).mean(axis=1), near.mean(axis=1)]
    for a, b in zip(SLAB_EDGES[:-1], SLAB_EDGES[1:]):
        cols.append(theta_grid[:, :, a:b].mean(axis=(1, 2)))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        cols.append(1.0 - np.abs(near[:, i] - near[:, j]).mean(axis=1))
    else:
        cols.append(np.zeros(len(theta_grid)))
    cols.append(theta_grid.mean(axis=(1, 2)))                             # glob (prior-structure)
    return np.column_stack(cols)


STAT_NAMES = (['dw_mean', 'near'] + [f'slab_{k}' for k in range(len(SLAB_EDGES) - 1)]
              + ['pair_agree', 'glob'])
STAT_GROUP = {'dw_mean': 'lya', 'near': 'lya', 'pair_agree': 'lya', 'glob': 'prior',
              'slab_0': 'lya', 'slab_1': 'prior', 'slab_2': 'prior', 'slab_3': 'prior'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    spf._add_catalog_args(ap)
    ap.add_argument('--ratio', type=str, required=True)
    ap.add_argument('--pool_dir', type=str, required=True)
    ap.add_argument('--pool_split', type=str, default='val', choices=['train', 'val'])
    ap.add_argument('--n_queries', type=int, default=1000)
    ap.add_argument('--n_post', type=int, default=200, help='SIR posterior samples per query.')
    ap.add_argument('--pair_sep', type=float, default=8.0,
                    help='Max transverse separation [cMpc] for the pair-agreement statistic.')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but CUDA is not available; use cpu.")

    ck = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator, n_gal, n_los = ck['ratio_estimator'], ck['n_gal'], ck['n_los']
    if n_los != SLAB_EDGES[-1]:
        raise ValueError(f"SLAB_EDGES assume n_los=75, ratio estimator has {n_los}.")

    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    s = rdr._S
    if len(s.x_gal) != n_gal:
        raise ValueError(f"Catalog has {len(s.x_gal)} galaxies, ratio estimator {n_gal}.")
    from astropy.cosmology import Planck18 as _cosmo
    import astropy.units as _u
    z_end = 5.3                                                           # = lightcone_field Z_END_DEFAULT
    d_gal = _cosmo.comoving_distance(np.asarray(s.redshifts)).to(_u.Mpc).value
    d_end = _cosmo.comoving_distance(z_end).to(_u.Mpc).value
    bin_width = (d_gal - d_end) / n_los
    W = dw_weights(bin_width, n_los)
    print(f"[sbc] damping-wing weights: first {N_NEAR} bins carry "
          f"{W[:, :N_NEAR].sum(axis=1).mean():.1%} of the weight (mean over galaxies).", flush=True)
    dx = s.x_gal[:, None] - s.x_gal[None, :]; dy = s.y_gal[:, None] - s.y_gal[None, :]
    sep = np.hypot(dx, dy)
    pairs = np.argwhere(np.triu(sep < args.pair_sep, k=1))
    print(f"[sbc] {len(pairs)} galaxy pairs closer than {args.pair_sep} cMpc (transverse).", flush=True)

    theta_all, x_all, pg, pl = spf.load_sims(args.pool_dir, args.pool_split)
    if (pg, pl) != (n_gal, n_los):
        raise ValueError(f"pool sims ({pg}x{pl}) don't match the ratio estimator ({n_gal}x{n_los}).")
    n_total = len(theta_all)
    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, max(1, n_total // 2))
    query_idx = rng.choice(n_total, size=n_queries, replace=False)
    pool_mask = np.ones(n_total, bool); pool_mask[query_idx] = False
    pool_idx = np.where(pool_mask)[0]
    theta_pool = theta_all[pool_idx].astype(np.float32)
    pool_grid = theta_pool.reshape(-1, n_gal, n_los)
    pool_stats = statistics(pool_grid, pairs, W)                                # (n_pool, n_stat)
    pool_dw = dw_per_galaxy(pool_grid, W)                                       # (n_pool, n_gal)
    pool_near_block = pool_grid[:, :, :N_NEAR].reshape(len(pool_grid), -1)      # TARP on informed block
    pool_col = pool_grid.mean(axis=1)                                           # (n_pool, n_los) per-bin fraction
    pool_flat = theta_pool.reshape(len(theta_pool), -1)
    n_cut = np.array(N_SWEEP)
    theta_pool_t = torch.as_tensor(theta_pool, dtype=torch.float32, device=args.device)
    print(f"[sbc] {n_queries} queries vs a disjoint {len(pool_idx)}-instance pool; "
          f"{args.n_post} SIR samples each; statistics: {STAT_NAMES}", flush=True)

    n_stat = len(STAT_NAMES)
    ranks = np.empty((n_queries, n_stat))           # fractional rank in [0, 1]
    true_stats = np.empty((n_queries, n_stat)); post_mean = np.empty((n_queries, n_stat))
    post_std = np.empty((n_queries, n_stat)); ess = np.empty(n_queries)
    tarp_f = np.empty(n_queries)
    ranks_dw = np.empty((n_queries, n_gal))     # per-galaxy damping-wing statistic ranks
    tarp_cum = np.empty((n_queries, len(N_SWEEP))); tarp_shell = np.empty((n_queries, len(N_SWEEP)))
    ranks_bin = np.empty((n_queries, n_los)); post_std_bin = np.empty((n_queries, n_los))
    post_p = np.empty((n_queries, n_gal, n_los), np.float32)    # importance-weighted P(neutral) maps
    true_grid = np.empty((n_queries, n_gal, n_los), np.uint8)

    t0 = time.perf_counter()
    with torch.no_grad():
        for qi in range(n_queries):
            q = query_idx[qi]
            x_q = torch.as_tensor(x_all[q], dtype=torch.float32, device=args.device)
            lr = ratio_estimator(theta=theta_pool_t, x=x_q.unsqueeze(0).expand(len(theta_pool), -1))
            lr = lr.squeeze(-1).detach().cpu().numpy()
            w = np.exp(lr - lr.max()); w /= w.sum()
            ess[qi] = 1.0 / np.sum(w ** 2)
            pick = rng.choice(len(w), size=args.n_post, replace=True, p=w)       # SIR posterior samples

            t_true = theta_all[q].astype(np.float32)
            t_grid = t_true.reshape(1, n_gal, n_los)
            f_true = statistics(t_grid, pairs, W)[0]
            f_post = pool_stats[pick]
            less = (f_post < f_true).sum(axis=0); equal = (f_post == f_true).sum(axis=0)
            ranks[qi] = (less + rng.random(n_stat) * (equal + 1)) / (args.n_post + 1)
            true_stats[qi], post_mean[qi], post_std[qi] = f_true, f_post.mean(0), f_post.std(0)

            dw_true = dw_per_galaxy(t_grid, W)[0]; dw_post = pool_dw[pick]
            less = (dw_post < dw_true).sum(axis=0); equal = (dw_post == dw_true).sum(axis=0)
            ranks_dw[qi] = (less + rng.random(n_gal) * (equal + 1)) / (args.n_post + 1)

            # TARP on the Lya-informed block only (first N_NEAR bins of every sightline)
            ref = pool_near_block[rng.integers(len(pool_near_block))]
            d_post = np.sqrt(((pool_near_block[pick] - ref) ** 2).sum(axis=1))
            d_true = np.sqrt(((t_grid[0, :, :N_NEAR].ravel() - ref) ** 2).sum())
            tarp_f[qi] = np.mean(d_post < d_true)

            # depth sweep: one reference map per query, squared distances accumulated bin by bin
            ref_g = pool_grid[rng.integers(len(pool_grid))]
            sq_post = np.cumsum(((pool_grid[pick] - ref_g) ** 2).sum(axis=1), axis=1)   # (n_post, n_los)
            sq_true = np.cumsum(((t_grid[0] - ref_g) ** 2).sum(axis=0))                # (n_los,)
            c_post, c_true = sq_post[:, n_cut - 1], sq_true[n_cut - 1]
            prev = np.concatenate([[0], n_cut[:-1]]) - 1
            s_post = c_post - np.where(prev >= 0, sq_post[:, np.maximum(prev, 0)], 0)
            s_true = c_true - np.where(prev >= 0, sq_true[np.maximum(prev, 0)], 0)
            # ties (identical distances, common for binary maps) broken at random
            tarp_cum[qi] = ((c_post < c_true).sum(0) + rng.random(len(n_cut)) * ((c_post == c_true).sum(0))) / args.n_post
            tarp_shell[qi] = ((s_post < s_true).sum(0) + rng.random(len(n_cut)) * ((s_post == s_true).sum(0))) / args.n_post

            col_true = t_grid[0].mean(axis=0); col_post = pool_col[pick]
            less = (col_post < col_true).sum(axis=0); equal = (col_post == col_true).sum(axis=0)
            ranks_bin[qi] = (less + rng.random(n_los) * (equal + 1)) / (args.n_post + 1)
            post_std_bin[qi] = col_post.std(0)
            post_p[qi] = (w @ pool_flat).reshape(n_gal, n_los)
            true_grid[qi] = t_grid[0]

            if (qi + 1) % max(1, n_queries // 20) == 0:
                el = time.perf_counter() - t0
                print(f"[sbc] {qi + 1}/{n_queries} ({(qi + 1) / el:.2f}/s, "
                      f"ETA {(n_queries - qi - 1) / ((qi + 1) / el):.0f}s)", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'pixel_sbc.npz')
    np.savez(out, ranks=ranks, true_stats=true_stats, post_mean=post_mean, post_std=post_std,
             ess=ess, tarp_f=tarp_f, stat_names=np.array(STAT_NAMES), n_post=args.n_post,
             stat_groups=np.array([STAT_GROUP[n] for n in STAT_NAMES]), ranks_dw=ranks_dw, dw_weights=W,
             n_queries=n_queries, n_pool=len(pool_idx), n_pairs=len(pairs), query_idx=query_idx,
             n_sweep=n_cut, tarp_cum=tarp_cum, tarp_shell=tarp_shell, ranks_bin=ranks_bin,
             post_std_bin=post_std_bin, prior_std_bin=pool_col.std(0), prior_p_bin=pool_grid.mean(axis=(0, 1)),
             post_p=post_p, true_grid=true_grid, bin_width=bin_width)

    # quick text summary: coverage of central 68% / 95% intervals per statistic
    print(f"[sbc] median ESS {np.median(ess):.0f}")
    rd = ranks_dw.ravel()
    print(f"[sbc] {'dw_g (pooled)':>14s}: coverage 68% -> {np.mean(np.abs(rd - 0.5) < 0.34):.3f}, "
          f"95% -> {np.mean(np.abs(rd - 0.5) < 0.475):.3f}; mean rank {rd.mean():.3f}  [lya]")
    for k, name in enumerate(STAT_NAMES):
        r = ranks[:, k]
        c68 = np.mean(np.abs(r - 0.5) < 0.34); c95 = np.mean(np.abs(r - 0.5) < 0.475)
        print(f"[sbc] {name:>14s}: coverage 68% -> {c68:.3f}, 95% -> {c95:.3f}; "
              f"mean rank {r.mean():.3f} (0.5 ideal)  [{STAT_GROUP[name]}]")
    print(f"[sbc] saved {out}", flush=True)


if __name__ == '__main__':
    main()
