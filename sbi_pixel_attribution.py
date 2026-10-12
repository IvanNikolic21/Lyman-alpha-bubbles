"""
sbi_pixel_attribution.py -- which galaxy drives which part of the pixel-field posterior?

THE QUESTION
------------
The real-data posterior P(neutral) map is built from all 43 observations at once. To see what each
galaxy contributes, we recompute the posterior with that galaxy's information REMOVED and compare:

    influence_S(g', k) = P(theta_{g'k} = 1 | x_obs)  -  P(theta_{g'k} = 1 | x_obs without set S)

where S is one galaxy (default: every galaxy in turn) or a group of galaxies (--groups).
Positive influence: the data of S make pixel (g', k) MORE neutral. Negative: more ionized.

HOW A GALAXY'S INFORMATION IS REMOVED
-------------------------------------
The network was trained on the full 86-number x, so we cannot simply drop two entries. Instead we use
that the likelihood of the remaining data is the full likelihood averaged over the removed data:

    p(x_rest | theta) = Int p(x_S, x_rest | theta) dx_S.

The network estimates the ratio r(theta, x) = p(x | theta) / p(x). Averaging it over replacement
values x_S' drawn from q(x_S') gives

    r_bar(theta) = (1/M) sum_m r(theta, [x_S'_m, x_rest])
                 ~ Int q(x_S') p(x_S', x_rest | theta) / p(x_S', x_rest) dx_S',

which equals p(x_rest | theta) / p(x_rest) exactly when q is the predictive distribution of x_S given
the other observations, p(x_S | x_rest). We use instead the MARGINAL predictive p(x_S): the values of
(EW, upper-limit flag) of the galaxies in S taken from M randomly chosen simulations of the pool.
APPROXIMATION: this ignores the correlation between x_S and x_rest. It is accurate when S carries
information mostly about its own surroundings; for a group, all its galaxies are replaced with values
from the SAME simulation, so correlations WITHIN the group are kept.

Built-in null test: if S is uninformative, replacing x_S changes nothing, r_bar = r, and the influence
is zero up to Monte Carlo noise (which shrinks as 1/sqrt(M)).

THE POSTERIOR ITSELF
--------------------
Exactly as `sbi_pixel_field.py infer`: score every pool map theta_i (prior draws, the `val` split)
with the ratio estimator, w_i = r_i / sum_j r_j (self-normalised importance weights), and
P(theta_{gk} = 1 | x) = sum_i w_i theta_{i,gk}. As a self-check, the full-data posterior is recomputed
here and, if --check_infer points at the existing pixel_infer.npz, compared with it.

Cluster-only (torch + sbi). Cost: (1 + n_sets * M) forward passes over the pool. With the default
43 single-galaxy sets and M = 32 that is ~1400 passes of 10 000 maps: minutes on a GPU.

Usage
-----
python sbi_pixel_attribution.py --n_los 75 \\
    --ratio sbi_runs/pixel_mock_gagnon_hartman/ratio_estimator.pt \\
    --pool_dir bubble_mocks_run1_x_gagnon_hartman --pool_split val \\
    --check_infer sbi_runs/pixel_mock_gagnon_hartman/pixel_infer.npz \\
    --groups 14-18 27-36 21,22,24,25 \\
    --lya_catalog tb_lya.txt --properties_catalog sample_nirspec_properties.txt \\
    --device cuda --output_dir sbi_runs/pixel_mock_gagnon_hartman

Then plot locally with plot_pixel_attribution.py.
"""
import os
import time
import argparse

import numpy as np

import real_data_run as rdr                      # catalog loading (fills rdr._S)
import sbi_pixel_field as spf                    # load_sims(), _add_catalog_args()
from sbi_real_data import _build_x_obs           # the real observation x_obs, same as `infer`


def parse_groups(tokens, n_gal):
    """'14-18' -> [14..18]; '21,22,24' -> [21, 22, 24]; '3' -> [3]. Galaxy indices are rows of the
    posterior map, i.e. the catalog order of rdr._S (z-sorted for GOODS-N)."""
    groups = []
    for tok in tokens:
        members = []
        for part in tok.split(','):
            if '-' in part:
                a, b = (int(v) for v in part.split('-'))
                members.extend(range(a, b + 1))
            else:
                members.append(int(part))
        if min(members) < 0 or max(members) >= n_gal:
            raise ValueError(f"group {tok!r} has galaxy indices outside 0..{n_gal - 1}")
        groups.append(sorted(set(members)))
    return groups


def log_mean_exp(a, axis):
    """log( mean(exp(a)) ) along `axis`, computed stably (subtract the max before exponentiating)."""
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def posterior_from_log_ratio(log_r, theta_grid):
    """Self-normalised importance weights and the marginal P(neutral) map, as in `infer`."""
    w = np.exp(log_r - log_r.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)                                  # effective sample size of the weights
    return np.tensordot(w, theta_grid, axes=(0, 0)), ess        # (n_gal, n_los), scalar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    spf._add_catalog_args(ap)                                   # catalog paths, z-window, --n_los, ...
    ap.add_argument('--ratio', type=str, required=True, help='ratio_estimator.pt from train_nre.')
    ap.add_argument('--pool_dir', type=str, required=True, help='(theta, x) sims used as the prior pool.')
    ap.add_argument('--pool_split', type=str, default='val', choices=['train', 'val'])
    ap.add_argument('--n_draws', type=int, default=32,
                    help='M: replacement draws per removed set (Monte Carlo noise ~ 1/sqrt(M)).')
    ap.add_argument('--groups', type=str, nargs='*', default=[],
                    help="Extra removal sets, removed JOINTLY, e.g. 14-18 27-36 21,22,24,25.")
    ap.add_argument('--no_singles', action='store_true',
                    help='Skip the one-galaxy-at-a-time sets (only run --groups).')
    ap.add_argument('--check_infer', type=str, default=None,
                    help='pixel_infer.npz of the same network and pool: the full-data posterior '
                         'recomputed here must match it.')
    ap.add_argument('--batch', type=int, default=10000, help='Pool maps per forward pass.')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output_dir', type=str, required=True)
    args = ap.parse_args()

    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but CUDA is not available; use cpu.")

    # ---- trained network ----
    ck = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator, n_gal, n_los = ck['ratio_estimator'], ck['n_gal'], ck['n_los']

    # ---- real catalog -> x_obs (identical to sbi_pixel_field.run_infer) ----
    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    if len(rdr._S.x_gal) != n_gal:
        raise ValueError(f"catalog has {len(rdr._S.x_gal)} galaxies, the network {n_gal}")
    x_obs = _build_x_obs().astype(np.float32)                  # (2 n_gal,) = [EWs, upper-limit flags]

    # ---- prior pool: maps theta_i and their simulated observations x_i ----
    theta_pool, x_pool, pg, pl = spf.load_sims(args.pool_dir, args.pool_split)
    if (pg, pl) != (n_gal, n_los):
        raise ValueError(f"pool ({pg} x {pl}) does not match the network ({n_gal} x {n_los})")
    n_pool = len(theta_pool)
    theta_grid = theta_pool.reshape(n_pool, n_gal, n_los)
    theta_t = torch.as_tensor(theta_pool, dtype=torch.float32, device=args.device)

    def log_ratio(x_vec):
        """log r(theta_i, x_vec) for every pool map theta_i, in batches of --batch maps."""
        x_t = torch.as_tensor(x_vec, dtype=torch.float32, device=args.device)
        out = np.empty(n_pool, dtype=np.float64)
        with torch.no_grad():
            for a in range(0, n_pool, args.batch):
                th = theta_t[a:a + args.batch]
                out[a:a + len(th)] = ratio_estimator(
                    theta=th, x=x_t.unsqueeze(0).expand(len(th), -1)).squeeze(-1).cpu().numpy()
        return out

    # ---- full-data posterior (and the self-check against `infer`) ----
    lr_full = log_ratio(x_obs)
    p_full, ess_full = posterior_from_log_ratio(lr_full, theta_grid)
    print(f"[attr] full posterior: ESS {ess_full:.1f} of {n_pool}", flush=True)
    if args.check_infer:
        ref = np.load(args.check_infer)
        dlr = np.abs((lr_full - lr_full.max()) - (ref['log_ratio'] - ref['log_ratio'].max())).max()
        dmap = np.abs(p_full - ref['marginal_map']).max()
        print(f"[attr] check vs {args.check_infer}: max |d log r| {dlr:.2e}, max |d P| {dmap:.2e} "
              f"(both should be ~0; otherwise network, pool or catalog differ from that run)", flush=True)

    # ---- removal sets ----
    sets = [] if args.no_singles else [[g] for g in range(n_gal)]
    sets += parse_groups(args.groups, n_gal)
    labels = [str(s[0]) if len(s) == 1 else ','.join(map(str, s)) for s in sets]
    rng = np.random.default_rng(args.seed)

    p_minus = np.empty((len(sets), n_gal, n_los), dtype=np.float32)
    ess_minus = np.empty(len(sets))
    draw_idx = np.empty((len(sets), args.n_draws), dtype=np.int64)
    t0 = time.perf_counter()
    for si, members in enumerate(sets):
        cols = np.array(members + [n_gal + g for g in members])  # EW entries and flag entries of S
        idx = rng.integers(n_pool, size=args.n_draws)            # M pool simulations supply x_S'
        draw_idx[si] = idx
        lr = np.empty((args.n_draws, n_pool))
        for m, j in enumerate(idx):
            x_mod = x_obs.copy()
            x_mod[cols] = x_pool[j, cols]                        # replace x_S by simulation j's values
            lr[m] = log_ratio(x_mod)
        lr_bar = log_mean_exp(lr, axis=0)                        # log of the averaged ratio, per pool map
        p_minus[si], ess_minus[si] = posterior_from_log_ratio(lr_bar, theta_grid)
        if (si + 1) % max(1, len(sets) // 10) == 0 or si == len(sets) - 1:
            el = time.perf_counter() - t0
            print(f"[attr] {si + 1}/{len(sets)} sets ({el:.0f}s); last: remove [{labels[si]}] -> "
                  f"ESS {ess_minus[si]:.0f}, max |influence| {np.abs(p_full - p_minus[si]).max():.3f}",
                  flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'pixel_attribution.npz')
    np.savez(out, p_full=p_full, p_minus=p_minus, influence=p_full[None] - p_minus,
             ess_full=ess_full, ess_minus=ess_minus, set_labels=np.array(labels),
             set_members=np.array([np.pad(s, (0, n_gal - len(s)), constant_values=-1) for s in sets]),
             x_obs=x_obs, n_draws=args.n_draws, draw_idx=draw_idx, log_ratio_full=lr_full,
             redshifts=np.asarray(rdr._S.redshifts), n_gal=n_gal, n_los=n_los)
    print(f"[attr] saved {out}", flush=True)


if __name__ == '__main__':
    main()
