"""
exact_pixel_sbc.py -- simulation-based calibration (SBC) of the EXACT-likelihood posterior on the full (chunked)
pool, e.g. the 1.2M-map v2 prior. This is the posterior behind exact_pixel_likelihood_chunked.py's real-data
results (hyperparameter posteriors, P(neutral) maps, posterior_bubble_fields.py's representative fields), so its
calibration is what those results rest on. (sbi_pixel_sbc.py tests the NETWORK posterior on the val pool.)

SBC: draw query mocks (theta*_q, x_q) from the prior, compute the posterior for each x_q exactly as for the real
data, and check that the truth looks like a posterior draw: for any scalar f(theta), the rank of f(theta*_q)
among posterior samples is uniform over queries (Talts+18). Here
  - the queries are n_queries maps of the val split; each query is excluded from its own pool (the rest of
    train + val is its prior sample), so the truth is never "found" in the pool;
  - the posterior is the self-normalised importance weighting of the pool by exp(sum_g l_g(theta_i; x_q)), with
    l_g from exact_pixel_likelihood.loglik_quadrature, as in exact_pixel_likelihood_chunked.py;
  - hyperparameter ranks (x_hi_7, z_end, zeta, f_host, tau_e, x_ion, ...; from --mocks_dir) use the weighted
    posterior CDF directly: u = sum_i w_i [f_i < f*] + U * sum_i w_i [f_i = f*], U ~ U(0,1) (ties);
  - map statistics (dw_g, dw_mean, near, slabs, pair_agree, glob, TARP, depth sweep) are those of sbi_pixel_sbc.py,
    from n_post SIR samples, with the SAME definitions, so plot_pixel_sbc.py / plot_pixel_sbc_depth.py read the
    output unchanged.

Three stages (like exact_pixel_likelihood_chunked.py):
1. queries  -- pick the query rows (val split), save their x, true maps and true hyperparameters. Light.
2. loglik   -- SLURM array over pool files: per file, the transmissions are computed once per galaxy; upper limits
               share one log-likelihood vector per galaxy (threshold 3 sigma_g of the catalog, as the simulator);
               detections need one quadrature per (query, detected galaxy). Output per pool file:
               <output_dir>/sbcll_<split>_batch_XXXXX.npz with logL (n_queries, n_maps) float32.
3. combine  -- weights per query, hyperparameter ranks, SIR samples, one streaming pass over theta (posterior
               mean maps, SIR sample maps, TARP reference maps), statistics and ranks -> one npz.

Usage
-----
python exact_pixel_sbc.py queries --pool_dir bubble_mocks_v2_run1_x_exponential --mocks_dir bubble_mocks_v2_run1 \\
    --n_queries 300 --output_dir sbc_exact_v2_exponential
python exact_pixel_sbc.py loglik --ew_model exponential --pool_dir bubble_mocks_v2_run1_x_exponential \\
    --output_dir sbc_exact_v2_exponential --task_id $SLURM_ARRAY_TASK_ID --n_tasks 40
python exact_pixel_sbc.py combine --pool_dir bubble_mocks_v2_run1_x_exponential --mocks_dir bubble_mocks_v2_run1 \\
    --output_dir sbc_exact_v2_exponential
(see exact_sbc_v2.sh / exact_sbc_v2_combine.sh)
"""
import os
import time
import argparse

import numpy as np

import exact_pixel_likelihood as epl
from exact_pixel_likelihood_chunked import pool_files, add_catalog_args, HYPER_KEYS

QUERIES = 'sbc_queries.npz'
# scalar hyperparameters ranked (whichever exist in the mock files; v1 has x_ion and k_factor)
RANK_KEYS = ('x_hi_7', 'z_end', 'zeta', 'f_host', 'tau_e', 'x_ion', 'z_mid', 'dz', 'k_factor')


def sbcll_path(output_dir, pool_path):
    return os.path.join(output_dir, 'sbcll_' + os.path.basename(pool_path))


def load_hyper(mocks_dir, files):
    """Scalar mock-prior parameters in pool order (as exact_pixel_likelihood_chunked.run_combine)."""
    per = {k: [] for k in HYPER_KEYS}
    for _, p in files:
        m = np.load(os.path.join(mocks_dir, os.path.basename(p)))
        for k in HYPER_KEYS:
            if k in m.files:
                per[k].append(m[k])
    return {k: np.concatenate(v) for k, v in per.items() if len(v) == len(files) and k in RANK_KEYS}


# ---------------------------------------------------------------- 1. queries
def run_queries(args):
    files = pool_files(args.pool_dir, args.splits)
    counts = np.array([len(np.load(p)['x']) for _, p in files])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    is_q_split = np.array([s == args.query_split for s, _ in files])
    cand = np.concatenate([np.arange(a, a + n) for a, n, ok in zip(starts, counts, is_q_split) if ok])
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(cand, size=args.n_queries, replace=False))     # global pool rows of the queries
    f_of = np.searchsorted(starts, rows, side='right') - 1
    x_q, theta_q = [], []
    for f in np.unique(f_of):
        d = np.load(files[f][1])
        loc = rows[f_of == f] - starts[f]
        x_q.append(d['x'][loc])
        theta_q.append(d['theta'][loc])
        n_gal, n_los = int(d['n_gal']), int(d['n_los'])
        if args.mocks_dir:                    # theta = 1 - masks (sightline axis reversed): right mocks for this pool
            masks = np.load(os.path.join(args.mocks_dir, os.path.basename(files[f][1])))['masks'][loc]
            if not np.array_equal(theta_q[-1].reshape(len(loc), n_gal, n_los), 1 - masks[:, :, ::-1]):
                raise ValueError(f"{files[f][1]}: pool maps differ from the --mocks_dir masks")
    hyper = load_hyper(args.mocks_dir, files) if args.mocks_dir else {}
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, QUERIES), rows=rows, x_q=np.concatenate(x_q),
             true_grid=np.concatenate(theta_q).reshape(-1, n_gal, n_los).astype(np.uint8),
             pool_files=np.array([os.path.basename(p) for _, p in files]), pool_counts=counts,
             n_gal=n_gal, n_los=n_los, seed=args.seed, **{f'true_{k}': v[rows] for k, v in hyper.items()})
    x = np.concatenate(x_q)
    print(f"[queries] {len(rows)} queries from the {args.query_split} split of a {counts.sum()}-map pool; "
          f"{(x[:, n_gal:] == 0).sum(axis=1).mean():.1f} detections per query on average; "
          f"hyperparameters: {', '.join(hyper) or 'none'}", flush=True)


# ---------------------------------------------------------------- 2. loglik
def run_loglik(args):
    if not 0 <= args.task_id < args.n_tasks:
        raise ValueError(f"--task_id {args.task_id} must be in [0, --n_tasks {args.n_tasks})")
    Q = np.load(os.path.join(args.output_dir, QUERIES))
    n_gal = int(Q['n_gal'])
    ew_q, ul_q = Q['x_q'][:, :n_gal].astype(float), Q['x_q'][:, n_gal:] == 1
    cat = epl.load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                                args.prefer, args.z_hi)
    if len(cat['muv']) != n_gal:
        raise ValueError(f"catalog has {len(cat['muv'])} galaxies, queries {n_gal}")
    sigma = cat['sigma']
    model = epl.GalaxyModel(cat, args.ew_model, 1, 0)      # only its deterministic methods are used
    dists = [model.ew_distribution(g) for g in range(n_gal)]

    files = pool_files(args.pool_dir, args.splits)
    if [os.path.basename(p) for _, p in files] != list(Q['pool_files']):
        raise ValueError("--pool_dir/--splits give a different pool than the queries were drawn from")
    mine = [f for j, f in enumerate(files) if j % args.n_tasks == args.task_id]
    print(f"[sbc-loglik] task {args.task_id}/{args.n_tasks}: {len(mine)} of {len(files)} pool files, "
          f"{len(ew_q)} queries ({(~ul_q).sum()} detections), ew_model={args.ew_model}", flush=True)
    for split, path in mine:
        out = sbcll_path(args.output_dir, path)
        if os.path.exists(out):
            print(f"[sbc-loglik] {out} exists, skipping (resumable).", flush=True)
            continue
        t0 = time.perf_counter()
        d = np.load(path)
        n_los = int(d['n_los'])
        theta_grid = d['theta'].reshape(-1, n_gal, n_los).astype(np.float64)
        logL = np.zeros((len(ew_q), len(theta_grid)))
        for g in range(n_gal):
            T = model.transmission(g, theta_grid[:, g, :], n_los, args.n_dv)       # query-independent
            if ul_q[:, g].any():                              # upper limit: threshold 3 sigma_g, shared
                logL[ul_q[:, g]] += epl.loglik_quadrature(T, None, True, sigma[g], dists[g], args.n_y)[None, :]
            for qi in np.where(~ul_q[:, g])[0]:               # detection: depends on the reported EW
                logL[qi] += epl.loglik_quadrature(T, ew_q[qi, g], False, sigma[g], dists[g], args.n_y)
        model._tr_cache.clear()                               # transmissions of this file only
        tmp = out[:-len('.npz')] + '.tmp.npz'
        np.savez(tmp, logL=logL.astype(np.float32), ew_model=args.ew_model, n_dv=args.n_dv, n_y=args.n_y,
                 pool_file=os.path.basename(path))
        os.replace(tmp, out)
        print(f"[sbc-loglik] {os.path.basename(path)}: {len(theta_grid)} maps x {len(ew_q)} queries in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    print(f"[sbc-loglik] task {args.task_id} done", flush=True)


# ---------------------------------------------------------------- 3. combine
def weighted_rank(f, w, f_true, rng):
    """u = P_post(f < f*) + U * P_post(f = f*): uniform over queries if the posterior is calibrated."""
    return w[f < f_true].sum() + rng.random() * w[f == f_true].sum()


def run_combine(args):
    import pixel_sbc_stats as sps        # statistics, dw weights, N_NEAR, N_SWEEP: same as sbi_pixel_sbc.py
    from astropy.cosmology import Planck18 as Cosmo
    import astropy.units as u

    Q = np.load(os.path.join(args.output_dir, QUERIES))
    rows, true_grid = Q['rows'], Q['true_grid'].astype(np.float32)
    n_gal, n_los = int(Q['n_gal']), int(Q['n_los'])
    nq = len(rows)
    files = pool_files(args.pool_dir, args.splits)
    counts = Q['pool_counts']
    N = int(counts.sum())
    rng = np.random.default_rng(args.seed)

    # ---- log-likelihoods (nq, N); each query excluded from its own pool ----
    missing = [p for _, p in files if not os.path.exists(sbcll_path(args.output_dir, p))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(files)} sbcll files missing "
                                f"(first {os.path.basename(missing[0])}) -- is the loglik stage still running?")
    logL = np.concatenate([np.load(sbcll_path(args.output_dir, p))['logL'] for _, p in files], axis=1)
    logL = logL.astype(np.float64)                           # float64 weights: SIR needs sum(p) = 1 exactly
    ew_model = str(np.load(sbcll_path(args.output_dir, files[0][1]))['ew_model'])
    logL[np.arange(nq), rows] = -np.inf
    W = np.exp(logL - logL.max(axis=1, keepdims=True))
    W /= W.sum(axis=1, keepdims=True)
    del logL
    ess = 1.0 / (W ** 2).sum(axis=1)
    print(f"[sbc-combine] {nq} queries x {N} pool maps ({ew_model}); exact ESS median {np.median(ess):.0f}, "
          f"min {ess.min():.0f}", flush=True)

    # ---- hyperparameter ranks from the weighted posterior CDF ----
    hyper = load_hyper(args.mocks_dir, files) if args.mocks_dir else {}
    hyper_names = [k for k in RANK_KEYS if k in hyper]
    hyper_ranks = np.empty((nq, len(hyper_names)))
    hyper_true = np.empty((nq, len(hyper_names)))
    hyper_mean = np.empty((nq, len(hyper_names)))
    hyper_std = np.empty((nq, len(hyper_names)))
    for j, k in enumerate(hyper_names):
        f = hyper[k]
        if not np.allclose(f[rows], Q[f'true_{k}']):
            raise ValueError(f"--mocks_dir {k} differs from the queries' truth: wrong mocks_dir")
        for qi in range(nq):
            hyper_ranks[qi, j] = weighted_rank(f, W[qi], f[rows[qi]], rng)
            m = W[qi] @ f
            hyper_true[qi, j], hyper_mean[qi, j] = f[rows[qi]], m
            hyper_std[qi, j] = np.sqrt(max(W[qi] @ (f - m) ** 2, 0.0))

    # ---- SIR samples and TARP reference maps (pool rows), gathered in one theta pass ----
    picks = np.array([rng.choice(N, size=args.n_post, replace=True, p=W[qi])
                      for qi in range(nq)])                                   # (nq, n_post)
    refs = rng.integers(N, size=(nq, 2))                                      # near-block and depth-sweep refs
    need = np.unique(np.concatenate([picks.ravel(), refs.ravel()]))
    got = np.empty((len(need), n_gal * n_los), np.uint8)
    post_p = np.zeros((nq, n_gal * n_los))
    col_sum = np.zeros(n_los); col_sq = np.zeros(n_los); pix_sum = 0.0      # prior per-bin fraction moments
    a = 0
    t0 = time.perf_counter()
    for k, ((_, p), n) in enumerate(zip(files, counts)):
        theta = np.load(p)['theta']
        if len(theta) != n:
            raise ValueError(f"{p}: {len(theta)} maps, the queries' pool had {n}")
        post_p += W[:, a:a + n] @ theta
        sel = (need >= a) & (need < a + n)
        got[sel] = theta[need[sel] - a]
        col = theta.reshape(n, n_gal, n_los).mean(axis=1)
        col_sum += col.sum(0); col_sq += (col ** 2).sum(0); pix_sum += theta.sum()
        a += n
        if (k + 1) % 50 == 0:
            print(f"[sbc-combine] {k + 1}/{len(files)} files streamed ({time.perf_counter() - t0:.0f} s)", flush=True)
    lookup = lambda r: got[np.searchsorted(need, r)].reshape(*np.shape(r), n_gal, n_los).astype(np.float32)

    # ---- map statistics, exactly as sbi_pixel_sbc.main (same names, same tie-breaking) ----
    cat = epl.load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                                args.prefer, args.z_hi)
    d_gal = Cosmo.comoving_distance(np.asarray(cat['z_gal_red'])).to(u.Mpc).value
    bin_width = (d_gal - Cosmo.comoving_distance(5.3).to(u.Mpc).value) / n_los
    W_dw = sps.dw_weights(bin_width, n_los)
    # transverse positions as real_data_run._load_catalog_and_priors (same loader, window and transform)
    c2 = epl.rd.load_catalog_v2(args.lya_catalog, args.properties_catalog, z_min=args.z_min,
                                muv_max=args.muv_max, prefer=args.prefer)
    win = c2.redshift <= args.z_hi
    xg, yg = epl.rd.radec_to_comoving(c2.ra[win], c2.dec[win], c2.redshift[win])[:2]
    sep = np.hypot(xg[:, None] - xg[None, :], yg[:, None] - yg[None, :])
    pairs = np.argwhere(np.triu(sep < args.pair_sep, k=1))
    n_stat, n_cut = len(sps.STAT_NAMES), np.array(sps.N_SWEEP)
    ranks = np.empty((nq, n_stat)); true_stats = np.empty((nq, n_stat))
    post_mean = np.empty((nq, n_stat)); post_std = np.empty((nq, n_stat))
    ranks_dw = np.empty((nq, n_gal)); tarp_f = np.empty(nq)
    tarp_cum = np.empty((nq, len(n_cut))); tarp_shell = np.empty((nq, len(n_cut)))
    ranks_bin = np.empty((nq, n_los)); post_std_bin = np.empty((nq, n_los))
    for qi in range(nq):
        samp = lookup(picks[qi])                                             # (n_post, n_gal, n_los)
        t_grid = true_grid[qi][None]
        f_true, f_post = sps.statistics(t_grid, pairs, W_dw)[0], sps.statistics(samp, pairs, W_dw)
        less, equal = (f_post < f_true).sum(0), (f_post == f_true).sum(0)
        ranks[qi] = (less + rng.random(n_stat) * (equal + 1)) / (args.n_post + 1)
        true_stats[qi], post_mean[qi], post_std[qi] = f_true, f_post.mean(0), f_post.std(0)
        dw_true, dw_post = sps.dw_per_galaxy(t_grid, W_dw)[0], sps.dw_per_galaxy(samp, W_dw)
        less, equal = (dw_post < dw_true).sum(0), (dw_post == dw_true).sum(0)
        ranks_dw[qi] = (less + rng.random(n_gal) * (equal + 1)) / (args.n_post + 1)
        ref = lookup(refs[qi, 0])[:, :sps.N_NEAR].ravel()                    # TARP on the near block
        d_post = np.sqrt(((samp[:, :, :sps.N_NEAR].reshape(args.n_post, -1) - ref) ** 2).sum(axis=1))
        d_true = np.sqrt(((t_grid[0, :, :sps.N_NEAR].ravel() - ref) ** 2).sum())
        tarp_f[qi] = np.mean(d_post < d_true)
        ref_g = lookup(refs[qi, 1])                                          # depth sweep
        sq_post = np.cumsum(((samp - ref_g) ** 2).sum(axis=1), axis=1)
        sq_true = np.cumsum(((t_grid[0] - ref_g) ** 2).sum(axis=0))
        c_post, c_true = sq_post[:, n_cut - 1], sq_true[n_cut - 1]
        prev = np.concatenate([[0], n_cut[:-1]]) - 1
        s_post = c_post - np.where(prev >= 0, sq_post[:, np.maximum(prev, 0)], 0)
        s_true = c_true - np.where(prev >= 0, sq_true[np.maximum(prev, 0)], 0)
        tarp_cum[qi] = ((c_post < c_true).sum(0) + rng.random(len(n_cut)) * (c_post == c_true).sum(0)) / args.n_post
        tarp_shell[qi] = ((s_post < s_true).sum(0) + rng.random(len(n_cut)) * (s_post == s_true).sum(0)) / args.n_post
        col_true, col_post = t_grid[0].mean(axis=0), samp.mean(axis=1)
        less, equal = (col_post < col_true).sum(0), (col_post == col_true).sum(0)
        ranks_bin[qi] = (less + rng.random(n_los) * (equal + 1)) / (args.n_post + 1)
        post_std_bin[qi] = col_post.std(0)

    out = os.path.join(args.output_dir, args.output)
    prior_mean_bin = col_sum / N
    np.savez(out, ranks=ranks, true_stats=true_stats, post_mean=post_mean, post_std=post_std, ess=ess,
             tarp_f=tarp_f, stat_names=np.array(sps.STAT_NAMES), n_post=args.n_post,
             stat_groups=np.array([sps.STAT_GROUP[n] for n in sps.STAT_NAMES]), ranks_dw=ranks_dw,
             dw_weights=W_dw, n_queries=nq, n_pool=N - 1, n_pairs=len(pairs), query_idx=rows, n_sweep=n_cut,
             tarp_cum=tarp_cum, tarp_shell=tarp_shell, ranks_bin=ranks_bin, post_std_bin=post_std_bin,
             prior_std_bin=np.sqrt(np.maximum(col_sq / N - prior_mean_bin ** 2, 0)),
             prior_p_bin=np.full(n_los, pix_sum / (N * n_gal * n_los)),
             post_p=post_p.reshape(nq, n_gal, n_los).astype(np.float32), true_grid=true_grid.astype(np.uint8),
             bin_width=bin_width, hyper_names=np.array(hyper_names), hyper_ranks=hyper_ranks,
             hyper_true=hyper_true, hyper_mean=hyper_mean, hyper_std=hyper_std, ew_model=ew_model,
             posterior='exact')
    print(f"[sbc-combine] saved {out}", flush=True)
    for j, k in enumerate(hyper_names):
        r = hyper_ranks[:, j]
        print(f"   {k:9s} coverage 68%: {np.mean(np.abs(r - 0.5) < 0.34):.2f}  95%: {np.mean(np.abs(r - 0.5) < 0.475):.2f}"
              f"  mean rank {r.mean():.3f}", flush=True)
    for j, k in enumerate(sps.STAT_NAMES):
        r = ranks[:, j]
        print(f"   {k:9s} coverage 68%: {np.mean(np.abs(r - 0.5) < 0.34):.2f}  95%: {np.mean(np.abs(r - 0.5) < 0.475):.2f}"
              f"  mean rank {r.mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    for name in ('queries', 'loglik', 'combine'):
        p = sub.add_parser(name)
        add_catalog_args(p)
        p.add_argument('--pool_dir', required=True)
        p.add_argument('--splits', nargs='+', default=['train', 'val'])
        p.add_argument('--output_dir', required=True, help='holds sbc_queries.npz, the sbcll files and the output')
        p.add_argument('--seed', type=int, default=0)
        if name == 'queries':
            p.add_argument('--n_queries', type=int, default=300)
            p.add_argument('--query_split', default='val')
            p.add_argument('--mocks_dir', default=None, help='mock files with the hyperparameters (v2: x_hi_7, ...)')
        if name == 'loglik':
            p.add_argument('--ew_model', required=True, choices=list(epl.EW_MODEL_FLAGS))
            p.add_argument('--n_dv', type=int, default=32)
            p.add_argument('--n_y', type=int, default=64)
            p.add_argument('--task_id', type=int, default=0)
            p.add_argument('--n_tasks', type=int, default=1)
        if name == 'combine':
            p.add_argument('--mocks_dir', default=None)
            p.add_argument('--n_post', type=int, default=200, help='SIR samples per query (map statistics)')
            p.add_argument('--pair_sep', type=float, default=8.0)
            p.add_argument('--output', default='exact_sbc.npz')
    args = ap.parse_args()
    {'queries': run_queries, 'loglik': run_loglik, 'combine': run_combine}[args.stage](args)


if __name__ == '__main__':
    main()
