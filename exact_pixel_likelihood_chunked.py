"""
exact_pixel_likelihood_chunked.py -- the exact (quadrature) pixel likelihood of exact_pixel_likelihood.py for
pools too large to hold in memory, e.g. the 1.2M-mock v2 prior (theta alone is 31 GB as float64).

Same likelihood, same numbers: every pool map's per-galaxy log-likelihood l_g(theta_i) is computed by
exact_pixel_likelihood.GalaxyModel.transmission + loglik_quadrature, which are deterministic (no random draws),
so splitting the pool into pieces changes nothing. Only the bookkeeping differs. Two stages:

1. loglik  -- one pool batch file at a time: load its theta, compute l_g for all galaxies, write
              <output_dir>/loglik_<split>_batch_XXXXX.npz (n, n_gal). Parallel over SLURM array tasks: the
              batch files of all --splits, in order, are dealt round-robin (file j -> task j % n_tasks).
              Resumable: an existing output file is skipped. Files are written under a temporary name and
              renamed when complete, so a killed task never leaves a truncated file that looks finished.
2. combine -- one process: read all loglik files (small: 1.2M x 43 floats), form the importance weights of the
              full posterior and of every leave-one-out set, then stream the pool's theta through ONCE,
              accumulating every set's weighted mean map. With --mocks_dir it also collects the mock prior's
              parameters (v2: x_hi_7, z_end, zeta, f_host, tau_e, ...) in pool order, checking row by row that
              theta_i = 1 - masks_i (sightline axis reversed), so their weighted histograms are the
              hyperparameter posterior. --tau_prior applies a Gaussian prior on the Thomson optical depth
              (e.g. a broadened Planck constraint) as an extra importance weight, to all sets.

Pool order = splits in the order given, batch files sorted within a split. The same order is used everywhere, so
row i of `w` / the hyperparameter arrays is map i.

Output of combine: the keys of exact_pixel_likelihood.py's output (p_full, p_minus, influence, ess_full, ess_minus,
set_labels, set_members, x_obs, redshifts, n_gal, n_los, off_support, ew_model), plus
    w        (N,)   normalised posterior weight of every pool map (full data, incl. the tau prior if given)
    L        (N,)   total log-likelihood sum_g l_g (without the tau prior)
    log_prior_tau (N,)  the tau-prior log-weight (zeros without --tau_prior)
    hyper_<name> (N, ...)  the mock-prior parameters, if --mocks_dir is given
    pool_files     the batch files in pool order, pool_counts their sizes

Usage
-----
# stage 1, as a 40-task array (see exact_pixel_likelihood_chunked.sh):
python exact_pixel_likelihood_chunked.py loglik --ew_model exponential \\
    --pool_dir bubble_mocks_v2_run1_x_exponential --output_dir exact_v2_exponential \\
    --task_id $SLURM_ARRAY_TASK_ID --n_tasks 40
# stage 2:
python exact_pixel_likelihood_chunked.py combine --ew_model exponential \\
    --pool_dir bubble_mocks_v2_run1_x_exponential --loglik_dir exact_v2_exponential \\
    --mocks_dir bubble_mocks_v2_run1 --groups 14-18 27-36 21,22,24,25 --output exact_v2_exponential.npz
"""
import os
import glob
import time
import argparse

import numpy as np

import exact_pixel_likelihood as epl

# mock-prior parameters collected by `combine --mocks_dir` when present in the mock files (v1 and v2 names)
HYPER_KEYS = ('x_ion', 'x_hi_7', 'z_end', 'z_mid', 'dz', 'zeta', 'f_host', 'tau_e', 'n_bubbles', 'realized',
              'realized_z', 'hosted', 'k_factor', 'K')


def pool_files(pool_dir, splits):
    """[(split, path), ...] in pool order: splits as given, batch files sorted within each split."""
    out = []
    for split in splits:
        paths = sorted(glob.glob(os.path.join(pool_dir, f'{split}_batch_*.npz')))
        if not paths:
            raise FileNotFoundError(f"no {split}_batch_*.npz in {pool_dir}")
        out += [(split, p) for p in paths]
    return out


def loglik_path(loglik_dir, pool_path):
    return os.path.join(loglik_dir, 'loglik_' + os.path.basename(pool_path))


def add_catalog_args(ap):
    # defaults as in exact_pixel_likelihood.py (= sbi_pixel_field._add_catalog_args)
    ap.add_argument('--lya_catalog', default=os.path.join(epl.REPO, 'tb_lya.txt'))
    ap.add_argument('--properties_catalog', default=os.path.join(epl.REPO, 'sample_nirspec_properties.txt'))
    ap.add_argument('--prefer', default='grating', choices=['grating', 'prism'])
    ap.add_argument('--z_hi', type=float, default=7.3)
    ap.add_argument('--z_min', type=float, default=5.0)
    ap.add_argument('--muv_max', type=float, default=-18.0)


def run_loglik(args):
    if not 0 <= args.task_id < args.n_tasks:
        raise ValueError(f"--task_id {args.task_id} must be in [0, --n_tasks {args.n_tasks})")
    cat = epl.load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                                args.prefer, args.z_hi)
    ew_obs, is_ul, sigma = cat['ew_obs'], cat['is_ul'], cat['sigma']
    n_gal = len(ew_obs)
    model = epl.GalaxyModel(cat, args.ew_model, 1, 0)       # n_mc, seed: unused by the quadrature
    dists = [model.ew_distribution(g) for g in range(n_gal)]
    os.makedirs(args.output_dir, exist_ok=True)

    files = pool_files(args.pool_dir, args.splits)
    mine = [f for j, f in enumerate(files) if j % args.n_tasks == args.task_id]
    print(f"[loglik] task {args.task_id}/{args.n_tasks}: {len(mine)} of {len(files)} pool files, "
          f"ew_model={args.ew_model}, n_dv={args.n_dv}, n_y={args.n_y}", flush=True)
    for split, path in mine:
        out = loglik_path(args.output_dir, path)
        if os.path.exists(out):
            print(f"[loglik] {out} exists, skipping (resumable).", flush=True)
            continue
        t0 = time.perf_counter()
        d = np.load(path)
        n_los = int(d['n_los'])
        if int(d['n_gal']) != n_gal:
            raise ValueError(f"{path} has {int(d['n_gal'])} galaxies, catalog {n_gal}")
        theta = d['theta']
        theta_grid = theta.reshape(len(theta), n_gal, n_los).astype(np.float64)
        ll = np.empty((len(theta), n_gal))
        for g in range(n_gal):
            T = model.transmission(g, theta_grid[:, g, :], n_los, args.n_dv)
            ll[:, g] = epl.loglik_quadrature(T, ew_obs[g], is_ul[g], sigma[g], dists[g], args.n_y)
        tmp = out[:-len('.npz')] + '.tmp.npz'
        np.savez(tmp, loglik=ll, n_gal=n_gal, n_los=n_los, ew_model=args.ew_model, n_dv=args.n_dv,
                 n_y=args.n_y, pool_file=os.path.basename(path))
        os.replace(tmp, out)                                 # appears under its final name only when complete
        print(f"[loglik] {os.path.basename(path)}: {len(theta)} maps in {time.perf_counter() - t0:.0f} s "
              f"-> {out}", flush=True)
    print(f"[loglik] task {args.task_id} done", flush=True)


def run_combine(args):
    cat = epl.load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                                args.prefer, args.z_hi)
    ew_obs, is_ul, sigma = cat['ew_obs'], cat['is_ul'], cat['sigma']
    n_gal = len(ew_obs)
    off_support = (~is_ul) & (ew_obs <= 3 * sigma)

    # ---- per-galaxy log-likelihoods of the whole pool, in pool order ----
    files = pool_files(args.pool_dir, args.splits)
    missing = [p for _, p in files if not os.path.exists(loglik_path(args.loglik_dir, p))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(files)} loglik files missing in {args.loglik_dir} "
                                f"(first: {os.path.basename(missing[0])}) -- is stage 1 still running?")
    lls = [np.load(loglik_path(args.loglik_dir, p)) for _, p in files]
    for ld in lls:
        if str(ld['ew_model']) != args.ew_model:
            raise ValueError(f"loglik file {ld['pool_file']} was computed for ew_model={ld['ew_model']}, "
                             f"not {args.ew_model}")
    counts = np.array([len(ld['loglik']) for ld in lls])
    loglik = np.concatenate([ld['loglik'] for ld in lls])    # (N, n_gal)
    n_los = int(lls[0]['n_los'])
    N = len(loglik)
    L = loglik.sum(axis=1)
    print(f"[combine] {N} pool maps from {len(files)} files, {n_gal} galaxies, n_los = {n_los}", flush=True)

    # ---- mock-prior parameters (small: read first; the masks are row-checked in the theta pass) ----
    hyper, mock_paths = {}, None
    if args.mocks_dir:
        mock_paths = [os.path.join(args.mocks_dir, os.path.basename(p)) for _, p in files]
        absent = [m for m in mock_paths if not os.path.exists(m)]
        if absent:
            raise FileNotFoundError(f"{len(absent)} mock files missing, e.g. {absent[0]}")
        per_file = {k: [] for k in HYPER_KEYS}
        for mp, n in zip(mock_paths, counts):
            m = np.load(mp)
            if len(m['x_ion']) != n:
                raise ValueError(f"{mp}: {len(m['x_ion'])} mocks, but its pool file has {n} maps")
            for k in HYPER_KEYS:
                if k in m.files:
                    per_file[k].append(m[k])
        hyper = {k: np.concatenate(v) for k, v in per_file.items() if len(v) == len(mock_paths)}
        print(f"[combine] mock parameters: {', '.join(hyper)}", flush=True)
    if args.tau_prior:
        if 'tau_e' not in hyper:
            raise ValueError("--tau_prior needs --mocks_dir with tau_e in the mock files (v2 prior)")
        mean, sd = args.tau_prior
        log_prior_tau = -0.5 * ((hyper['tau_e'] - mean) / sd) ** 2
        wt = np.exp(log_prior_tau - log_prior_tau.max())
        print(f"[combine] tau prior N({mean}, {sd}): ESS of this reweighting alone {wt.sum() ** 2 / np.sum(wt ** 2):.0f} "
              f"of {N}", flush=True)
    else:
        log_prior_tau = np.zeros(N)

    # ---- importance weights: full posterior (row 0) and every leave-one-out set ----
    sets = [[g] for g in range(n_gal)] + epl.parse_groups(args.groups, n_gal)
    labels = [str(s[0]) if len(s) == 1 else ','.join(map(str, s)) for s in sets]

    logw = np.vstack([L + log_prior_tau] + [L - loglik[:, s].sum(axis=1) + log_prior_tau for s in sets])
    logw -= logw.max(axis=1, keepdims=True)
    W = np.exp(logw)
    W /= W.sum(axis=1, keepdims=True)                        # (1 + n_sets, N), rows sum to 1
    ess = 1.0 / np.sum(W ** 2, axis=1)
    print(f"[combine] exact-likelihood posterior: ESS {ess[0]:.1f} of {N}", flush=True)

    # ---- one streaming pass over theta: every set's weighted mean map ----
    maps = np.zeros((len(W), n_gal * n_los))
    a = 0
    t0 = time.perf_counter()
    for i, ((split, p), n) in enumerate(zip(files, counts)):
        theta = np.load(p)['theta']
        if len(theta) != n:
            raise ValueError(f"{p}: {len(theta)} maps, but its loglik file has {n}")
        maps += W[:, a:a + n] @ theta.astype(np.float64)
        if mock_paths:                                       # row check theta_i <-> masks_i
            masks = np.load(mock_paths[i])['masks']
            if not np.array_equal(theta.reshape(n, n_gal, n_los), 1 - masks[:, :, ::-1].astype(theta.dtype)):
                raise ValueError(f"{p} and {mock_paths[i]}: theta != 1 - masks reversed -- rows do not match")
        a += n
        if (i + 1) % 50 == 0:
            print(f"[combine] {i + 1}/{len(files)} files streamed ({time.perf_counter() - t0:.0f} s)", flush=True)
    maps = maps.reshape(len(W), n_gal, n_los)
    p_full, p_minus = maps[0], maps[1:]

    out = dict(p_full=p_full, p_minus=p_minus, influence=p_full[None] - p_minus, ess_full=ess[0],
               ess_minus=ess[1:], set_labels=np.array(labels),
               set_members=np.array([np.pad(s, (0, n_gal - len(s)), constant_values=-1) for s in sets]),
               x_obs=np.concatenate([ew_obs, is_ul.astype(float)]).astype(np.float32),
               redshifts=cat['z_gal_red'], n_gal=n_gal, n_los=n_los, off_support=off_support,
               ew_model=args.ew_model, w=W[0], L=L, log_prior_tau=log_prior_tau,
               tau_prior=np.array(args.tau_prior if args.tau_prior else [np.nan, np.nan]),
               pool_files=np.array([os.path.basename(p) for _, p in files]), pool_counts=counts)
    out.update({f'hyper_{k}': v for k, v in hyper.items()})
    np.savez(args.output, **out)
    print(f"[combine] saved {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('loglik', 'combine'):
        p = sub.add_parser(name)
        p.add_argument('--ew_model', required=True, choices=list(epl.EW_MODEL_FLAGS))
        p.add_argument('--pool_dir', required=True, help='(theta, x) batch files (add_mock_observations.py output)')
        p.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='which splits form the pool, in this order (default: train val)')
        add_catalog_args(p)
    pl = sub.choices['loglik']
    pl.add_argument('--output_dir', required=True, help='where the per-batch loglik files go')
    pl.add_argument('--n_dv', type=int, default=32, help='velocity-offset quantiles (as exact_pixel_likelihood)')
    pl.add_argument('--n_y', type=int, default=64, help='cells of the EW integral (as exact_pixel_likelihood)')
    pl.add_argument('--task_id', type=int, default=0)
    pl.add_argument('--n_tasks', type=int, default=1)
    pc = sub.choices['combine']
    pc.add_argument('--loglik_dir', required=True, help='--output_dir of the loglik stage')
    pc.add_argument('--mocks_dir', default=None,
                    help='generate_bubble_mocks.py output (masks + prior parameters) of the same pool')
    pc.add_argument('--tau_prior', nargs=2, type=float, default=None, metavar=('MEAN', 'SD'),
                    help='Gaussian prior on tau_e as an importance weight, e.g. 0.054 0.014 (Planck 2018, '
                         'sd doubled). Needs --mocks_dir with tau_e.')
    pc.add_argument('--groups', nargs='*', default=[], help="extra leave-out sets, e.g. 14-18 21,22,24")
    pc.add_argument('--output', required=True)
    args = ap.parse_args()
    for k in ('pool_dir', 'output_dir', 'loglik_dir', 'mocks_dir', 'output'):
        if getattr(args, k, None):
            setattr(args, k, os.path.expanduser(getattr(args, k)))
    run_loglik(args) if args.cmd == 'loglik' else run_combine(args)


if __name__ == '__main__':
    main()
