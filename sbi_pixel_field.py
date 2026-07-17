"""
Pixelated field-level SBI mode: infers the ionization state of the volume
itself as a per-galaxy binary line-of-sight mask, via Neural Ratio Estimation
(NRE), instead of a small number of bubble parameters (position, radius) --
an additional, optional, exploratory inference path alongside BOTH the
dynesty pipeline (`real_data_run.py`) and the bubble-parametric SBI pipeline
(`sbi_real_data.py`). Neither is modified by this file. See
`.claude/plans/bright-growing-goblet.md` for the full design writeup; this
docstring covers the essentials.

Why this exists
----------------
The bubble-parametric model assumes the ionized structure is a small number
of spheres. This mode instead treats the ionization state itself as the
inference target: a grid of **43 on-sky columns** (one per real galaxy --
with only 43 galaxies, a regular angular grid would have cells with no
constraining data at all) **x N_LOS bins along each galaxy's own line of
sight** (its personal path from its own redshift down to `z_end=5.3`),
each pixel a strict binary ionized/neutral latent (not a float in [0, 1] --
a pixel is either ionized or it isn't, by design). Total theta dimensionality
is `n_gal * n_los` (~2,150-4,300 for the discussed N_LOS=50-100 range).

theta layout: `theta[g, k]` = bin `k` of galaxy `g`'s own line of sight,
`g` in the SAME order as `_S.x_gal`/etc, `k=0` nearest the source, `k=N_LOS-1`
nearest `z_end` -- stored flattened as `(n_gal * n_los,)` in the `.npz`
batch files (same convention `sbi_real_data.py` uses), reshape via
`theta.reshape(n_gal, n_los)`.

x layout is IDENTICAL to `sbi_real_data.py` -- this file imports `_pack_x`/
`_build_x`/`_build_x_obs`/`_per_galaxy_sigma` directly rather than
duplicating them, since keeping simulated and real x construction from ever
drifting apart is a hard invariant of this whole SBI approach, independent
of which theta parametrization is used.

Physics: reuses `lyabubbles/lightcone_field.py`'s multi-box lightcone
stacking (`ray_trace_segments_stacked`/`draw_stack_augmentations`,
`discretize_to_fixed_bins`, `bin_z_edges`) -- a real, evolving reionization
lightcone built by walking through several 21cmFAST snapshot boxes as the
sightline approaches the observer, rather than the bubble pipeline's
single-box-plus-analytic-tail. Augmentations are drawn ONCE per simulated
draw and shared across every galaxy in that draw (see
`draw_stack_augmentations`'s docstring) -- this is what gives galaxies
correlated background structure; drawing a fresh augmentation per galaxy
would silently reintroduce per-galaxy independence.

Inference is POOL-BASED, not direct posterior sampling: a trained ratio
`r(theta, x) ~ p(x|theta)/p(x)` is evaluated against every `(theta_i, x_i)`
pair in a held-out simulation pool at the real `x_obs`, giving self-
normalized importance weights. This gives BOTH deliverables requested: a
per-pixel marginal probability map (weighted average of `theta_i` per
pixel) and coherent joint generative map samples (sampling-importance-
resampling of the pool by those weights -- each resampled map is a real,
physically-generated field, so its internal correlation structure is
automatic, unlike independent per-pixel draws would be).

Requires `torch`/`sbi` (see requirements-sbi.txt) ONLY for `train_nre`/
`infer` -- `simulate` needs neither, same as `sbi_real_data.py`.

NOT in scope here (see plan doc): a calibration/SBC analog for this pixel
model (the existing `sbi_calibrate.py` assumes low-dimensional continuous
theta); any change to `real_data_run.py`/`sbi_real_data.py`/`sbi_calibrate.py`.

Usage
-----
python sbi_pixel_field.py simulate  --n_los 75 --n_sim 15000 --output_dir sbi_runs/pixel/sims
python sbi_pixel_field.py train_nre --n_los 75 --sims_dir sbi_runs/pixel/sims --output_dir sbi_runs/pixel
python sbi_pixel_field.py infer     --n_los 75 --ratio sbi_runs/pixel/ratio.pt \\
    --pool_dir sbi_runs/pixel/sims --output_dir sbi_runs/pixel
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import glob
import time
import argparse
import multiprocessing as mp
import numpy as np
from astropy.cosmology import Planck18 as Cosmo
import astropy.units as u

import real_data_run as rdr
from lyabubbles.igm_prop import get_xH
from lyabubbles import lightcone_field as lf
from sbi_real_data import _pack_x, _build_x, _build_x_obs, _per_galaxy_sigma


# ── Per-galaxy fixed bin geometry (catalog-fixed, NOT theta/MC-draw-dependent
#    -- computed once right after `_load_catalog_and_priors`, reused for
#    every simulated draw and at inference time) ────────────────────────────

def _precompute_bin_z_edges(meta, n_los):
    """`(n_gal, n_los)` z_b/z_e arrays, one row per galaxy, via
    `lf.bin_z_edges` -- each galaxy's own path from its own `z_gal` (fixed by
    the catalog) down to `z_end`, so this only needs computing once per
    `simulate`/`train_nre`/`infer` run, not per simulated draw."""
    s = rdr._S
    n_gal = len(s.x_gal)
    d_c0 = Cosmo.comoving_distance(meta['z0']).to(u.Mpc).value
    z_b_edges = np.empty((n_gal, n_los))
    z_e_edges = np.empty((n_gal, n_los))
    for g in range(n_gal):
        z_b_edges[g], z_e_edges[g] = lf.bin_z_edges(
            n_los, s.z_gal[g], d_c0, z_end=lf.Z_END_DEFAULT)
    return z_b_edges, z_e_edges


# ── Bulk theta (binary LOS masks) + tau generation, shared-lightcone ───────

def _compute_pixel_theta_and_tau_batch(this_batch, z0, main_dir, rng, n_los,
                                       z_b_edges, z_e_edges,
                                       n_done_before=0, n_sim_total=None, prefix=''):
    """For each simulation k in this batch: draw `x_h_target` ONCE (shared by
    every galaxy, same convention as `sbi_real_data.py`), pick the anchor
    snapshot, draw ONE `aug_by_box` (shared by every galaxy -- see
    `lf.draw_stack_augmentations`'s docstring for why this must not be
    redrawn per galaxy), then for each galaxy independently: stacked ray
    trace from its own `z_gal` -> binarize onto its own fixed `n_los` bins
    (theta) -> tau via `segments_to_tau` on the SAME fixed bin edges.

    Returns `(theta_out, tau_out)`: `theta_out` shape `(n_gal, batch_size,
    n_los)` binary, `tau_out` shape `(n_gal, batch_size, n_wave)`.
    """
    s = rdr._S
    n_gal = len(s.x_gal)
    wave_em_vals = rdr.wave_em.value
    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value
    n_sim_total = n_sim_total if n_sim_total is not None else this_batch
    print_every = max(1, this_batch // 20)
    t_start = time.perf_counter()

    theta_out = np.empty((n_gal, this_batch, n_los), dtype=np.float32)
    tau_out = np.empty((n_gal, this_batch, len(wave_em_vals)), dtype=np.float64)

    for k in range(this_batch):
        x_h_target = float(get_xH(z0, main_dir=main_dir))
        anchor_idx = lf.select_snapshot_index(x_h_target)
        aug_by_box = lf.draw_stack_augmentations(anchor_idx, rng)

        for g in range(n_gal):
            z_b, z_e, x_hi = lf.ray_trace_segments_stacked(
                anchor_idx, aug_by_box, s.x_gal[g], s.y_gal[g], s.z_gal[g], d_c0,
                z_end=lf.Z_END_DEFAULT,
            )
            theta_col = lf.discretize_to_fixed_bins(
                z_b, z_e, x_hi, n_los, s.z_gal[g], d_c0,
                z_end=lf.Z_END_DEFAULT, threshold=0.5,
            )
            theta_out[g, k, :] = theta_col
            tau_out[g, k, :] = lf.segments_to_tau(
                z_b_edges[g], z_e_edges[g], theta_col, s.redshifts[g], wave_em_vals,
            )

        if (k + 1) % print_every == 0 or k == this_batch - 1:
            n_overall = n_done_before + k + 1
            elapsed = time.perf_counter() - t_start
            rate = (k + 1) / elapsed if elapsed > 0 else 0.0
            eta_s = (this_batch - (k + 1)) / rate if rate > 0 else float('nan')
            print(f"[simulate:{prefix}] {n_overall}/{n_sim_total} sims done "
                  f"({100.0 * n_overall / n_sim_total:.1f}%) -- "
                  f"{k + 1}/{this_batch} in this batch, "
                  f"{rate:.2f} sims/s, ETA this batch {eta_s:.0f}s", flush=True)

    return theta_out, tau_out


def _ew_pred_from_pixel_tau(k_idx):
    """Same trapz/exp(-tau) combination `_ew_and_t_for_inside`/
    `_ew_pred_from_lightcone_tau` (real_data_run.py/sbi_real_data.py) already
    use, reading `_S.pixel_tau` instead."""
    s = rdr._S
    tau_now  = s.pixel_tau[:, k_idx, :]
    j_s_k    = s.j_s[:, k_idx, :]
    weighted = j_s_k * s.tau_cgm * np.exp(-tau_now)
    numerator = np.trapz(weighted, rdr.wave_em.value, axis=1)
    t_in = numerator / s.j_s_trapz_denom[:, k_idx]
    return s.ew_int[:, k_idx] * t_in


def simulate_one(k_idx, rng):
    ew_pred = _ew_pred_from_pixel_tau(k_idx)
    return _build_x(ew_pred, _per_galaxy_sigma(), rng)


def _simulate_worker(task):
    """Runs in a forked worker -- `_S` fully populated pre-fork, same
    pattern as `sbi_real_data.py::_simulate_worker`."""
    i, k_idx, noise_seed = task
    rng = np.random.default_rng(noise_seed)
    return i, simulate_one(k_idx, rng)


def _generate_split(meta, n_sim, batch_size, n_workers, main_dir, n_los,
                    z_b_edges, z_e_edges, seed, output_dir, prefix):
    """Bulk-generate `n_sim` (theta, x) pairs in resumable batches of
    `batch_size`, saved as `{prefix}_batch_{i:05d}.npz` -- mirrors
    `sbi_real_data.py::_generate_split`'s resumable-batching/progress-
    reporting/fork-pool structure exactly, with the theta-generation step
    (`_compute_pixel_theta_and_tau_batch`) swapped in."""
    rng_master = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)
    n_done, batch_idx = 0, 0
    while n_done < n_sim:
        this_batch = min(batch_size, n_sim - n_done)
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")
        if os.path.exists(out_path):
            existing_n = len(np.load(out_path)['theta'])
            if existing_n != this_batch:
                print(f"[simulate:{prefix}] WARNING: {out_path} exists with {existing_n} sims, "
                      f"not the {this_batch} this run expects at batch {batch_idx} -- looks like "
                      f"it's from a run with different --n_sim/--batch_size in this same "
                      f"--output_dir. Using the {existing_n} sims already on disk (will generate "
                      f"more batches to top up to --n_sim); delete this file first if you want it "
                      f"regenerated with the new parameters instead.", flush=True)
            print(f"[simulate:{prefix}] {out_path} exists ({existing_n} sims), skipping (resumable).",
                  flush=True)
            n_done += existing_n
            batch_idx += 1
            continue

        rdr._refresh_mc_state(
            meta['muv'], meta['redshifts'], meta['x_gal'], meta['y_gal'], meta['z_gal'],
            meta['beta'], meta['z0'], this_batch, main_dir,
        )

        theta_batch, tau_batch = _compute_pixel_theta_and_tau_batch(
            this_batch, meta['z0'], main_dir, rng_master, n_los, z_b_edges, z_e_edges,
            n_done_before=n_done, n_sim_total=n_sim, prefix=prefix,
        )
        rdr._S.pixel_tau = tau_batch

        noise_seeds = rng_master.integers(0, 2**31 - 1, size=this_batch)
        tasks = [(i, i, int(noise_seeds[i])) for i in range(this_batch)]

        with mp.get_context('fork').Pool(n_workers) as pool:
            results = pool.map(_simulate_worker, tasks)

        x_dim = results[0][1].shape[0]
        n_gal = theta_batch.shape[0]
        x_out = np.empty((this_batch, x_dim), dtype=np.float32)
        for i, x in results:
            x_out[i] = x
        theta_out = theta_batch.transpose(1, 0, 2).reshape(this_batch, n_gal * n_los)

        np.savez(out_path, theta=theta_out, x=x_out, n_gal=n_gal, n_los=n_los)
        print(f"[simulate:{prefix}] batch {batch_idx}: {this_batch} sims -> {out_path}", flush=True)
        n_done += this_batch
        batch_idx += 1


def load_sims(sims_dir, prefix):
    """Load and concatenate all `{prefix}_batch_*.npz` files -- theta stays
    flattened `(n_sim, n_gal * n_los)`; reshape to `(n_sim, n_gal, n_los)`
    with the `n_gal`/`n_los` also stored in each file if needed."""
    paths = sorted(glob.glob(os.path.join(sims_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_batch_*.npz files found in {sims_dir}")
    thetas, xs = [], []
    n_gal = n_los = None
    for p in paths:
        d = np.load(p)
        thetas.append(d['theta'])
        xs.append(d['x'])
        n_gal, n_los = int(d['n_gal']), int(d['n_los'])
    return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0), n_gal, n_los


def run_simulate(args):
    meta = rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    z_b_edges, z_e_edges = _precompute_bin_z_edges(meta, args.n_los)

    n_sim_val = args.n_sim_val if args.n_sim_val is not None else max(1, args.n_sim // 5)
    _generate_split(meta, args.n_sim, args.batch_size, args.n_workers, args.main_dir,
                    args.n_los, z_b_edges, z_e_edges, args.seed, args.output_dir, prefix='train')
    _generate_split(meta, n_sim_val, args.batch_size, args.n_workers, args.main_dir,
                    args.n_los, z_b_edges, z_e_edges, args.seed + 1_000_000, args.output_dir,
                    prefix='val')
    print(f"[simulate] done: {args.n_sim} train + {n_sim_val} val sims "
          f"({len(meta['x_gal'])} galaxies x {args.n_los} LOS bins each) in {args.output_dir}",
          flush=True)


# ── NRE training ─────────────────────────────────────────────────────────────
#
# NOTE: exact `sbi` API for `NRE`/`SNRE_B` (constructor args, whether a
# custom embedding_net for theta is required for a `(n_gal, n_los)`-shaped
# input to train well) is UNVERIFIED -- `sbi`/`torch` aren't available in any
# local environment (same situation as `sbi_real_data.py`'s NPE usage).
# First real test happens on the cluster; if `sbi`'s built-in class doesn't
# fit this theta shape/size well, fall back to a hand-written binary
# classifier (real (theta,x) pairs vs. shuffled-marginal negatives) trained
# with plain PyTorch.

def run_train_nre(args):
    import torch
    from sbi.inference import SNRE_B

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but torch.cuda.is_available() "
                           f"is False on this machine/allocation.")
    print(f"[train_nre] device: {args.device}", flush=True)

    theta_train, x_train, n_gal, n_los = load_sims(args.sims_dir, 'train')
    if n_los != args.n_los:
        raise ValueError(f"--n_los {args.n_los} doesn't match the {n_los} used to "
                         f"generate {args.sims_dir}'s sims.")
    print(f"[train_nre] loaded {len(theta_train)} training sims "
          f"({n_gal} galaxies x {n_los} LOS bins) from {args.sims_dir}", flush=True)

    # theta is genuinely binary -- BoxUniform(0, 1) is a placeholder prior
    # object satisfying sbi's constructor requirement (SNRE trains a
    # classifier on theta/x pairs vs. shuffled negatives; unlike NPE it
    # doesn't need to SAMPLE from this prior for anything downstream except
    # possibly a default `.sample()`-based posterior, which this file doesn't
    # use -- inference is pool-based instead, see `run_infer`). Revisit if
    # sbi's SNRE_B constructor validates against this more strictly than
    # expected once actually run.
    from sbi.utils import BoxUniform
    prior = BoxUniform(low=torch.zeros(theta_train.shape[1]),
                       high=torch.ones(theta_train.shape[1]))

    inference = SNRE_B(prior=prior, device=args.device)
    inference.append_simulations(
        torch.as_tensor(theta_train, dtype=torch.float32, device=args.device),
        torch.as_tensor(x_train, dtype=torch.float32, device=args.device),
    )
    ratio_estimator = inference.train()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'ratio_estimator.pt')
    torch.save({
        'ratio_estimator': ratio_estimator,
        'n_gal': n_gal, 'n_los': n_los, 'n_sim': len(theta_train),
    }, out_path)
    print(f"[train_nre] saved trained ratio estimator to {out_path}", flush=True)


# ── Pool-based inference: marginal map + joint generative samples ──────────

def run_infer(args):
    import torch

    checkpoint = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator = checkpoint['ratio_estimator']
    n_gal, n_los = checkpoint['n_gal'], checkpoint['n_los']

    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    if len(rdr._S.x_gal) != n_gal:
        raise ValueError(f"Catalog has {len(rdr._S.x_gal)} galaxies but the ratio estimator "
                         f"was trained on {n_gal} -- make sure --lya_catalog/--properties_catalog/"
                         f"--z_lo/--z_hi/etc match the `simulate` run that produced --pool_dir.")

    theta_pool, x_pool, pool_n_gal, pool_n_los = load_sims(args.pool_dir, args.pool_split)
    if pool_n_gal != n_gal or pool_n_los != n_los:
        raise ValueError(f"--pool_dir sims ({pool_n_gal} gal x {pool_n_los} los) don't match "
                         f"the ratio estimator ({n_gal} gal x {n_los} los).")

    x_obs = _build_x_obs()
    x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32, device=args.device)
    theta_pool_t = torch.as_tensor(theta_pool, dtype=torch.float32, device=args.device)
    x_obs_tiled = x_obs_t.unsqueeze(0).expand(len(theta_pool), -1)

    with torch.no_grad():
        log_ratio = ratio_estimator(theta=theta_pool_t, x=x_obs_tiled).squeeze(-1)
    log_ratio = log_ratio.detach().cpu().numpy()

    # Self-normalized importance weights -- mirrors
    # `real_data_run.py::fesc_effective_sample_size`'s pattern for the same
    # underlying concern (pool-coverage/degeneracy visible via ESS, not
    # silent). log-sum-exp for numerical stability.
    log_w = log_ratio - log_ratio.max()
    w = np.exp(log_w)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    print(f"[infer] pool size {len(theta_pool)}, effective sample size {ess:.1f} "
          f"({100 * ess / len(theta_pool):.2f}% of pool) -- low ESS means the pool "
          f"has poor coverage of the true posterior region; results below are then "
          f"dominated by very few draws and should not be trusted.", flush=True)

    theta_pool_grid = theta_pool.reshape(len(theta_pool), n_gal, n_los)
    marginal_map = np.tensordot(w, theta_pool_grid, axes=(0, 0))   # (n_gal, n_los)

    rng = np.random.default_rng(args.seed)
    resample_idx = rng.choice(len(theta_pool), size=args.n_samples, replace=True, p=w)
    joint_samples = theta_pool_grid[resample_idx]   # (n_samples, n_gal, n_los)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'pixel_infer.npz')
    np.savez(
        out_path,
        marginal_map=marginal_map, joint_samples=joint_samples,
        log_ratio=log_ratio, weights=w, ess=ess, pool_size=len(theta_pool),
        n_gal=n_gal, n_los=n_los,
    )
    print(f"[infer] saved {out_path} (marginal_map {marginal_map.shape}, "
          f"{args.n_samples} joint_samples {joint_samples.shape})", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _add_catalog_args(p):
    p.add_argument('--lya_catalog', type=str, default='tb_lya.txt')
    p.add_argument('--properties_catalog', type=str, default='sample_nirspec_properties.txt')
    p.add_argument('--prefer', type=str, default='grating', choices=['grating', 'prism'])
    p.add_argument('--legacy_catalog', type=str, default=None)
    p.add_argument('--z_lo', type=float, default=None)
    p.add_argument('--z_hi', type=float, default=7.3)
    p.add_argument('--z_min', type=float, default=5.0)
    p.add_argument('--muv_max', type=float, default=-18.0)
    p.add_argument('--r_max', type=float, default=None)
    p.add_argument('--main_dir', type=str,
                   default='/groups/astro/ivannik/programs/Lyman-alpha-bubbles/')
    p.add_argument('--n_los', type=int, default=75,
                   help='LOS bins per galaxy (discussed range 50-100).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_sim = sub.add_parser('simulate', help='Bulk-generate (theta, x) pairs -- theta = binary '
                                            'per-galaxy LOS masks.')
    _add_catalog_args(p_sim)
    p_sim.add_argument('--n_sim', type=int, default=15000, help='Training simulations.')
    p_sim.add_argument('--n_sim_val', type=int, default=None,
                       help='Held-out pool simulations (default: 20%% of --n_sim) -- this pool '
                            'doubles as the `infer` candidate pool, so size it generously.')
    p_sim.add_argument('--batch_size', type=int, default=2000)
    p_sim.add_argument('--n_workers', type=int, default=8)
    p_sim.add_argument('--seed', type=int, default=0)
    p_sim.add_argument('--output_dir', type=str, required=True)

    p_train = sub.add_parser('train_nre', help='Train a Neural Ratio Estimator on generated sims.')
    _add_catalog_args(p_train)
    p_train.add_argument('--sims_dir', type=str, required=True,
                         help='--output_dir from a prior `simulate` run.')
    p_train.add_argument('--device', type=str, default='cpu')
    p_train.add_argument('--output_dir', type=str, required=True)

    p_infer = sub.add_parser('infer', help='Pool-based inference at the real x_obs: marginal '
                                           'per-pixel probability map + joint generative samples.')
    _add_catalog_args(p_infer)
    p_infer.add_argument('--ratio', type=str, required=True, help='ratio_estimator.pt from `train_nre`.')
    p_infer.add_argument('--pool_dir', type=str, required=True,
                         help='--output_dir from `simulate` -- the candidate pool to reweight.')
    p_infer.add_argument('--pool_split', type=str, default='val', choices=['train', 'val'],
                         help='Which split to use as the reweighting pool (default val, since '
                              'it is held out and can be sized independently).')
    p_infer.add_argument('--n_samples', type=int, default=1000,
                         help='Number of joint map samples to draw via SIR.')
    p_infer.add_argument('--device', type=str, default='cpu')
    p_infer.add_argument('--seed', type=int, default=0)
    p_infer.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    {'simulate': run_simulate, 'train_nre': run_train_nre, 'infer': run_infer}[args.command](args)
