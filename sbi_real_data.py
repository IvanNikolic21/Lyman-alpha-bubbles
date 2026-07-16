"""
Simulation-Based Inference (SBI) mode for the Lyman-alpha bubble real-data
catalog -- an additional, optional inference path alongside the default
dynesty pipeline (`real_data_run.py`, unmodified by this file). See
`.claude/plans/bright-growing-goblet.md` (or ask the assistant) for the full
design writeup; this docstring covers the essentials.

Why this exists
----------------
The dynesty pipeline's EW likelihood (`real_data_run.py`'s
`_ew_loglike_from_pred`) is a hand-built Student-t detection term + censored
upper-limit term, and factorizes as a product over galaxies given theta. SBI
sidesteps both: it needs no closed-form likelihood (the simulator only has to
be *sampleable*), and its observation vector `x` is the FULL joint vector of
simulated per-galaxy outcomes for the fixed real catalog -- training a neural
density estimator on many `(theta, x)` pairs lets it learn whatever joint
correlation structure exists (e.g. several nearby LAEs explained by one
bubble) automatically, with no explicit correlation term.

`x` layout: `concat([ew_channel, is_ul_channel])`, `2*n_gal` dims, built by
`_build_x`/`_build_x_obs` -> `_pack_x` (the ONE place x's meaning is defined,
so simulated and real x can never structurally drift apart). Detection
status is stochastic and theta-dependent (see `_build_x`): a galaxy that's a
real non-detection CAN appear as a simulated detection if theta boosts its
predicted EW past its own catalog-implied 3-sigma threshold, and vice versa.

IMPORTANT caveat on the correlation claim above, found by an independent
review (Fable) and fixed via `lyabubbles/lightcone_field.py`: the original
version of this simulator drew each galaxy's "beyond the fitted bubble" IGM
structure independently (via `get_content`'s per-galaxy `get_xH`/`get_bubbles`
calls), so the true generative model factorized exactly like the old dynesty
likelihood -- no galaxy-galaxy correlation beyond what's mediated by the
shared fitted bubble. `_compute_lightcone_tau_batch` (below) fixes this for
the *near-source* part of each sightline by ray-tracing every galaxy through
one shared, real 21cmFAST snapshot per simulation (with the fitted bubble(s)
carved in) instead -- bounded to one box-length (384 cMpc) near the source,
by deliberate choice; the analytic tail beyond that (folded into the same ray
trace via `x_h_tail`) is still a uniform approximation, not sourced from the
shared field. See the plan doc for the full reasoning and the tradeoffs of
that scope choice.

Reuses `real_data_run.py`'s physics/state machinery directly: `_S`
(module-level shared state, fork-inherited by worker processes),
`_load_catalog_and_priors`/`_refresh_mc_state` (the `build_state` split --
catalog loaded once, MC draws refreshed per simulation batch; still used
here for `j_s`/`ew_int`/`tau_cgm`, NOT for the old `tau_prec`/`z_up`/etc.
"outside bubble" fields, which this file no longer reads),
`_inside_and_z_end` (theta-dependent geometry -- who's inside which bubble,
used by `_compute_lightcone_tau_batch` to carve the bubble(s) into the
lightcone), `_prior_transform`/`_prior_transform_2bub`/`_prior_transform_3bub`
(the simulation-informed bubble-size prior, unchanged by this file).
`real_data_run.py`/`speed_up.py`/`get_content` themselves are NOT modified.

Requires `torch`/`sbi` (see requirements-sbi.txt) ONLY for the `train`/`infer`
subcommands -- `simulate` needs neither, just the existing physics stack
(plus `h5py` for the 21cmFAST snapshot files, already a base dependency).

Usage
-----
python sbi_real_data.py simulate --n_bub 1 --n_sim 15000 --output_dir sbi_runs/m1/sims
python sbi_real_data.py train    --n_bub 1 --sims_dir sbi_runs/m1/sims --output_dir sbi_runs/m1
python sbi_real_data.py infer    --n_bub 1 --posterior sbi_runs/m1/posterior.pt \\
    --output_dir sbi_runs/m1
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import glob
import argparse
import multiprocessing as mp
import numpy as np
from scipy.stats import exponnorm
from astropy.cosmology import Planck18 as Cosmo
import astropy.units as u

import real_data_run as rdr
from lyabubbles.igm_prop import get_xH
from lyabubbles import lightcone_field as lf

_N_BUB_TO_NDIM = {1: rdr.NDIM, 2: rdr.NDIM_2BUB, 3: rdr.NDIM_3BUB}
_N_BUB_TO_PRIOR_TRANSFORM = {1: rdr._prior_transform, 2: rdr._prior_transform_2bub,
                             3: rdr._prior_transform_3bub}
_N_BUB_TO_PARAM_NAMES = {1: rdr.PARAM_NAMES, 2: rdr.PARAM_NAMES_2BUB, 3: rdr.PARAM_NAMES_3BUB}


# ── The one place x's meaning is defined ────────────────────────────────────

def _pack_x(ew_channel, is_ul_channel):
    """x's layout: concat(ew_channel, is_ul_channel), both length n_gal, in
    the SAME galaxy order as `_S.x_gal`/`_S.ew_obs`/etc. `_build_x`
    (simulated) and `_build_x_obs` (real data) both funnel through this so
    the two can never structurally drift apart."""
    return np.concatenate([np.asarray(ew_channel, dtype=np.float32),
                           np.asarray(is_ul_channel, dtype=np.float32)])


def _per_galaxy_sigma():
    """Per-galaxy noise scale implied by the real catalog: the reported
    1-sigma error for real detections (`_S.ew_err`), or (reported 3-sigma
    upper-limit value)/3 for real upper limits (`_S.ew_obs` holds the raw
    threshold value for those rows -- see `lyabubbles/real_data.py`'s
    `load_catalog_v2`). This is a fixed per-galaxy detector characteristic,
    NOT theta-dependent -- only the detection *decision* built from it is
    (see `_build_x`)."""
    s = rdr._S
    return np.where(s.is_upper_limit, s.ew_obs / 3.0, s.ew_err)


def _build_x(ew_pred, sigma, rng):
    """One simulated observation: sample a noisy EW around the model's
    noiseless prediction, then decide detection status via a 3-sigma cut
    against THIS galaxy's own catalog-implied noise level. Theta enters only
    through `ew_pred` -- a bubble that raises a galaxy's predicted EW past
    `3*sigma` can flip it from simulated-non-detection to
    simulated-detection, and vice versa, so detectability itself carries
    theta information (not just the continuous value).

    A simulated non-detection reports the threshold `3*sigma` itself (matching
    the real catalog's convention -- an upper-limit row's `ew` field holds the
    reported N-sigma threshold, not a raw flux draw), NOT
    `min(ew_sim_raw, 3*sigma)` -- that would leave deeply negative raw draws
    unchanged instead of censoring them, since min() already favors the
    smaller number regardless of how far below threshold it is."""
    ew_sim_raw  = ew_pred + rng.normal(0.0, sigma)
    is_detected = ew_sim_raw > 3.0 * sigma
    ew_channel  = np.where(is_detected, ew_sim_raw, 3.0 * sigma)
    return _pack_x(ew_channel, ~is_detected)


def _build_x_obs():
    """x for the actual real catalog -- the real reported EW/upper-limit
    status directly, no noise re-injection (it's already a measurement)."""
    s = rdr._S
    return _pack_x(s.ew_obs, s.is_upper_limit)


def _theta_to_spheres(theta, n_bub):
    """theta (length 4*n_bub) -> list of (x, y, z, r) tuples, same centered
    comoving-Mpc frame as `_S.x_gal/y_gal/z_gal` -- theta is already
    parametrized in this frame (see `_prior_transform`/`_inside_and_z_end`)."""
    theta = np.asarray(theta)
    return [tuple(theta[4 * i:4 * i + 4]) for i in range(n_bub)]


# ── Shared 21cmFAST lightcone: replaces get_content's per-galaxy-independent
#    "outside bubble" draw with a real reionization field, shared across every
#    galaxy within one simulation and ray-traced per sightline -- see
#    .claude/plans/bright-growing-goblet.md for the full design. Runs entirely
#    in the PARENT process (per the chosen "simplicity over parallelism"
#    architecture), so the fork pool below only ever does the cheap
#    noise-injection step, as before. `get_content`/`_refresh_mc_state`
#    (speed_up.py/real_data_run.py) are untouched -- still called for
#    `j_s`/`ew_int`/etc.; their `tau_prec`/`z_up`/`red_up`/`z_lo`/`red_lo`
#    outputs are simply not used by this path. ──────────────────────────────

_SNAPSHOT_FIELD_CACHE = {}   # path -> (256,256,256) float32 array, module-level,
                             # persists across batches/splits within one process
                             # (only 13 distinct snapshots exist -- ~800 MiB worst case)


def _load_snapshot_cached(path):
    field = _SNAPSHOT_FIELD_CACHE.get(path)
    if field is None:
        field = lf.load_neutral_fraction_field(path)
        _SNAPSHOT_FIELD_CACHE[path] = field
    return field


def _compute_lightcone_tau_batch(theta_batch, n_bub, z0, main_dir, rng):
    """For each simulation k in this batch: draw x_H_target ONCE (shared by
    every galaxy -- this alone also fixes the part of the original bug where
    x_H was independently redrawn per galaxy, even before the field-sharing
    fix), select+load(+cache) a matching snapshot, draw a fresh periodic-box
    augmentation, carve this simulation's theta bubble(s) in, and ray-trace
    every galaxy's sightline through the shared result.

    Returns tau_lightcone, shape (n_gal, batch_size, n_wave) -- the
    theta-dependent "outside the fitted bubble" IGM optical depth, to be
    combined with `_S.j_s`/`_S.ew_int`/`_S.tau_cgm` (unchanged, from
    `_refresh_mc_state`) downstream.
    """
    s = rdr._S
    n_gal = len(s.x_gal)
    batch_size = len(theta_batch)
    wave_em_vals = rdr.wave_em.value
    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value

    tau_out = np.empty((n_gal, batch_size, len(wave_em_vals)), dtype=np.float64)

    for k in range(batch_size):
        theta = theta_batch[k]
        inside_gals, z_end_bub_arr = rdr._inside_and_z_end(theta, n_bub)
        bubbles = _theta_to_spheres(theta, n_bub)

        x_h_target = float(get_xH(z0, main_dir=main_dir))
        snap = lf.select_snapshot(x_h_target)
        field = _load_snapshot_cached(snap.path)
        aug = lf.draw_augmentation(rng)

        for g in range(n_gal):
            # z_end_bub_arr[g] == s.redshifts[g] exactly for a galaxy not
            # inside any bubble (see _inside_and_z_end), so this single
            # real-cosmology conversion handles both cases uniformly --
            # for a not-inside galaxy it reproduces z_gal[g] exactly (since
            # that's how radec_to_comoving computed it in the first place).
            z_start_mpc = (Cosmo.comoving_distance(z_end_bub_arr[g]).to(u.Mpc).value
                          - d_c0)
            tau_out[g, k, :] = lf.ray_trace_outside_tau(
                field, aug, s.x_gal[g], s.y_gal[g], z_start_mpc, d_c0,
                s.redshifts[g], wave_em_vals, bubbles=bubbles, x_h_tail=x_h_target,
            )

    return tau_out


def _ew_pred_from_lightcone_tau(k_idx):
    """Combine this simulation's ray-traced tau with the (theta-independent,
    already-populated-by-_refresh_mc_state) line profile/intrinsic-EW/CGM
    draws -- same trapz/exp(-tau) combination `_ew_and_t_for_inside`
    (real_data_run.py) already does, just reading `_S.lightcone_tau` instead
    of the discrete-bubble tau_prec/tau_post."""
    s = rdr._S
    tau_now  = s.lightcone_tau[:, k_idx, :]           # (n_gal, n_wave)
    j_s_k    = s.j_s[:, k_idx, :]                      # (n_gal, n_wave)
    weighted = j_s_k * s.tau_cgm * np.exp(-tau_now)
    numerator = np.trapz(weighted, rdr.wave_em.value, axis=1)
    t_in = numerator / s.j_s_trapz_denom[:, k_idx]
    return s.ew_int[:, k_idx] * t_in


def simulate_one(k_idx, rng):
    """One stochastic forward-model draw: model-predicted EW for this
    MC-draw index (theta's effect is already baked into `_S.lightcone_tau[
    :, k_idx, :]`, computed in the parent -- see `_compute_lightcone_tau_batch`),
    then one noisy/censored simulated x."""
    ew_pred = _ew_pred_from_lightcone_tau(k_idx)
    return _build_x(ew_pred, _per_galaxy_sigma(), rng)


# ── Bulk (theta, x) generation ──────────────────────────────────────────────

def _simulate_worker(task):
    """Runs in a forked worker -- `_S` is fully populated pre-fork (parent
    calls `_refresh_mc_state` AND `_compute_lightcone_tau_batch` before
    creating the Pool), inherited via copy-on-write with no repickling, same
    pattern as `_run_dynesty`. Unlike the pre-lightcone version, `theta`
    itself is no longer needed here -- its effect on the prediction was
    already computed in the parent (the ray trace), so this step is purely
    the noise-injection/detection-decision layer."""
    i, k_idx, noise_seed = task
    rng = np.random.default_rng(noise_seed)
    return i, simulate_one(k_idx, rng)


def _generate_split(meta, n_bub, n_sim, batch_size, n_workers, main_dir,
                    seed, output_dir, prefix):
    """Bulk-generate `n_sim` (theta, x) pairs in resumable batches of
    `batch_size`, saved as `{prefix}_batch_{i:05d}.npz`. Each batch calls
    `_refresh_mc_state` (unchanged) and `_compute_lightcone_tau_batch` (new,
    the shared-field ray trace) once -- both in the parent, before forking --
    then parallelizes the noise/detection step across a fork pool, 1:1
    paired via k_idx."""
    ndim = _N_BUB_TO_NDIM[n_bub]
    prior_transform = _N_BUB_TO_PRIOR_TRANSFORM[n_bub]
    rng_master = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)
    n_done, batch_idx = 0, 0
    while n_done < n_sim:
        this_batch = min(batch_size, n_sim - n_done)
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")
        if os.path.exists(out_path):
            print(f"[simulate:{prefix}] {out_path} exists, skipping (resumable).", flush=True)
            n_done += this_batch
            batch_idx += 1
            continue

        rdr._refresh_mc_state(
            meta['muv'], meta['redshifts'], meta['x_gal'], meta['y_gal'], meta['z_gal'],
            meta['beta'], meta['z0'], this_batch, main_dir,
        )

        u_batch = rng_master.uniform(size=(this_batch, ndim))
        theta_batch = np.array([prior_transform(u) for u in u_batch])

        rdr._S.lightcone_tau = _compute_lightcone_tau_batch(
            theta_batch, n_bub, meta['z0'], main_dir, rng_master,
        )

        noise_seeds = rng_master.integers(0, 2**31 - 1, size=this_batch)
        tasks = [(i, i, int(noise_seeds[i])) for i in range(this_batch)]

        with mp.get_context('fork').Pool(n_workers) as pool:
            results = pool.map(_simulate_worker, tasks)

        x_dim = results[0][1].shape[0]
        x_batch = np.empty((this_batch, x_dim), dtype=np.float32)
        for i, x in results:
            x_batch[i] = x

        np.savez(out_path, theta=theta_batch, x=x_batch)
        print(f"[simulate:{prefix}] batch {batch_idx}: {this_batch} sims -> {out_path}", flush=True)
        n_done += this_batch
        batch_idx += 1


def load_sims(sims_dir, prefix):
    """Load and concatenate all `{prefix}_batch_*.npz` files in `sims_dir`."""
    paths = sorted(glob.glob(os.path.join(sims_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_batch_*.npz files found in {sims_dir}")
    thetas, xs = [], []
    for p in paths:
        d = np.load(p)
        thetas.append(d['theta'])
        xs.append(d['x'])
    return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)


def run_simulate(args):
    meta = rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    n_sim_val = args.n_sim_val if args.n_sim_val is not None else max(1, args.n_sim // 5)
    _generate_split(meta, args.n_bub, args.n_sim, args.batch_size, args.n_workers,
                    args.main_dir, args.seed, args.output_dir, prefix='train')
    _generate_split(meta, args.n_bub, n_sim_val, args.batch_size, args.n_workers,
                    args.main_dir, args.seed + 1_000_000, args.output_dir, prefix='val')
    print(f"[simulate] done: {args.n_sim} train + {n_sim_val} val sims in {args.output_dir}",
          flush=True)


# ── Prior wrapper for `sbi` ──────────────────────────────────────────────────
#
# NOTE: exact `sbi` API (constructor args, whether `.log_prob()` is required
# by `build_posterior`'s default sampler) is TBD pending confirming the
# installed version on the cluster -- `sbi`/`torch` aren't available in any
# local environment. This is a best-effort implementation of a proper
# torch.distributions-compatible prior matching `_prior_transform`'s ACTUAL
# density (uniform x/y/z box, EMG-truncated r_bub via `R_BUB_DIST_PARAMS`,
# and -- for M2/M3 -- the z-ordering density implied by the sequential
# z2~Uniform(z_lo, z1) construction, NOT a plain independent-uniform box).
# Verify against the installed sbi version before trusting `train`/`infer`.

def _log_prob_theta(theta_batch, n_bub):
    """log p(theta) under the ACTUAL prior_transform density (not a plain
    BoxUniform) -- torch tensor in, numpy out is fine since this only needs
    to be differentiable if sbi's posterior sampler requires gradients
    through the prior (unlikely for a rejection/MCMC-based default sampler;
    revisit if it errors)."""
    s = rdr._S
    theta_batch = np.atleast_2d(np.asarray(theta_batch))
    x_lo, x_hi = s.prior_lo[0], s.prior_hi[0]
    y_lo, y_hi = s.prior_lo[1], s.prior_hi[1]
    z_lo, z_hi = s.prior_lo[2], s.prior_hi[2]
    r_lo, r_hi = s.prior_lo[3], s.prior_hi[3]
    f_lo = exponnorm.cdf(r_lo, *rdr.R_BUB_DIST_PARAMS)
    f_hi = exponnorm.cdf(r_hi, *rdr.R_BUB_DIST_PARAMS)

    def _log_f_r(r):
        pdf = exponnorm.pdf(r, *rdr.R_BUB_DIST_PARAMS) / (f_hi - f_lo)
        return np.where((r >= r_lo) & (r <= r_hi), np.log(np.maximum(pdf, 1e-300)), -np.inf)

    log_p = np.zeros(theta_batch.shape[0])
    if n_bub == 1:
        x, y, z, r = theta_batch.T
        log_p += -np.log(x_hi - x_lo) - np.log(y_hi - y_lo) - np.log(z_hi - z_lo)
        log_p += _log_f_r(r)
    elif n_bub == 2:
        x1, y1, z1, r1, x2, y2, z2, r2 = theta_batch.T
        log_p += -2 * np.log(x_hi - x_lo) - 2 * np.log(y_hi - y_lo)
        log_p += _log_f_r(r1) + _log_f_r(r2)
        valid = (z1 >= z2) & (z2 >= z_lo) & (z1 <= z_hi)
        log_p += np.where(valid, -np.log(z_hi - z_lo) - np.log(np.maximum(z1 - z_lo, 1e-300)), -np.inf)
    elif n_bub == 3:
        x1, y1, z1, r1, x2, y2, z2, r2, x3, y3, z3, r3 = theta_batch.T
        log_p += -3 * np.log(x_hi - x_lo) - 3 * np.log(y_hi - y_lo)
        log_p += _log_f_r(r1) + _log_f_r(r2) + _log_f_r(r3)
        valid = (z1 >= z2) & (z2 >= z3) & (z3 >= z_lo) & (z1 <= z_hi)
        log_p += np.where(
            valid,
            -np.log(z_hi - z_lo) - np.log(np.maximum(z1 - z_lo, 1e-300))
            - np.log(np.maximum(z2 - z_lo, 1e-300)),
            -np.inf,
        )
    else:
        raise ValueError(f"n_bub must be 1, 2, or 3, got {n_bub}")
    return log_p


def _make_sbi_prior(n_bub, device='cpu'):
    """A torch.distributions.Distribution-compatible prior object wrapping
    `_prior_transform`/`_log_prob_theta`, for `sbi.inference.NPE(prior=...)`.
    `device`: must match the `device=` passed to `NPE(...)` -- sbi calls
    `.sample()`/`.log_prob()` on this prior during training/posterior
    building, and a device mismatch between the prior's tensors and the
    density estimator's (e.g. prior on CPU, network on CUDA) will error or
    silently force a slow/incorrect fallback."""
    import torch

    s = rdr._S
    ndim = _N_BUB_TO_NDIM[n_bub]
    prior_transform = _N_BUB_TO_PRIOR_TRANSFORM[n_bub]

    class _BubblePrior(torch.distributions.Distribution):
        arg_constraints = {}
        has_rsample = False

        def __init__(self):
            super().__init__(event_shape=torch.Size([ndim]), validate_args=False)

        def sample(self, sample_shape=torch.Size()):
            n = int(np.prod(sample_shape)) if len(sample_shape) else 1
            u = np.random.uniform(size=(n, ndim))
            theta = np.array([prior_transform(ui) for ui in u])
            out = torch.as_tensor(theta, dtype=torch.float32, device=device)
            return out.reshape(*sample_shape, ndim) if len(sample_shape) else out[0]

        def log_prob(self, value):
            value_np = value.detach().cpu().numpy()
            lp = _log_prob_theta(value_np, n_bub)
            return torch.as_tensor(lp, dtype=torch.float32, device=device)

        @property
        def support(self):
            # Approximate: a per-dimension box, NOT the true support for
            # M2/M3 (which excludes z2 > z1 / z3 > z2 -- `log_prob` already
            # returns -inf there, but this box constraint alone wouldn't
            # reject a support-restriction/rejection sampler that only
            # checks the box). Fine for M1; for M2/M3 this may let a few
            # invalid (wrong z-order) proposals through the box check that
            # `log_prob` then correctly zeroes out downstream -- verify this
            # is actually handled correctly by whatever sbi does with
            # `support` once installed.
            lo = torch.as_tensor(np.tile(s.prior_lo, n_bub), dtype=torch.float32, device=device)
            hi = torch.as_tensor(np.tile(s.prior_hi, n_bub), dtype=torch.float32, device=device)
            return torch.distributions.constraints.interval(lo, hi)

    return _BubblePrior()


# ── Training ─────────────────────────────────────────────────────────────────

def run_train(args):
    import torch
    from sbi.inference import NPE

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but torch.cuda.is_available() "
                           f"is False on this machine/allocation.")
    print(f"[train] device: {args.device}", flush=True)

    theta_train, x_train = load_sims(args.sims_dir, 'train')
    print(f"[train] loaded {len(theta_train)} training sims from {args.sims_dir}", flush=True)

    # _log_prob_theta/_make_sbi_prior read _S.prior_lo/_S.prior_hi -- need the
    # catalog loaded (cheap, no MC draw needed) so the prior's support matches
    # what these sims were actually drawn from.
    rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )

    prior = _make_sbi_prior(args.n_bub, device=args.device)
    inference = NPE(prior=prior, density_estimator=args.density_estimator, device=args.device)
    inference.append_simulations(
        torch.as_tensor(theta_train, dtype=torch.float32, device=args.device),
        torch.as_tensor(x_train, dtype=torch.float32, device=args.device),
    )
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'posterior_m{args.n_bub}.pt')
    torch.save({
        'posterior': posterior,
        'n_bub': args.n_bub,
        'n_sim': len(theta_train),
        'param_names': _N_BUB_TO_PARAM_NAMES[args.n_bub],
    }, out_path)
    print(f"[train] saved trained posterior to {out_path}", flush=True)


# ── Real-data inference ──────────────────────────────────────────────────────

def run_infer(args):
    import torch

    # weights_only=False + map_location: torch.load defaults to the device the
    # checkpoint was SAVED on, which errors if that was 'cuda' and this
    # machine/allocation has no GPU (or vice versa) -- map_location=args.device
    # moves the whole posterior (including its internal network) explicitly.
    checkpoint = torch.load(args.posterior, weights_only=False, map_location=args.device)
    posterior = checkpoint['posterior']
    n_bub = checkpoint['n_bub']
    param_names = checkpoint['param_names']

    meta = rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    x_obs = _build_x_obs()
    x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32, device=args.device)

    samples = posterior.sample((args.n_samples,), x=x_obs_t)
    samples = samples.detach().cpu().numpy()

    post_mean   = samples.mean(axis=0)
    post_median = np.median(samples, axis=0)
    post_std    = samples.std(axis=0)
    post_p16    = np.percentile(samples, 16, axis=0)
    post_p84    = np.percentile(samples, 84, axis=0)
    try:
        post_map = posterior.map(x=x_obs_t).detach().cpu().numpy()
    except Exception as e:   # sbi version-dependent; fall back rather than crash the whole run
        print(f"[infer] posterior.map() unavailable ({e}), using KDE-free sample closest to "
              f"the median as a MAP proxy.", flush=True)
        post_map = samples[np.argmin(np.sum((samples - post_median) ** 2, axis=1))]

    print(f"[infer] M{n_bub} posterior from {args.n_samples} SBI samples:", flush=True)
    for pi, pn in enumerate(param_names):
        print(f"  {pn:6s}  median={post_median[pi]:.3f}  map={post_map[pi]:.3f}  "
              f"std={post_std[pi]:.3f}  [{post_p16[pi]:.2f}, {post_p84[pi]:.2f}]", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'sbi_infer_m{n_bub}.npz')
    np.savez(
        out_path,
        posterior_samples=samples,
        post_mean=post_mean, post_median=post_median, post_std=post_std,
        post_p16=post_p16, post_p84=post_p84, post_map=post_map,
        n_sim=checkpoint['n_sim'], sbi_algorithm='NPE',
        # NOTE: no logz/ncall -- vanilla NPE has no Bayesian-evidence analog,
        # so M1-vs-M2-vs-M3 Bayes-factor comparison (free from dynesty's
        # nested sampling) is NOT available from this output.
        **{k: v for k, v in meta.items() if k not in ('muv', 'beta')},
    )
    print(f"[infer] saved {out_path}", flush=True)


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
    p.add_argument('--n_bub', type=int, default=1, choices=[1, 2, 3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_sim = sub.add_parser('simulate', help='Bulk-generate (theta, x) training pairs.')
    _add_catalog_args(p_sim)
    p_sim.add_argument('--n_sim', type=int, default=15000, help='Training simulations.')
    p_sim.add_argument('--n_sim_val', type=int, default=None,
                       help='Held-out validation simulations (default: 20%% of --n_sim).')
    p_sim.add_argument('--batch_size', type=int, default=2000,
                       help='Simulations per batch (bounds memory; each batch is one fresh '
                            'get_content() MC-draw refresh).')
    p_sim.add_argument('--n_workers', type=int, default=8)
    p_sim.add_argument('--seed', type=int, default=0)
    p_sim.add_argument('--output_dir', type=str, required=True)

    p_train = sub.add_parser('train', help='Train an amortized NPE posterior on generated sims.')
    _add_catalog_args(p_train)
    p_train.add_argument('--sims_dir', type=str, required=True,
                         help='--output_dir from a prior `simulate` run.')
    p_train.add_argument('--device', type=str, default='cpu',
                         help="'cpu', 'cuda', or 'cuda:N' for a specific GPU.")
    p_train.add_argument('--density_estimator', type=str, default='maf',
                         choices=['maf', 'nsf', 'mdn'],
                         help="sbi's NDE family. 'mdn' (mixture density network) is worth trying "
                              "if M2/M3 posteriors look multimodal -- its mixture components map "
                              "naturally onto separate modes, unlike a single flow.")
    p_train.add_argument('--output_dir', type=str, required=True)

    p_infer = sub.add_parser('infer', help='Sample the trained posterior at the real x_obs.')
    _add_catalog_args(p_infer)
    p_infer.add_argument('--posterior', type=str, required=True, help='.pt from `train`.')
    p_infer.add_argument('--n_samples', type=int, default=10000)
    p_infer.add_argument('--device', type=str, default='cpu',
                         help="Should generally match the --device used in `train` -- "
                              "map_location handles moving the checkpoint if it doesn't.")
    p_infer.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    {'simulate': run_simulate, 'train': run_train, 'infer': run_infer}[args.command](args)
