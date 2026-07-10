"""
Bias test for the EW likelihood using real galaxy positions and MUVs.

Keeps positions/MUVs from the real catalog unchanged; assigns synthetic EWs by
sampling a known bubble configuration (truth drawn from the prior) and applying
the forward model.  No observational noise is added:

    ew_obs_mock[i] = mean_k [ ew_int[i,k] * T_truth[i,k] ]
    ew_err_mock[i] = std_k  [ ew_int[i,k] * T_truth[i,k] ]  (floored)

This isolates model/likelihood biases from measurement noise. ew_err and
is_upper_limit from the real catalog are intentionally discarded -- both depend
on the true Lya flux and would introduce circularity. Add noise/ULs in later
stages once any intrinsic bias is characterised.

Usage
-----
python mock_bias_test.py --z_lo 6.8 --z_hi 7.3 \
    --n_bub 1 --n_seeds 8 --nlive 300 --output_dir bias_M1/

python mock_bias_test.py --z_lo 6.8 --z_hi 7.3 \
    --n_bub 2 --n_seeds 8 --nlive 300 --output_dir bias_M2/
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import numpy as np
import real_data_run as rdr   # reuses build_state, _S, likelihoods, _run_dynesty

# Floor on ew_err: prevents near-zero scale for non-emitter (ew_int ≈ 0) galaxies.
EW_ERR_FLOOR = 0.5  # Angstrom

# Minimum galaxies inside the truth bubble(s); seeds with fewer are resampled.
MIN_INSIDE = 2


# ── Per-model EW prediction (truth-forward pass) ────────────────────────────

def _ew_pred_truth_1bub(theta):
    """ew_pred (n_gal, K) for a 1-bubble truth theta."""
    s = rdr._S
    xb, yb, zb, rb = theta
    dx = s.x_gal - xb
    dy = s.y_gal - yb
    dz = s.z_gal - zb
    inside       = dx**2 + dy**2 + dz**2 < rb**2
    dist_arr     = np.where(inside, dz + np.sqrt(np.where(inside, rb**2 - dx**2 - dy**2, 0.0)), 0.0)
    z_end        = s.redshifts - np.where(inside, dist_arr / s.R_H, 0.0)
    inside_gals  = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()
    if len(inside_gals):
        tau_now = rdr._tau_now_for_inside(inside_gals, z_end)
        ew_pred[inside_gals] = rdr._ew_pred_for_inside(inside_gals, tau_now)
    return ew_pred, inside


def _ew_pred_truth_2bub(theta):
    """ew_pred (n_gal, K) for a 2-bubble truth theta."""
    s = rdr._S
    x1, y1, z1, r1, x2, y2, z2, r2 = theta
    dx1 = s.x_gal - x1;  dy1 = s.y_gal - y1;  dz1 = s.z_gal - z1
    dx2 = s.x_gal - x2;  dy2 = s.y_gal - y2;  dz2 = s.z_gal - z2
    in1 = dx1**2 + dy1**2 + dz1**2 < r1**2
    in2 = dx2**2 + dy2**2 + dz2**2 < r2**2
    inside = in1 | in2

    dist1  = dz1 + np.sqrt(np.maximum(r1**2 - dx1**2 - dy1**2, 0.0))
    dist2  = dz2 + np.sqrt(np.maximum(r2**2 - dx2**2 - dy2**2, 0.0))
    z_end1 = np.where(in1, s.redshifts - dist1 / s.R_H, np.inf)
    z_end2 = np.where(in2, s.redshifts - dist2 / s.R_H, np.inf)
    z_end  = np.minimum(z_end1, z_end2)
    inside_gals = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()
    if len(inside_gals):
        tau_now = rdr._tau_now_for_inside(inside_gals, z_end)
        ew_pred[inside_gals] = rdr._ew_pred_for_inside(inside_gals, tau_now)
    return ew_pred, inside


def _ew_pred_truth_3bub(theta):
    """ew_pred (n_gal, K) for a 3-bubble truth theta."""
    s = rdr._S
    x1, y1, z1, r1, x2, y2, z2, r2, x3, y3, z3, r3 = theta
    dx1 = s.x_gal - x1;  dy1 = s.y_gal - y1;  dz1 = s.z_gal - z1
    dx2 = s.x_gal - x2;  dy2 = s.y_gal - y2;  dz2 = s.z_gal - z2
    dx3 = s.x_gal - x3;  dy3 = s.y_gal - y3;  dz3 = s.z_gal - z3
    in1 = dx1**2 + dy1**2 + dz1**2 < r1**2
    in2 = dx2**2 + dy2**2 + dz2**2 < r2**2
    in3 = dx3**2 + dy3**2 + dz3**2 < r3**2
    inside = in1 | in2 | in3

    dist1  = dz1 + np.sqrt(np.maximum(r1**2 - dx1**2 - dy1**2, 0.0))
    dist2  = dz2 + np.sqrt(np.maximum(r2**2 - dx2**2 - dy2**2, 0.0))
    dist3  = dz3 + np.sqrt(np.maximum(r3**2 - dx3**2 - dy3**2, 0.0))
    z_end1 = np.where(in1, s.redshifts - dist1 / s.R_H, np.inf)
    z_end2 = np.where(in2, s.redshifts - dist2 / s.R_H, np.inf)
    z_end3 = np.where(in3, s.redshifts - dist3 / s.R_H, np.inf)
    z_end  = np.minimum(np.minimum(z_end1, z_end2), z_end3)
    inside_gals = np.where(inside)[0]

    ew_pred = s.ew_pred_outside.copy()
    if len(inside_gals):
        tau_now = rdr._tau_now_for_inside(inside_gals, z_end)
        ew_pred[inside_gals] = rdr._ew_pred_for_inside(inside_gals, tau_now)
    return ew_pred, inside


_EW_PRED_FUNCS = {
    1: _ew_pred_truth_1bub,
    2: _ew_pred_truth_2bub,
    3: _ew_pred_truth_3bub,
}
_PRIOR_TRANSFORMS = {
    1: rdr._prior_transform,
    2: rdr._prior_transform_2bub,
    3: rdr._prior_transform_3bub,
}
_LOG_LIKES = {
    1: rdr._log_likelihood_ew,
    2: rdr._log_likelihood_ew_2bub,
    3: rdr._log_likelihood_ew_3bub,
}
_PARAM_NAMES = {
    1: rdr.PARAM_NAMES,
    2: rdr.PARAM_NAMES_2BUB,
    3: rdr.PARAM_NAMES_3BUB,
}
_NDIMS = {1: rdr.NDIM, 2: rdr.NDIM_2BUB, 3: rdr.NDIM_3BUB}
_SAMPLES = {1: 'auto', 2: 'rslice', 3: 'rslice'}


def generate_mock_ew(theta_truth, n_bub, rng):
    """Return noiseless mock EW observations and associated scale for a truth config.

    The intrinsic EW is sampled: one MC draw index k is chosen at random, and
    ew_obs[i] = ew_int[i, k] * T_truth[i, k] -- a single realization of the
    galaxy population, not the mean. ew_err is the std across all K draws,
    which is the self-consistent scale for the likelihood's logsumexp
    marginalization.

    Returns
    -------
    ew_obs      : (n_gal,)  sampled mock observation
    ew_err      : (n_gal,)  std of ew_pred over MC draws (floored at EW_ERR_FLOOR)
    n_inside    : int       galaxies inside the truth bubble(s)
    n_lae_inside: int       inside galaxies with sampled ew_int > 0 (actual LAEs)
    """
    ew_pred_mock, inside_mask = _EW_PRED_FUNCS[n_bub](theta_truth)  # (n_gal, K)
    k = rng.integers(ew_pred_mock.shape[1])
    ew_obs  = ew_pred_mock[:, k]
    ew_err  = np.maximum(ew_pred_mock.std(axis=1), EW_ERR_FLOOR)
    inside_gals = np.where(inside_mask)[0]
    n_lae_inside = int((ew_obs[inside_gals] > 0).sum())
    return ew_obs, ew_err, int(inside_mask.sum()), n_lae_inside


def _sample_truth(n_bub, rng):
    """Draw a truth bubble config uniformly from the inference prior."""
    u = rng.uniform(size=_NDIMS[n_bub])
    return _PRIOR_TRANSFORMS[n_bub](u)


def run_bias_seed(seed, n_bub, nlive, dlogz, n_workers):
    """Run one seed: sample truth, generate mock, run dynesty, return summary dict."""
    rng = np.random.default_rng(seed)

    # Resample truth until: (a) enough galaxies inside, (b) at least one is a LAE.
    # (b) is required so the bubble leaves a detectable imprint in the mock EW data.
    for attempt in range(50):
        theta_truth = _sample_truth(n_bub, rng)
        ew_obs_mock, ew_err_mock, n_inside, n_lae_inside = generate_mock_ew(theta_truth, n_bub, rng)
        if n_inside >= MIN_INSIDE and n_lae_inside >= 1:
            break
        print(f"[seed {seed}] attempt {attempt}: {n_inside} inside, {n_lae_inside} LAE inside -- resampling.", flush=True)
    else:
        raise RuntimeError(f"seed {seed}: could not draw a valid truth after 50 attempts.")

    param_names = _PARAM_NAMES[n_bub]
    print(f"[seed {seed}] truth ({n_inside} inside, {n_lae_inside} LAE): "
          + "  ".join(f"{n}={v:.3f}" for n, v in zip(param_names, theta_truth)), flush=True)

    # Override _S with the mock observations (all detections, no ULs).
    s = rdr._S
    s.ew_obs          = ew_obs_mock
    s.ew_err          = ew_err_mock
    s.is_upper_limit  = np.zeros(len(ew_obs_mock), dtype=bool)

    fit = rdr._run_dynesty(
        _LOG_LIKES[n_bub], _PRIOR_TRANSFORMS[n_bub], _NDIMS[n_bub],
        param_names, nlive, dlogz, n_workers,
        label=f'M{n_bub}-seed{seed}',
        sample=_SAMPLES[n_bub],
    )

    bias_median = fit['post_median'] - theta_truth
    bias_map    = fit['post_map']    - theta_truth
    print(f"[seed {seed}] bias (median-truth): "
          + "  ".join(f"{n}={b:+.3f}" for n, b in zip(param_names, bias_median)), flush=True)
    print(f"[seed {seed}] bias (map   -truth): "
          + "  ".join(f"{n}={b:+.3f}" for n, b in zip(param_names, bias_map)), flush=True)

    return dict(
        seed=seed,
        n_bub=n_bub,
        n_inside=n_inside,
        n_lae_inside=n_lae_inside,
        theta_truth=theta_truth,
        bias_median=bias_median,
        bias_map=bias_map,
        ew_obs_mock=ew_obs_mock,
        ew_err_mock=ew_err_mock,
        prior_lo=rdr._S.prior_lo,
        prior_hi=rdr._S.prior_hi,
        x_gal=rdr._S.x_gal,
        y_gal=rdr._S.y_gal,
        z_gal=rdr._S.z_gal,
        **fit,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lya_catalog',        type=str, default='tb_lya.txt')
    parser.add_argument('--properties_catalog', type=str, default='sample_nirspec_properties.txt')
    parser.add_argument('--prefer',             type=str, default='grating',
                        choices=['grating', 'prism'])
    parser.add_argument('--legacy_catalog',     type=str, default=None,
                        help='If set, load the old single-file fixed-width catalog '
                             '(e.g. table.dat) instead of the CDS two-file catalog.')
    parser.add_argument('--z_lo',         type=float, default=None)
    parser.add_argument('--z_hi',         type=float, default=7.3)
    parser.add_argument('--z_min',        type=float, default=5.0)
    parser.add_argument('--muv_max',      type=float, default=-18.0)
    parser.add_argument('--r_max',        type=float, default=None)
    parser.add_argument('--n_bub',        type=int,   default=1,
                        help='Number of bubbles in the truth configuration (1, 2, or 3).')
    parser.add_argument('--n_seeds',      type=int,   default=8)
    parser.add_argument('--seed_offset',  type=int,   default=0,
                        help='Add to seed index so multiple runs can use non-overlapping seeds.')
    parser.add_argument('--n_inside_tau', type=int,   default=200)
    parser.add_argument('--nlive',        type=int,   default=300)
    parser.add_argument('--dlogz',        type=float, default=0.5)
    parser.add_argument('--n_workers',    type=int,   default=8)
    parser.add_argument('--main_dir',     type=str,
                        default='/groups/astro/ivannik/programs/Lyman-alpha-bubbles/')
    parser.add_argument('--output_dir',   type=str,   default='bias_test_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build state once -- populates rdr._S with all precomputed galaxy quantities.
    rdr.build_state(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi, args.n_inside_tau,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )

    all_results = []
    for i in range(args.n_seeds):
        seed = args.seed_offset + i
        result = run_bias_seed(seed, args.n_bub, args.nlive, args.dlogz, args.n_workers)
        all_results.append(result)

        # Save after each seed so partial results aren't lost.
        out_path = os.path.join(
            args.output_dir,
            f'bias_M{args.n_bub}_seed{seed:03d}.npz',
        )
        np.savez(out_path, **{k: np.asarray(v) for k, v in result.items()
                              if isinstance(v, (int, float, np.ndarray))})
        print(f"[seed {seed}] Saved {out_path}", flush=True)

    # ── Summary across seeds ────────────────────────────────────────────────
    param_names = _PARAM_NAMES[args.n_bub]
    print(f"\n{'='*60}", flush=True)
    print(f"Bias summary: M{args.n_bub}, {args.n_seeds} seeds", flush=True)
    print(f"{'='*60}", flush=True)
    biases_median = np.array([r['bias_median'] for r in all_results])  # (n_seeds, ndim)
    biases_map    = np.array([r['bias_map']    for r in all_results])
    n_insides     = np.array([r['n_inside']    for r in all_results])
    print(f"  n_inside per seed: {n_insides}", flush=True)
    for pi, pn in enumerate(param_names):
        bm = biases_median[:, pi]
        bmap = biases_map[:, pi]
        print(f"  {pn:8s}  median bias: mean={bm.mean():+.3f}  std={bm.std():.3f}  "
              f"| map bias: mean={bmap.mean():+.3f}  std={bmap.std():.3f}", flush=True)

    summary_path = os.path.join(args.output_dir, f'bias_M{args.n_bub}_summary.npz')
    np.savez(
        summary_path,
        param_names=param_names,
        biases_median=biases_median,
        biases_map=biases_map,
        n_insides=n_insides,
        seeds=np.arange(args.seed_offset, args.seed_offset + args.n_seeds),
    )
    print(f"\nSummary saved to {summary_path}", flush=True)