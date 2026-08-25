"""
Calibrate F_ESC10 (holding ALPHA_ESC and everything else fixed at the
pipeline's other fiducial values) to hit 5 target tau_e values spanning
Planck 2018's tau_e = 0.0544 +/- 0.0073 -- the F_ESC10 grid for the
lightcone campaign (see cluster-storage-expansion memory).

Confirmed starting point (2026-08-25 cluster run of
check_fiducial_astro_params.py): F_ESC10=-0.971704175935387 (this
pipeline's existing fiducial) gives tau_e=0.0719, which is ABOVE even the
highest target (+2 sigma = 0.0690) -- so every target needs a LOWER
F_ESC10 than the current fiducial. There is no bracket yet in either
direction from a single point.

Strategy (deliberately NOT 5 independent bisections -- much more compute
than needed): build one shared, resumable table of (F_ESC10, tau_e)
evaluations via an outward-expanding search below the fiducial, then use
inverse interpolation across that table to serve all 5 targets from the
same evaluations, refining near each target only if the nearest evaluated
point isn't already close enough. Each invocation of this script evaluates
exactly ONE new point and exits -- rerun it repeatedly (e.g. resubmit as a
SLURM job each time) until it reports all 5 targets satisfied. State is
persisted in STATE_JSON so this is safe to interrupt/resume at any point.

Only runs run_global_evolution + compute_tau (section 2a of
check_fiducial_astro_params.py) -- NOT the expensive full spatial
lightcone (section 2b). That's deferred to the final campaign generation
once these 5 F_ESC10 values are actually chosen.

Shares every v4-API design choice already validated in
check_fiducial_astro_params.py (RECOMB_MODEL='inhomogeneous',
PHOTON_CONS_TYPE='z-photoncons', R_BUBBLE_MAX=50.0, Z_MIN/Z_MAX=5.3/40.0,
HII_DIM/BOX_LEN=256/384, ascending-order sort before compute_tau) -- see
that script's docstring for the reasoning behind each. If any of those
change, update both scripts together (deliberately duplicated, not
imported, so each stays a single self-contained cluster script).
"""
import json
import os

import numpy as np
import py21cmfast as p21c

# ---- shared fixed params (everything except F_ESC10) ----
BASE_OVERRIDES = dict(
    F_STAR10=-1.21,
    ALPHA_STAR=0.50,
    M_TURN=8.65,
    t_STAR=0.55,
    ALPHA_ESC=-0.498862738992503996,   # held fixed -- only F_ESC10 varies
    hlittle=0.6688,
    OMm=0.321,
    OMb=0.04952,
    POWER_INDEX=0.9626,
    SIGMA_8=0.8118,
    RECOMB_MODEL='inhomogeneous',
    PHOTON_CONS_TYPE='z-photoncons',
    R_BUBBLE_MAX=50.0,
    HII_DIM=256,
    BOX_LEN=384,
    N_THREADS=16,   # ADJUST to match --cpus-per-task
)

CACHE_DIR = '/lustre/astro/ivannik/21cmFAST_cache/tau_uvlf_check/'          # same as check_fiducial_astro_params.py
OUT_DIR   = '/lustre/astro/ivannik/21cmFAST_cache/tau_uvlf_check_results/'  # same as check_fiducial_astro_params.py
STATE_JSON = os.path.join(OUT_DIR, 'f_esc_calibration_state.json')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

Z_MIN, Z_MAX = 5.3, 40.0

F_ESC10_FIDUCIAL = -0.971704175935387049
TAU_PLANCK, TAU_PLANCK_ERR = 0.0544, 0.0073
TAU_TARGETS = {n: TAU_PLANCK + n * TAU_PLANCK_ERR for n in (-2, -1, 0, 1, 2)}
TOL = 0.0015   # ~20% of the Planck sigma-spacing -- "close enough" per target; adjust if tighter precision is wanted

STEP_INIT = 0.3     # dex, initial step size for the expanding search below the fiducial
STEP_MAX  = 2.0     # dex, cap so one bad step can't jump absurdly far
STEP_GROW = 1.6      # geometric growth factor if repeated expansion is needed


def evaluate_tau_e(f_esc10: float) -> float:
    """One run_global_evolution + compute_tau call at this F_ESC10, holding
    everything else at BASE_OVERRIDES. Mirrors check_fiducial_astro_params.py
    section 2a exactly (see that script for why each design choice here is
    what it is)."""
    node_z = p21c.wrapper.inputs.get_logspaced_redshifts(
        min_redshift=Z_MIN, z_step_factor=1.05, max_redshift=Z_MAX,
    )
    inputs = p21c.InputParameters.from_template(
        ['simple'],
        node_redshifts=node_z,
        random_seed=0,
        F_ESC10=f_esc10,
        **BASE_OVERRIDES,
    )
    global_evolution = p21c.run_global_evolution(inputs=inputs, progressbar=True)
    z_hist   = np.asarray(global_evolution.node_redshifts)
    xHI_hist = np.asarray(global_evolution.quantities['neutral_fraction'])
    order = np.argsort(z_hist)
    z_hist, xHI_hist = z_hist[order], xHI_hist[order]
    return float(p21c.compute_tau(redshifts=z_hist, global_xHI=xHI_hist, inputs=inputs))


def load_state():
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as f:
            points = json.load(f)
    else:
        points = []
    # Bootstrap from the already-run fiducial point (check_fiducial_astro_
    # params.py's saved result) if this is a fresh state and that file
    # exists -- avoids re-running a point we already have.
    if not points:
        fid_npz = os.path.join(OUT_DIR, 'tau_fiducial_check.npz')
        if os.path.exists(fid_npz):
            tau_fid = float(np.load(fid_npz)['tau_e'])
            points = [{'F_ESC10': F_ESC10_FIDUCIAL, 'tau_e': tau_fid}]
            print(f"[state] bootstrapped from {fid_npz}: F_ESC10={F_ESC10_FIDUCIAL:.6f} -> tau_e={tau_fid:.4f}")
        else:
            points = [{'F_ESC10': F_ESC10_FIDUCIAL, 'tau_e': 0.0719}]
            print("[state] no tau_fiducial_check.npz found -- bootstrapping from the "
                  "printed value (0.0719) instead; less precise than the saved float.")
    return points


def save_state(points):
    with open(STATE_JSON, 'w') as f:
        json.dump(points, f, indent=2)


def pick_next_point(points):
    """Returns (next_F_ESC10, reason) or (None, None) if every target is
    already satisfied to within TOL."""
    f_esc = np.array([p['F_ESC10'] for p in points])
    tau   = np.array([p['tau_e'] for p in points])
    order = np.argsort(f_esc)
    f_esc, tau = f_esc[order], tau[order]   # ascending F_ESC10

    lo_target, hi_target = min(TAU_TARGETS.values()), max(TAU_TARGETS.values())

    # Phase 1: expand outward until the full target tau_e range is bracketed.
    if tau.min() > lo_target:
        # Haven't gone low enough in F_ESC10 yet -- step down from the
        # lowest evaluated point. Grow the step if this branch keeps firing
        # (each fire beyond the first doubles-ish the previous gap).
        n_down_steps = sum(1 for p in points if p.get('_direction') == 'down')
        step = min(STEP_INIT * STEP_GROW ** n_down_steps, STEP_MAX)
        return f_esc.min() - step, f"expanding search: tau_e.min()={tau.min():.4f} still above lowest target {lo_target:.4f}"
    if tau.max() < hi_target:
        # Shouldn't happen given the confirmed fiducial point, but handle
        # it symmetrically in case F_ESC10_FIDUCIAL or the targets change.
        n_up_steps = sum(1 for p in points if p.get('_direction') == 'up')
        step = min(STEP_INIT * STEP_GROW ** n_up_steps, STEP_MAX)
        return f_esc.max() + step, f"expanding search: tau_e.max()={tau.max():.4f} still below highest target {hi_target:.4f}"

    # Phase 2: bracketed -- refine near whichever target has the largest gap
    # to its nearest evaluated point (in tau_e space).
    worst_gap, worst_target, worst_guess = -1.0, None, None
    for n, target_tau in TAU_TARGETS.items():
        nearest_gap = np.min(np.abs(tau - target_tau))
        if nearest_gap > TOL and nearest_gap > worst_gap:
            # Inverse-interpolate F_ESC10 as a function of tau_e (tau_e vs
            # F_ESC10 assumed monotonic increasing over the relevant range --
            # true physically: more escape fraction -> earlier reionization
            # -> higher tau_e).
            guess = float(np.interp(target_tau, tau, f_esc))
            worst_gap, worst_target, worst_guess = nearest_gap, n, guess

    if worst_target is None:
        return None, None   # all targets satisfied
    return worst_guess, (f"refining target n={worst_target:+d}sigma (tau_e={TAU_TARGETS[worst_target]:.4f}), "
                          f"nearest evaluated point is {worst_gap:.4f} away (tol={TOL})")


def main():
    points = load_state()
    next_f_esc10, reason = pick_next_point(points)

    print("\n[state] evaluated points so far:")
    for p in sorted(points, key=lambda p: p['F_ESC10']):
        print(f"    F_ESC10={p['F_ESC10']:+.4f}  ->  tau_e={p['tau_e']:.4f}")

    if next_f_esc10 is None:
        print("\n[done] all 5 targets satisfied to within TOL -- final F_ESC10 per target:")
        f_esc = np.array([p['F_ESC10'] for p in points])
        tau   = np.array([p['tau_e'] for p in points])
        order = np.argsort(f_esc)
        f_esc, tau = f_esc[order], tau[order]
        for n, target_tau in sorted(TAU_TARGETS.items()):
            i_nearest = np.argmin(np.abs(tau - target_tau))
            print(f"    {n:+d} sigma (target tau_e={target_tau:.4f}): "
                  f"F_ESC10={f_esc[i_nearest]:+.4f}  (achieved tau_e={tau[i_nearest]:.4f})")
        return

    print(f"\n[next] evaluating F_ESC10={next_f_esc10:+.4f} -- {reason}")
    tau_e = evaluate_tau_e(next_f_esc10)
    print(f"[result] F_ESC10={next_f_esc10:+.4f} -> tau_e={tau_e:.4f}")

    direction = 'down' if next_f_esc10 < min(p['F_ESC10'] for p in points) else \
                ('up' if next_f_esc10 > max(p['F_ESC10'] for p in points) else 'refine')
    points.append({'F_ESC10': next_f_esc10, 'tau_e': tau_e, '_direction': direction})
    save_state(points)
    print(f"[state] saved to {STATE_JSON} ({len(points)} points total) -- rerun this script for the next point.")


if __name__ == '__main__':
    main()
