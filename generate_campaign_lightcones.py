"""
Generate the lightcone campaign: 5 F_ESC10 values (calibrated to the Planck
tau_e grid, see cluster-storage-expansion memory, 2026-08-25) x N_SEEDS
independent random seeds each -- the multi-seed morphology library that
replaces the single-box/single-seed simulator currently used by
lyabubbles/lightcone_field.py + sbi_pixel_field.py.

DESIGN: one (F_ESC10, seed) pair per invocation, chosen by a task index
(argv[1]) -- built for a SLURM ARRAY JOB (see generate_campaign_lightcones.sh),
not a shared-mutable-state sequential loop. This is deliberately different
from calibrate_fesc_to_tau.py's design: that script's JSON read-modify-write
pattern silently lost 5 already-evaluated points to a race condition when
two submissions overlapped. Each task here is independent and IDEMPOTENT
(checks for its own marker file and skips if already done), so concurrent
or relaunched array tasks can't corrupt each other -- worst case is wasted
duplicate compute, never lost data. Safe to submit the whole array at once.

KEY PHYSICS POINT (not obvious, derived from the Z_HEAT_MAX validator error
hit while building check_fiducial_astro_params.py): RECOMB_MODEL=
'inhomogeneous' needs `node_redshifts` (what actually gets SIMULATED) to
extend up to/above Z_HEAT_MAX=35 regardless of what redshift range we
actually care about in the output -- inhomogeneous recombination tracking
needs the full high-z history to get the cumulative recombination rate
right. So Z_MAX_SIM stays 40.0 (same as the tau_e calibration runs -- same
per-lightcone COMPUTE cost, no savings there). What DOES shrink is the
RectilinearLightconer's own min/max_redshift (Z_MIN_SAVE/Z_MAX_SAVE below),
which controls what actually gets SLICED into the saved lightcone -- that's
where the real STORAGE savings come from (~5-6x less than the tau_e
calibration lightcones' implicit 5.3-40 span). This node_redshifts-vs-
lightconer-range split is standard 21cmFAST v4 usage (simulate wide, save
narrow) but this specific wide/narrow combination hasn't been empirically
run yet -- report back if it errors.

Z_MAX_SAVE=8.5 chosen to match/slightly exceed the historical single-box
snapshot table's ceiling (z=8.28, see lyabubbles/lightcone_field.py) and
give margin above real_data_run.py's z_hi=7.3 default -- ADJUST if your
actual catalog z-window differs.

quantities=('neutral_fraction',) only -- brightness_temp dropped
deliberately: unreliable with USE_TS_FLUCT=False (see check_fiducial_
astro_params.py's warning-log note) and nothing downstream uses it. Add
'density' back if a future refinement wants density-weighted optical depth
(lyabubbles/lightcone_field.py's ray-tracing currently only uses x_HI, per
pixel-field-sbi memory's discretize_to_fixed_bins).

Shares every other v4-API design choice already validated in
check_fiducial_astro_params.py / calibrate_fesc_to_tau.py (RECOMB_MODEL,
PHOTON_CONS_TYPE, R_BUBBLE_MAX, HII_DIM/BOX_LEN, cosmo=inputs.cosmo_params.
cosmo fix, node_redshifts-attribute fallback) -- deliberately duplicated,
not imported, so each cluster script stays self-contained; update all
three together if any of those choices change.

Usage:
    python generate_campaign_lightcones.py <task_index>
    task_index in [0, N_ESC10 * N_SEEDS) -- see .sh wrapper for the array range.
"""
import json
import os
import sys

import numpy as np
import py21cmfast as p21c

# ---- calibrated F_ESC10 grid (cluster-storage-expansion memory, 2026-08-25) ----
F_ESC10_GRID = {
    -2: -1.9536,
    -1: -1.6087,
     0: -1.3933,
     1: -1.2151,
     2: -1.0415,
}
N_SEEDS = 8   # seeds per F_ESC10 value -- the 8-10 originally planned; ADJUST here if wanted

# ---- shared fixed params, identical to check_fiducial_astro_params.py / calibrate_fesc_to_tau.py ----
BASE_OVERRIDES = dict(
    F_STAR10=-1.21,
    ALPHA_STAR=0.50,
    M_TURN=8.65,
    t_STAR=0.55,
    ALPHA_ESC=-0.498862738992503996,
    hlittle=0.6688,
    OMm=0.321,
    OMb=0.04952,
    POWER_INDEX=0.9626,
    SIGMA_8=0.8118,
    RECOMB_MODEL='inhomogeneous',
    PHOTON_CONS_TYPE='z-photoncons',
    R_BUBBLE_MAX=50.0,
    HII_DIM=256,
    BOX_LEN=384,     # transverse box unchanged from the validated setup -- NOT yet
                     # shrunk to the real catalog's actual sky footprint (the
                     # "size the transverse box to the survey field" refinement
                     # from the original campaign plan is still open/deferred,
                     # not applied here to keep this rollout on already-validated
                     # ground rather than introduce a second new unknown at once)
    N_THREADS=16,    # ADJUST to match --cpus-per-task
)

CACHE_DIR    = '/lustre/astro/ivannik/21cmFAST_cache/lightcone_campaign/'
MANIFEST_DIR = '/lustre/astro/ivannik/21cmFAST_cache/lightcone_campaign_manifest/'
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

Z_MIN         = 5.3    # matches Z_END_DEFAULT in lyabubbles/lightcone_field.py
Z_MAX_SIM     = 40.0   # node_redshifts range -- must stay wide, see module docstring
Z_MAX_SAVE    = 8.5    # RectilinearLightconer range -- this is what actually gets saved


def parse_task_index(idx: int):
    """idx -> (sigma_key, F_ESC10, seed). Ordering: F_ESC10 varies fastest,
    so early task indices sample across the whole grid quickly rather than
    exhausting one F_ESC10's seeds before starting the next -- useful if you
    only get through part of the array before deciding you have enough."""
    sigma_keys = sorted(F_ESC10_GRID.keys())
    n_esc = len(sigma_keys)
    sigma_key = sigma_keys[idx % n_esc]
    seed = idx // n_esc
    return sigma_key, F_ESC10_GRID[sigma_key], seed


def main():
    if len(sys.argv) != 2:
        n_total = len(F_ESC10_GRID) * N_SEEDS
        print(f"Usage: python generate_campaign_lightcones.py <task_index 0..{n_total - 1}>")
        sys.exit(1)
    idx = int(sys.argv[1])
    sigma_key, f_esc10, seed = parse_task_index(idx)
    tag = f"fesc{sigma_key:+d}sigma_seed{seed}"
    marker = os.path.join(MANIFEST_DIR, tag + '.json')

    if os.path.exists(marker):
        print(f"[skip] {tag} already completed (marker found at {marker}) -- idempotent, nothing to do.")
        return

    print(f"[task {idx}] {tag}: F_ESC10={f_esc10:+.4f} (sigma={sigma_key:+d}), seed={seed}")

    node_z = p21c.wrapper.inputs.get_logspaced_redshifts(
        min_redshift=Z_MIN, z_step_factor=1.05, max_redshift=Z_MAX_SIM,
    )
    inputs = p21c.InputParameters.from_template(
        ['simple'],
        node_redshifts=node_z,
        random_seed=seed,
        F_ESC10=f_esc10,
        **BASE_OVERRIDES,
    )

    cache = p21c.OutputCache(CACHE_DIR)
    lcn = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=Z_MIN,
        max_redshift=Z_MAX_SAVE,   # narrower than node_redshifts -- see module docstring
        quantities=("neutral_fraction",),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )

    lightcone = p21c.run_lightcone(lightconer=lcn, inputs=inputs, cache=cache, progressbar=True)
    if not hasattr(lightcone, "node_redshifts"):
        lightcone = lightcone.exhaust_lightcone()

    print(f"[done] {tag}: {len(lightcone.node_redshifts)} node redshifts saved "
          f"(z={Z_MIN}-{Z_MAX_SAVE}), cached in {CACHE_DIR}")

    with open(marker, 'w') as f:
        json.dump({
            'task_index': idx, 'sigma': sigma_key, 'F_ESC10': f_esc10, 'seed': seed,
            'z_min': Z_MIN, 'z_max_save': Z_MAX_SAVE, 'z_max_sim': Z_MAX_SIM,
            'n_node_redshifts_saved': len(lightcone.node_redshifts),
        }, f, indent=2)
    print(f"[marker] wrote {marker}")


if __name__ == '__main__':
    main()
