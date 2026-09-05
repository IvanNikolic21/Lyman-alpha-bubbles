"""
JWST proposal plot, compute step: does a small survey AREA bias the inferred
bubble size distribution (BSD) low, relative to what a wider survey would
recover? No galaxies here by design (per explicit instruction) -- this is a
pure field-level argument: given one 21cmFAST ionization field, how does the
BSD measured through a small transverse footprint compare to a larger one,
and to the field's own "true" (unrestricted) BSD?

This script does ONLY the (potentially slow) computation and saves raw
per-ray results to a .npz -- plotting is a separate script
(plot_bsd_survey_area.py) so the numbers can be inspected/sanity-checked on
their own before anything is rendered.

Method: mean free path (MFP), Mesinger & Furlanetto 2007 -- from a random
ionized cell, cast a ray in a random 3D direction, walk it in sub-cell steps
until it hits a neutral cell (a "bubble size" sample) or leaves the
accessible volume. This is the standard method in the reionization-bubble
literature and is directly sensitive to the finite-survey-volume truncation
this plot is meant to demonstrate.

Three cases, same field, same threshold, same ray budget -- only the
transverse footprint differs:
  - 'current'  : 70 arcmin^2 (your real proposal number)
  - 'proposed' : 140 arcmin^2 (your real proposal number)
  - 'full_box' : no transverse restriction -- periodic in all 3 directions,
                 the field's own best estimate of its "true" BSD, i.e. what
                 the current/proposed cases are being biased away from.
                 Caveat (real, not hidden): even this is still just one
                 384 Mpc periodic box, so it is itself not perfectly
                 unbiased for the largest bubbles -- it is a reference, not
                 a ground truth from an arbitrarily large volume.

For 'current'/'proposed': the transverse (x, y) extent is converted from
your arcmin^2 numbers via the ang.-diameter/comoving-transverse relation at
the snapshot's redshift (assumed SQUARE footprint, side = sqrt(area) -- flag
this if you actually want a non-square shape). The box is tiled into as many
non-overlapping footprint-sized columns as fit (each spanning the FULL box
depth in the line-of-sight/z direction -- z is treated identically to the
full_box case, periodic, un-truncated in both 'current' and 'proposed', so
the ONLY thing being compared is the transverse AREA, which is what the
proposal argument is actually about, not an assumption about how deep any
specific spectroscopic survey probes). Rays are cast from random ionized
cells pooled across ALL tiles of that footprint size -- this is the
"tile/subsample within one box" statistics plan already agreed on, not
multiple independent 21cmFAST realizations.

A ray that reaches its tile's transverse (x/y) wall without hitting a
neutral cell is CAPPED there and recorded as a real sample at that
(truncated) distance, with status='capped_by_area' -- this, not merely
excluding it, is what actually encodes the survey-area bias: an observer
limited to that footprint cannot see past it, so a bubble that is really
larger reads as only exactly as large as the footprint allowed them to see.

Neutral fraction is treated purely as an input (per instruction) -- this
reuses an ALREADY-EXTRACTED snapshot field (extract_real_snapshot_field.py),
not a fresh 21cmFAST run. Z_SNAPSHOT below picks WHICH cached snapshot --
must exactly match one of lyabubbles/lightcone_field.py's `_RAW_SNAPSHOTS`
entries (there is no continuous z control, only the 13 cached timesteps of
that one simulation) -- and both FIELD_PATH/OUT_PATH are derived from it, so
switching redshift is a one-line change and never silently overwrites a
previous redshift's results.
"""
import time

import numpy as np
from astropy.cosmology import Planck18 as Cosmo
from astropy import units as u

REPO = "/Users/dxf836/Documents/git_code/Lyman-alpha-bubbles"

Z_SNAPSHOT = 7.2436               # closest cached entry to the requested "z~7.3"
                                  # (x_H=0.4738, i.e. ~53% ionized -- see
                                  # lyabubbles/lightcone_field.py's _RAW_SNAPSHOTS
                                  # for the other 12 available timesteps)
                                  # -- z=6.5 results/plots from earlier stay on
                                  # disk untouched; this script now only targets
                                  # z=7.2436 per instruction to focus on it.
FIELD_PATH = f"/Users/dxf836/Downloads/real_snapshot_z{Z_SNAPSHOT:.4f}_field.npy"
OUT_PATH = f"{REPO}/bsd_survey_area_results_z{Z_SNAPSHOT:.4f}.npz"

BOX_LEN_MPC = 384.0
THRESHOLD = 0.5                  # neutral_frac >= THRESHOLD -> neutral; matches
                                  # compare_real_vs_mock.py's established convention

# your real proposal numbers -- 210 added as a third point (3x current) to
# see how much of the gap to the full-box reference a further increase
# closes, since 140 alone only closed part of it.
AREA_ARCMIN2 = {"current": 70.0, "proposed": 140.0, "proposed_210": 210.0}

N_RAYS_TOTAL = 200_000            # per case
# STEP_FRAC_OF_CELL=0.5 (0.75 Mpc steps) quantizes each ray's recorded R to
# one of only ~150-350 distinct values across 200,000 rays -- visibly jagged
# once histogrammed (bins straddle inconsistent numbers of quantization
# levels). 0.15 (0.225 Mpc) gives ~3.3x finer resolution; MAX_STEPS scaled
# up to match so the physical distance cap (MAX_STEPS*step_size) is
# unchanged. Runtime is still well under a minute even so (was ~1s/case at
# the coarser step).
STEP_FRAC_OF_CELL = 0.15
MAX_STEPS = 1333
SEED = 0


def load_field():
    neutral_frac = np.load(FIELD_PATH).astype(np.float32)
    n_cell = neutral_frac.shape[0]
    cell_size = BOX_LEN_MPC / n_cell
    ionized = neutral_frac < THRESHOLD
    print(f"[load] {FIELD_PATH}: shape={neutral_frac.shape}, cell_size={cell_size:.4f} Mpc, "
          f"mean(neutral_frac)={neutral_frac.mean():.4f}, ionized_fraction(<{THRESHOLD})="
          f"{ionized.mean():.4f}", flush=True)
    return neutral_frac, ionized, n_cell, cell_size


def arcmin2_to_side_mpc(area_arcmin2, z):
    """sqrt(area) side length, arcmin -> comoving Mpc via the flat-LambdaCDM
    comoving-transverse relation (angle_rad * D_C(z), the same convention
    already used throughout this project, e.g. radec_to_comoving)."""
    d_c = Cosmo.comoving_distance(z).to(u.Mpc).value
    side_arcmin = np.sqrt(area_arcmin2)
    side_mpc = (side_arcmin * u.arcmin).to(u.rad).value * d_c
    return side_mpc, d_c


def build_tiles(n_cell, cell_size, side_mpc):
    """Non-overlapping square tiles of `side_mpc` (rounded to whole cells),
    covering as much of the transverse [0, n_cell) grid as divides evenly --
    any leftover partial tile at the high edge is dropped (kept simple; the
    box is much bigger than the footprint so this loses little coverage)."""
    n_cells_side = max(1, int(round(side_mpc / cell_size)))
    n_tiles_per_axis = n_cell // n_cells_side
    tiles = []
    for ti in range(n_tiles_per_axis):
        for tj in range(n_tiles_per_axis):
            tiles.append((ti * n_cells_side, (ti + 1) * n_cells_side,
                         tj * n_cells_side, (tj + 1) * n_cells_side))
    actual_side_mpc = n_cells_side * cell_size
    print(f"[tiles] target side={side_mpc:.2f} Mpc -> {n_cells_side} cells/side "
          f"(actual side={actual_side_mpc:.2f} Mpc), {n_tiles_per_axis}x{n_tiles_per_axis} = "
          f"{len(tiles)} non-overlapping tiles covering the box", flush=True)
    return tiles, n_cells_side, actual_side_mpc


def sample_start_points(ionized, tiles, n_rays_total, rng):
    """Pool ionized-cell coordinates across ALL tiles of this footprint size,
    then draw n_rays_total starting points (with replacement) from that
    pool, carrying each ray's own tile's transverse (x_lo, x_hi, y_lo, y_hi)
    bounds in Mpc alongside it. Full-z-depth per tile (k unrestricted), per
    the module docstring."""
    all_i, all_j, all_k, all_bounds_idx = [], [], [], []
    # tile_bounds_mpc has ONE entry per tile in `tiles`, in the SAME order --
    # real bug caught before running: an earlier version only appended this
    # for non-empty tiles, which silently desynced it from `all_bounds_idx`
    # (which stores the ORIGINAL t_idx) the moment any tile had zero ionized
    # cells, misassigning every later tile's bounds to the wrong rays. Kept
    # in lockstep with `tiles` unconditionally instead, regardless of
    # whether a given tile contributes any starting points.
    tile_bounds_mpc = [(i_lo, i_hi, j_lo, j_hi) for (i_lo, i_hi, j_lo, j_hi) in tiles]
    for t_idx, (i_lo, i_hi, j_lo, j_hi) in enumerate(tiles):
        sub = ionized[i_lo:i_hi, j_lo:j_hi, :]
        ii, jj, kk = np.nonzero(sub)
        if len(ii) == 0:
            continue
        all_i.append(ii + i_lo)
        all_j.append(jj + j_lo)
        all_k.append(kk)
        all_bounds_idx.append(np.full(len(ii), t_idx, dtype=np.int64))
    if not all_i:
        raise ValueError("No ionized cells found in ANY tile -- threshold/field mismatch?")
    all_i = np.concatenate(all_i)
    all_j = np.concatenate(all_j)
    all_k = np.concatenate(all_k)
    all_bounds_idx = np.concatenate(all_bounds_idx)

    pick = rng.integers(0, len(all_i), size=n_rays_total)
    return all_i[pick], all_j[pick], all_k[pick], all_bounds_idx[pick], tile_bounds_mpc


def run_mfp(ionized, n_cell, cell_size, rng, tiles=None, n_rays_total=N_RAYS_TOTAL):
    """Vectorized MFP ray-marching for one case. `tiles=None` means the
    'full_box' case: periodic in x/y/z, no transverse cap. `tiles` given
    means area-restricted: periodic in z only, capped (opaque) at each
    ray's own tile's x/y walls."""
    step_size = STEP_FRAC_OF_CELL * cell_size

    if tiles is None:
        ii, jj, kk = np.nonzero(ionized)
        pick = rng.integers(0, len(ii), size=n_rays_total)
        start_i, start_j, start_k = ii[pick], jj[pick], kk[pick]
        x_lo = y_lo = np.zeros(n_rays_total)
        x_hi = y_hi = np.full(n_rays_total, BOX_LEN_MPC)
        area_capped_possible = False
    else:
        start_i, start_j, start_k, tile_of_ray, tile_bounds_mpc = sample_start_points(
            ionized, tiles, n_rays_total, rng)
        bounds_arr = np.array([(i_lo * cell_size, i_hi * cell_size,
                               j_lo * cell_size, j_hi * cell_size)
                              for (i_lo, i_hi, j_lo, j_hi) in tile_bounds_mpc])
        x_lo, x_hi, y_lo, y_hi = bounds_arr[tile_of_ray].T
        area_capped_possible = True

    n = len(start_i)
    pos = np.stack([(start_i + 0.5) * cell_size, (start_j + 0.5) * cell_size,
                    (start_k + 0.5) * cell_size], axis=1)

    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    dist = np.zeros(n)
    status = np.full(n, "hit_max_steps", dtype=object)
    active = np.ones(n, dtype=bool)

    for step in range(MAX_STEPS):
        if not active.any():
            break
        idx = np.where(active)[0]
        pos[idx] += dirs[idx] * step_size
        dist[idx] += step_size

        px, py, pz = pos[idx, 0], pos[idx, 1], pos[idx, 2]
        pz_wrapped = np.mod(pz, BOX_LEN_MPC)
        pos[idx, 2] = pz_wrapped

        if area_capped_possible:
            out_of_area = (px < x_lo[idx]) | (px >= x_hi[idx]) | (py < y_lo[idx]) | (py >= y_hi[idx])
        else:
            px_wrapped = np.mod(px, BOX_LEN_MPC)
            py_wrapped = np.mod(py, BOX_LEN_MPC)
            pos[idx, 0] = px_wrapped
            pos[idx, 1] = py_wrapped
            out_of_area = np.zeros(len(idx), dtype=bool)

        ci = np.clip((pos[idx, 0] / cell_size).astype(int), 0, n_cell - 1)
        cj = np.clip((pos[idx, 1] / cell_size).astype(int), 0, n_cell - 1)
        ck = np.clip((pos[idx, 2] / cell_size).astype(int), 0, n_cell - 1)
        hit_neutral = ~ionized[ci, cj, ck]

        newly_capped = idx[out_of_area & ~hit_neutral]
        newly_hit = idx[hit_neutral]

        status[newly_capped] = "capped_by_area"
        status[newly_hit] = "hit_neutral"
        active[newly_capped] = False
        active[newly_hit] = False

    return dist, status


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    neutral_frac, ionized, n_cell, cell_size = load_field()

    results = {}
    print(f"\n[case] full_box: periodic MFP, {N_RAYS_TOTAL} rays", flush=True)
    t1 = time.perf_counter()
    dist, status = run_mfp(ionized, n_cell, cell_size, rng, tiles=None)
    print(f"  done in {time.perf_counter()-t1:.1f}s -- mean R={dist.mean():.2f} Mpc, "
          f"median R={np.median(dist):.2f} Mpc, hit_max_steps frac="
          f"{(status=='hit_max_steps').mean():.4f}", flush=True)
    results["full_box"] = dict(dist=dist, status=status, side_mpc=BOX_LEN_MPC, n_tiles=1)

    for label, area in AREA_ARCMIN2.items():
        side_mpc, d_c = arcmin2_to_side_mpc(area, Z_SNAPSHOT)
        print(f"\n[case] {label}: {area} arcmin^2 -> side={side_mpc:.2f} Mpc "
              f"(D_C(z={Z_SNAPSHOT})={d_c:.1f} Mpc)", flush=True)
        tiles, n_cells_side, actual_side_mpc = build_tiles(n_cell, cell_size, side_mpc)
        t1 = time.perf_counter()
        dist, status = run_mfp(ionized, n_cell, cell_size, rng, tiles=tiles)
        frac_capped = (status == "capped_by_area").mean()
        frac_neutral = (status == "hit_neutral").mean()
        frac_maxstep = (status == "hit_max_steps").mean()
        print(f"  done in {time.perf_counter()-t1:.1f}s -- mean R={dist.mean():.2f} Mpc, "
              f"median R={np.median(dist):.2f} Mpc", flush=True)
        print(f"  status breakdown: capped_by_area={frac_capped:.4f}, "
              f"hit_neutral={frac_neutral:.4f}, hit_max_steps={frac_maxstep:.4f}", flush=True)
        results[label] = dict(dist=dist, status=status, side_mpc=actual_side_mpc,
                              n_tiles=len(tiles), area_arcmin2=area, target_side_mpc=side_mpc)

    np.savez(
        OUT_PATH,
        cell_size=cell_size, n_cell=n_cell, box_len_mpc=BOX_LEN_MPC, threshold=THRESHOLD,
        z_snapshot=Z_SNAPSHOT, step_size_mpc=STEP_FRAC_OF_CELL * cell_size, max_steps=MAX_STEPS,
        n_rays_total=N_RAYS_TOTAL, seed=SEED,
        **{f"{k}__dist": v["dist"] for k, v in results.items()},
        **{f"{k}__status": v["status"] for k, v in results.items()},
        **{f"{k}__side_mpc": v["side_mpc"] for k, v in results.items()},
        **{f"{k}__n_tiles": v["n_tiles"] for k, v in results.items()},
    )
    print(f"\n[saved] {OUT_PATH}", flush=True)
    print(f"[done] total wall time {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()