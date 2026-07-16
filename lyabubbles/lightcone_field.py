"""
Shared 21cmFAST lightcone field for the SBI simulator's "outside the fitted
bubble" IGM optical depth -- see `.claude/plans/bright-growing-goblet.md` for
the full design writeup.

Replaces the per-galaxy-independent random "outside bubble" draw
(`get_xH`/`get_bubbles`/`calculate_taus_prep` in `speed_up.py`/`igm_prop.py`,
used by the dynesty pipeline and left untouched here) with a real reionization
simulation snapshot shared across every galaxy within one SBI simulation:
picking a coeval box whose neutral fraction matches a drawn target, giving a
quasi-independent view of it via periodic-box augmentation, carving the
fitted bubble(s) into it, and ray-tracing each galaxy's sightline through the
result -- bounded to one box-length (384 cMpc) near the source, handing off
to the existing analytic tail formula (`remaining_tail_tau`) beyond that.

Standalone module: only depends on `lyabubbles.helpers.I` and
`astropy.cosmology` -- no dependency on `real_data_run.py`/`sbi_real_data.py`
internals, so every piece here is independently testable.
"""

import itertools
from dataclasses import dataclass

import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo, z_at_value
from scipy.interpolate import InterpolatedUnivariateSpline

from lyabubbles.helpers import I, wave_Lya

# ── Box geometry (confirmed from the actual snapshot files) ────────────────
BOX_LEN_MPC   = 384.0
N_CELL        = 256
CELL_SIZE_MPC = BOX_LEN_MPC / N_CELL   # 1.5 cMpc/cell
Z_END_DEFAULT = 5.3                     # matches calculate_taus_post_batched's convention


# ── Snapshot table ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SnapshotRecord:
    z: float
    x_h: float
    path: str


_PATH_TEMPLATE = (
    "/lustre/astro/ivannik/21cmFAST_cache/cf853e8a92de487bf9c865a26e65e76d/"
    "1956/ffa852ccaa39d8f82951cc98ff798ab4/{z}/"
    "ff490db45ce98b111ca6e375b0d8c8f0/IonizedBox.h5"
)

# (z, native mean neutral fraction) -- confirmed against np.mean(neutral_fraction)
# for each file by the user.
_RAW_SNAPSHOTS = [
    (6.5000, 0.27111432),
    (6.6158, 0.30373916),
    (6.7681, 0.34622738),
    (6.9235, 0.38924363),
    (7.0819, 0.43000000),
    (7.2436, 0.47382167),
    (7.4084, 0.51404920),
    (7.5766, 0.55213356),
    (7.7481, 0.58826315),
    (7.9231, 0.62222904),
    (8.0000, 0.63884680),
    (8.1016, 0.65660940),
    (8.2836, 0.68588560),
]

SNAPSHOT_TABLE = [
    SnapshotRecord(z=z, x_h=x_h, path=_PATH_TEMPLATE.format(z=f"{z:.4f}"))
    for z, x_h in _RAW_SNAPSHOTS
]


def select_snapshot(x_h_target: float, bubble_volume_fraction: float = 0.0,
                    table: list = None) -> SnapshotRecord:
    """Nearest-neutral-fraction lookup. `bubble_volume_fraction` (default 0,
    i.e. off) applies the volume-bookkeeping correction -- carving the fitted
    bubble(s) out and forcing them ionized removes some native neutral gas,
    so the box's post-carve mean would undershoot `x_h_target` unless a
    slightly-more-neutral-than-target native box is picked. In practice this
    correction is negligible: for the fitted bubble-size prior
    (`R_BUB_DIST_PARAMS`, `real_data_run.py`), bubble-volume/box-volume is
    ~1e-4-5e-4 for typical/90th-percentile radii, well under the spacing
    between the 13 available snapshots' x_H values -- so nearest-neighbor
    lookup with the correction off is what's actually used by default."""
    table = table or SNAPSHOT_TABLE
    if bubble_volume_fraction > 0:
        x_h_native_target = min(x_h_target / (1 - bubble_volume_fraction), 1.0)
    else:
        x_h_native_target = x_h_target
    idx = int(np.argmin([abs(rec.x_h - x_h_native_target) for rec in table]))
    return table[idx]


def load_neutral_fraction_field(path: str) -> np.ndarray:
    """(256, 256, 256) float32 neutral-fraction field from a 21cmFAST
    IonizedBox.h5 cache file (confirmed HDF5 layout:
    f['IonizedBox']['OutputFields']['neutral_fraction'])."""
    with h5py.File(path, "r") as f:
        field = np.asarray(f["IonizedBox"]["OutputFields"]["neutral_fraction"],
                           dtype=np.float32)
    if field.shape != (N_CELL, N_CELL, N_CELL):
        raise ValueError(f"Unexpected field shape {field.shape} from {path}, "
                         f"expected ({N_CELL},{N_CELL},{N_CELL})")
    return field


# ── Periodic-box augmentation (index remap, no field copy) ─────────────────
#
# The 48 symmetries of a cube (6 axis permutations x 8 sign-flip
# combinations) are used, rather than arbitrary rotations, specifically
# because they keep an axis-aligned ray in the augmented frame axis-aligned
# in the raw frame -- cell-stepping along a sightline stays exact, no
# interpolation needed. Combined with a random cyclic shift (exploiting the
# box's periodicity), this gives many statistically-decorrelated "views" of
# the one available box without ever materializing a transformed copy of it.

_AXIS_PERMS  = list(itertools.permutations([0, 1, 2]))       # 6
_SIGN_FLIPS  = list(itertools.product([1, -1], repeat=3))    # 8
CUBE_SYMMETRIES = [(perm, flip) for perm in _AXIS_PERMS for flip in _SIGN_FLIPS]   # 48


@dataclass(frozen=True)
class AugmentationParams:
    shift: tuple          # (dx, dy, dz) cell offset, each in [0, N_CELL)
    symmetry_idx: int     # index into CUBE_SYMMETRIES


def draw_augmentation(rng: np.random.Generator) -> AugmentationParams:
    shift = tuple(int(v) for v in rng.integers(0, N_CELL, size=3))
    symmetry_idx = int(rng.integers(0, len(CUBE_SYMMETRIES)))
    return AugmentationParams(shift=shift, symmetry_idx=symmetry_idx)


def raw_cell_index(aug: AugmentationParams, ax, ay, az):
    """Vectorized augmented-frame cell index -> raw (as-loaded) array index.
    `ax, ay, az` are integer arrays/scalars of augmented-frame cell indices
    (periodic wraparound handled here, i.e. any integer is valid, not just
    [0, N_CELL))."""
    ax = np.asarray(ax) % N_CELL
    ay = np.asarray(ay) % N_CELL
    az = np.asarray(az) % N_CELL
    perm, flip = CUBE_SYMMETRIES[aug.symmetry_idx]
    a = [ax, ay, az]
    flipped = [((-a[k]) % N_CELL) if flip[k] == -1 else a[k] for k in range(3)]
    raw = [None, None, None]
    for k in range(3):
        raw[perm[k]] = flipped[k]
    sx, sy, sz = aug.shift
    raw_x = (raw[0] + sx) % N_CELL
    raw_y = (raw[1] + sy) % N_CELL
    raw_z = (raw[2] + sz) % N_CELL
    return raw_x, raw_y, raw_z


def augmented_cell_index(aug: AugmentationParams, rx, ry, rz):
    """Inverse of `raw_cell_index`: raw array index -> augmented-frame index.
    Used only for testing the round-trip; the ray tracer only ever needs the
    forward direction (`raw_cell_index`)."""
    rx = np.asarray(rx) % N_CELL
    ry = np.asarray(ry) % N_CELL
    rz = np.asarray(rz) % N_CELL
    sx, sy, sz = aug.shift
    perm, flip = CUBE_SYMMETRIES[aug.symmetry_idx]
    u0 = (rx - sx) % N_CELL
    u1 = (ry - sy) % N_CELL
    u2 = (rz - sz) % N_CELL
    raw = [u0, u1, u2]
    # Forward is raw[perm[k]] = flipped[k], i.e. flipped[k] = raw[perm[k]]
    # directly -- NOT via a separately-computed inverse permutation (that
    # was the bug: using perm's inverse here double-undoes the permutation
    # for non-self-inverse perms, i.e. the two 3-cycles among the 6 axis
    # permutations, which is exactly the failure pattern that showed up).
    flipped = [raw[perm[k]] for k in range(3)]
    a = [((-flipped[k]) % N_CELL) if flip[k] == -1 else flipped[k] for k in range(3)]
    return a[0], a[1], a[2]


# ── Real-cosmology comoving-distance <-> redshift (replaces the linear R_H
#    approximation used elsewhere in this codebase, invalid at box scales) ──

_Z_SPLINE_RANGE = (4.5, 10.0)
_Z_SPLINE_N     = 400
_z_of_dc_spline = None   # lazily built, module-level cache


def z_of_comoving_distance(d_c_mpc):
    """Real (not linearized) comoving distance [Mpc] -> redshift, via a
    spline built once by inverting astropy's Cosmo.comoving_distance on a
    coarse grid (then reused for every per-cell lookup in every ray trace --
    a raw `z_at_value` solve per cell would be a real bottleneck given
    O(hundreds) of cells per galaxy x n_gal x batch_size calls)."""
    global _z_of_dc_spline
    if _z_of_dc_spline is None:
        z_grid = np.linspace(*_Z_SPLINE_RANGE, _Z_SPLINE_N)
        dc_grid = Cosmo.comoving_distance(z_grid).to(u.Mpc).value
        # comoving_distance is monotonic increasing in z -- safe to spline z(dc) directly.
        _z_of_dc_spline = InterpolatedUnivariateSpline(dc_grid, z_grid, k=3)
    return _z_of_dc_spline(d_c_mpc)


# ── Tau physics (reused formula, generalized from discrete bubbles to a
#    continuous per-cell neutral fraction) ──────────────────────────────────

def _tau_pref(z_source):
    """Same prefactor as calculate_taus_prep/_post -- see lyabubbles/helpers.py::I
    and igm_prop.py::tau_wv for the original discrete-bubble version."""
    tau_gp = 7.16e5 * ((1 + z_source) / 10) ** 1.5
    r_alpha = 6.25e8 / (4 * np.pi * (const.c / wave_Lya).to(u.Hz).value)
    return tau_gp * r_alpha / np.pi


def segments_to_tau(z_b, z_e, x_hi, z_source, wave_em):
    """(z_b, z_e, x_hi) per cell-run (arrays, same length) -> (len(wave_em),)
    tau array. Wavelength-INdependent geometry (z_b/z_e/x_hi) is computed
    once per galaxy by the ray tracer; this step is the only wavelength-
    dependent part, vectorized over all `wave_em` bins at once.

    `I(x)` (lyabubbles/helpers.py) has a genuine pole at x=1 (the
    `x^4.5/(1-x)` term) -- whenever a segment's `z_b` sits at or very near
    `z_source` itself (always true for a ray's first segment, since it
    starts at the source or a fitted-bubble exit close to it) and a
    wavelength bin resonates at very nearly the same redshift,
    `(1+z_b)/(1+z_wave) -> 1` and `I()` diverges, flipping sign right at the
    pole. This is a pre-existing property of this formula, not new here --
    the exact same thing happens in the current codebase's baseline
    "no bubble" path (`calculate_taus_post_batched(redshifts, redshifts,
    ...)` in `real_data_run.py`, `z_b=z_source` there too), which is why
    `_tau_now_for_inside` already has a wavelength-monotonicity sanity clip.
    Same fix here: a per-segment negative contribution is a numerical
    artifact of the pole, not real negative absorption, so it's clipped to
    zero before summing (never removes genuine signal, since tau can't
    physically be negative)."""
    z_wave = wave_em / 1215.67 * (1 + z_source) - 1   # rest-wave -> resonance redshift
    pref = _tau_pref(z_source)
    ratio_b = (1 + z_b[:, None]) / (1 + z_wave[None, :])
    ratio_e = (1 + z_e[:, None]) / (1 + z_wave[None, :])
    per_segment = (pref * x_hi[:, None] * ratio_b ** 1.5
                  * (I(ratio_b) - I(ratio_e)))
    per_segment = np.clip(per_segment, 0, np.inf)
    return np.sum(per_segment, axis=0)


def _merge_runs(z_b, z_e, x_hi, tol=1e-9):
    """Merge adjacent segments with equal x_HI into single runs (mirrors the
    segment-merging already done in calculate_taus_prep) -- purely a size
    reduction before segments_to_tau, doesn't change the physics."""
    if len(z_b) == 0:
        return z_b, z_e, x_hi
    out_b, out_e, out_x = [z_b[0]], [z_e[0]], [x_hi[0]]
    for i in range(1, len(z_b)):
        if abs(x_hi[i] - out_x[-1]) < tol:
            out_e[-1] = z_e[i]
        else:
            out_b.append(z_b[i]); out_e.append(z_e[i]); out_x.append(x_hi[i])
    return np.array(out_b), np.array(out_e), np.array(out_x)


def ray_trace_segments(field: np.ndarray, aug: AugmentationParams,
                       x_transverse_mpc: float, y_transverse_mpc: float,
                       z_start_mpc: float, d_c0_mpc: float,
                       bubbles: list = None, x_h_tail: float = None,
                       box_len_mpc: float = BOX_LEN_MPC, z_end: float = Z_END_DEFAULT):
    """Ray-trace one galaxy's sightline through the shared, augmented field,
    from `z_start_mpc` (in the catalog's centered comoving-Mpc frame -- the
    SAME convention as `_S.z_gal`; pass `z_gal[g]` itself, or
    `z_gal[g] - dist_arr[g]` from the existing `_inside_and_z_end`/bubble-exit
    computation if the galaxy is inside a fitted bubble) toward the observer
    (decreasing centered-z), for at most `box_len_mpc` of comoving path or
    until `z_end` is reached, whichever comes first.

    `x_transverse_mpc, y_transverse_mpc`: the galaxy's fixed transverse
    position (same centered frame) -- constant along the whole sightline
    (flat-sky approximation, matching `radec_to_comoving`'s convention
    already used throughout this codebase).

    `d_c0_mpc`: `Cosmo.comoving_distance(z0)` in Mpc, the same reference
    constant `radec_to_comoving` uses to center the catalog frame -- needed
    to convert `z_start_mpc` to an absolute comoving distance for the
    real-cosmology redshift lookup.

    `bubbles`: list of `(x, y, z, r)` tuples (same centered frame) -- the
    fitted theta bubble(s). Cells whose center falls inside any of them get
    `x_HI` forced to 0 (a lazy per-sample override -- no field mutation).

    `x_h_tail`: if the ray-trace stops because it exhausted `box_len_mpc`
    (not because it reached `z_end`), a final segment `(z_reached, z_end,
    x_h_tail)` is appended before returning, using the SAME `x_H` used to
    select this snapshot -- i.e. "the universe beyond the ray-traced region
    continues at the same average ionization state." Pass `None` to skip
    this (e.g. for the closed-form/synthetic-field unit tests below, where
    the caller wants only the ray-traced segments back).

    Returns `(z_b, z_e, x_hi)` arrays, adjacent-equal-x_HI runs merged.
    """
    ax = int(np.floor(x_transverse_mpc / CELL_SIZE_MPC))
    ay = int(np.floor(y_transverse_mpc / CELL_SIZE_MPC))

    d_c_start = d_c0_mpc + z_start_mpc
    d_c_offset = z_start_mpc - d_c_start   # z_mpc(d_c) = d_c_offset + d_c, constant along this ray

    d_c_box_cap = d_c_start - box_len_mpc
    d_c_z_end = float(Cosmo.comoving_distance(z_end).to(u.Mpc).value)
    d_c_stop = max(d_c_box_cap, d_c_z_end)
    stopped_at_box_cap = d_c_box_cap >= d_c_z_end   # True unless z_end is reached first

    az = int(np.floor(z_start_mpc / CELL_SIZE_MPC))
    first_step = z_start_mpc - az * CELL_SIZE_MPC
    if first_step <= 0:
        first_step = CELL_SIZE_MPC

    z_b_list, z_e_list, xhi_list = [], [], []
    d_c_cursor = d_c_start
    step = first_step
    while True:
        d_c_next = d_c_cursor - step
        reached_stop = d_c_next <= d_c_stop
        if reached_stop:
            d_c_next = d_c_stop

        z_b = float(z_of_comoving_distance(d_c_cursor))
        z_e = float(z_of_comoving_distance(d_c_next))

        rx, ry, rz = raw_cell_index(aug, np.array([ax]), np.array([ay]), np.array([az]))
        x_hi = float(field[int(rx[0]), int(ry[0]), int(rz[0])])
        if bubbles:
            z_mpc_mid = d_c_offset + 0.5 * (d_c_cursor + d_c_next)
            for (bx, by, bz, br) in bubbles:
                if ((x_transverse_mpc - bx) ** 2 + (y_transverse_mpc - by) ** 2
                        + (z_mpc_mid - bz) ** 2 < br ** 2):
                    x_hi = 0.0
                    break

        z_b_list.append(z_b); z_e_list.append(z_e); xhi_list.append(x_hi)

        if reached_stop:
            break
        d_c_cursor = d_c_next
        az -= 1
        step = CELL_SIZE_MPC

    if x_h_tail is not None and stopped_at_box_cap and d_c_box_cap > d_c_z_end:
        z_reached = float(z_of_comoving_distance(d_c_box_cap))
        z_b_list.append(z_reached); z_e_list.append(z_end); xhi_list.append(float(x_h_tail))

    return _merge_runs(np.array(z_b_list), np.array(z_e_list), np.array(xhi_list))


def ray_trace_outside_tau(field, aug, x_transverse_mpc, y_transverse_mpc,
                          z_start_mpc, d_c0_mpc, z_source, wave_em,
                          bubbles=None, x_h_tail=None,
                          box_len_mpc=BOX_LEN_MPC, z_end=Z_END_DEFAULT):
    """Convenience: ray_trace_segments + segments_to_tau in one call, for one galaxy."""
    z_b, z_e, x_hi = ray_trace_segments(
        field, aug, x_transverse_mpc, y_transverse_mpc, z_start_mpc, d_c0_mpc,
        bubbles=bubbles, x_h_tail=x_h_tail, box_len_mpc=box_len_mpc, z_end=z_end,
    )
    return segments_to_tau(z_b, z_e, x_hi, z_source, wave_em)
