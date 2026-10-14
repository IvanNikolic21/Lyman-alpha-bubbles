"""
Production batched generator for the synthetic bubble mock training set
(see synthetic-bubble-mock-prior memory) -- scales mock_bubble_lightcone.py's
9-example prototype up to a real, resumable dataset.

No py21cmfast needed -- pure geometry (Poisson bubble placement + point-in-
sphere ray tracing of the real, fixed galaxy catalog). Runs anywhere, but
built to be `sbatch`'d for a large N_sim (see generate_bubble_mocks.sh).

Design, mirroring sbi_real_data.py's `_generate_split` (train/val split,
resumable per-batch .npz files, skip-if-already-done -- checked by the
ACTUAL saved array length, not just assuming the file matches, in case an
earlier run used a different --n_sim/--batch_size in the same --output_dir):

Two continuous priors per mock (see mock_bubble_lightcone.py's docstring
for the full reasoning/history): target ionized fraction (IONIZED_FRAC
range, Uniform) and bubble-size-distribution tail weight (K_FACTOR range,
log-Uniform, multiplies the fitted exponnorm K -- confirmed via
compare_bsd_tailweight.py's sweep that this controls cross-galaxy skewer
rigidity in the expected direction).

ONE OPTIMIZATION over the prototype: mean_vol(K) -- needed to convert a
target ionized fraction into a Poisson bubble number density via the
Boolean-model relation f=1-exp(-lambda*E[V]) -- was recomputed via a fresh
300k-sample Monte Carlo for EVERY mock in the prototype. Fine for 9
examples, wasteful at production scale. Precomputed ONCE here as a lookup
table over the K_FACTOR range and interpolated at runtime.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
from scipy.stats import exponnorm

REPO_DEFAULT = os.path.dirname(os.path.abspath(__file__))

K_FIT, LOC, SCALE = 1.787595494643556, 8.57306023571156, 2.438386898231336   # real_data_run.py's R_BUB_DIST_PARAMS
R_LO, R_HI = 0.5, 60.0
Z_END = 5.3   # matches Z_END_DEFAULT in lyabubbles/lightcone_field.py


def load_real_catalog(repo, z_hi):
    spec = importlib.util.spec_from_file_location("real_data_standalone", f"{repo}/lyabubbles/real_data.py")
    rdmod = importlib.util.module_from_spec(spec)
    sys.modules["real_data_standalone"] = rdmod
    spec.loader.exec_module(rdmod)

    cat = rdmod.load_catalog_v2(f"{repo}/tb_lya.txt", f"{repo}/sample_nirspec_properties.txt",
                                z_min=5.0, muv_max=-18.0, prefer="grating")
    in_window = cat.redshift <= z_hi
    ra, dec, redshift = cat.ra[in_window], cat.dec[in_window], cat.redshift[in_window]
    x_gal, y_gal, z_gal, ra0, dec0, z0, x_mean, y_mean, z_mean = rdmod.radec_to_comoving(ra, dec, redshift)
    return dict(ra=ra, dec=dec, redshift=redshift, x_gal=x_gal, y_gal=y_gal, z_gal=z_gal,
                ra0=ra0, dec0=dec0, z0=z0, x_mean=x_mean, y_mean=y_mean, z_mean=z_mean)


def build_geometry(cat, pad_box=1.3):
    x_gal, y_gal, z_gal, z0, z_mean = cat["x_gal"], cat["y_gal"], cat["z_gal"], cat["z0"], cat["z_mean"]
    d_c0 = Cosmo.comoving_distance(z0).to(u.Mpc).value
    d_c_end = Cosmo.comoving_distance(Z_END).to(u.Mpc).value
    z_end_offset = (d_c_end - d_c0) - z_mean

    x_half = pad_box * np.abs(x_gal).max()
    y_half = pad_box * np.abs(y_gal).max()
    z_lo = z_end_offset - 10.0
    z_hi = pad_box * np.abs(z_gal).max()

    pad_r = R_HI   # edge-effect fix, see mock_bubble_lightcone.py
    gx_half, gy_half = x_half + pad_r, y_half + pad_r
    gz_lo, gz_hi = z_lo - pad_r, z_hi + pad_r
    gen_vol = (2 * gx_half) * (2 * gy_half) * (gz_hi - gz_lo)

    return dict(z_end_offset=z_end_offset, x_half=x_half, y_half=y_half, z_lo=z_lo, z_hi=z_hi,
               gx_half=gx_half, gy_half=gy_half, gz_lo=gz_lo, gz_hi=gz_hi, gen_vol=gen_vol)


def draw_radii(n, rng_, K, n_grid=4000):
    """Inverse-CDF sampling via a fine interpolation table, NOT
    exponnorm.ppf directly. Real bug found and fixed: exponnorm.ppf has no
    closed form and falls back to per-element numerical root-finding, which
    is usually fast but occasionally hangs for a very long time on extreme
    tail probabilities (confirmed: f_lo~6e-5 here, near the R_LO=0.5
    truncation bound, triggered a case where just 1000 .ppf calls took
    minutes with no other explanation -- not a general array-size slowdown,
    an unpredictable per-element landmine). exponnorm.cdf, by contrast, is
    closed-form and effectively instant (a 4000-point grid costs ~0.1ms) --
    building the CDF grid once per K and inverting via np.interp is both
    ~1000x faster in the typical case AND removes the unpredictable-hang
    failure mode entirely, which matters far more at production scale
    (thousands of mocks, each drawing potentially thousands of radii) than
    the speedup alone."""
    r_grid = np.linspace(R_LO, R_HI, n_grid)
    cdf_grid = exponnorm.cdf(r_grid, K, LOC, SCALE)
    f_lo, f_hi = cdf_grid[0], cdf_grid[-1]
    u_ = rng_.uniform(size=n)
    return np.interp(f_lo + u_ * (f_hi - f_lo), cdf_grid, r_grid)


def build_mean_vol_lookup(k_factor_lo, k_factor_hi, n_grid=60, n_mc=200_000, seed=12345):
    """mean_vol(K) precomputed on a grid spanning the full K_FACTOR prior
    range, interpolated at runtime -- see module docstring for why."""
    k_grid = K_FIT * np.geomspace(k_factor_lo, k_factor_hi, n_grid)
    mean_vols = np.zeros(n_grid)
    rng = np.random.default_rng(seed)
    for i, K in enumerate(k_grid):
        mean_vols[i] = np.mean(4 / 3 * np.pi * draw_radii(n_mc, rng, K) ** 3)
    return k_grid, mean_vols


def generate_one_mock(rng, x_ion, K, geom, k_grid, mean_vol_grid, x_gal, y_gal, z_gal,
                      z_end_offset, n_gal, n_los):
    mean_vol = np.interp(K, k_grid, mean_vol_grid)
    lam = -np.log(1 - x_ion) / mean_vol
    n_b = rng.poisson(lam * geom["gen_vol"])
    centers = np.column_stack([
        rng.uniform(-geom["gx_half"], geom["gx_half"], n_b),
        rng.uniform(-geom["gy_half"], geom["gy_half"], n_b),
        rng.uniform(geom["gz_lo"], geom["gz_hi"], n_b),
    ])
    radii = draw_radii(n_b, rng, K)

    mask = np.zeros((n_gal, n_los), dtype=np.uint8)
    for i in range(n_gal):
        z_samples = np.linspace(z_end_offset, z_gal[i], n_los)
        pts = np.column_stack([np.full(n_los, x_gal[i]), np.full(n_los, y_gal[i]), z_samples])
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        mask[i] = np.any(d2 <= radii[None, :] ** 2, axis=1)
    return mask, n_b


def generate_split(n_sim, batch_size, seed, output_dir, prefix, geom, k_grid, mean_vol_grid,
                   x_gal, y_gal, z_gal, z_end_offset, n_gal, n_los,
                   ionized_frac_range, k_factor_range):
    rng_master = np.random.default_rng(seed)
    n_done, batch_idx = 0, 0
    while n_done < n_sim:
        this_batch = min(batch_size, n_sim - n_done)
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")

        # ALWAYS draw this batch's random numbers from rng_master, whether or
        # not the batch is actually computed below -- real bug found/fixed
        # here: the previous version only drew when a batch was missing, so
        # on a resumed/extended run (e.g. raising --n_sim in the same
        # --output_dir) every skipped batch left rng_master un-advanced, and
        # the first genuinely NEW batch silently replayed the exact same
        # draws (hence the exact same x_ion/k_factor/masks) as batch 0
        # already on disk. Drawing unconditionally keeps rng_master's stream
        # identical to what a single from-scratch run at the larger --n_sim
        # would have produced, so resuming/extending is actually equivalent
        # to a longer uninterrupted run, not a source of duplicate mocks.
        x_ion_batch = rng_master.uniform(*ionized_frac_range, size=this_batch)
        k_factor_batch = np.exp(rng_master.uniform(np.log(k_factor_range[0]), np.log(k_factor_range[1]),
                                                    size=this_batch))
        mock_seeds = rng_master.integers(0, 2**31 - 1, size=this_batch)

        if os.path.exists(out_path):
            existing_n = len(np.load(out_path)["x_ion"])
            if existing_n != this_batch:
                print(f"[generate:{prefix}] WARNING: {out_path} exists with {existing_n} sims, "
                      f"not the {this_batch} this run expects at batch {batch_idx} -- looks like "
                      f"a run with different --n_sim/--batch_size used this --output_dir before. "
                      f"Using the {existing_n} sims already on disk; delete this file first if you "
                      f"want it regenerated with the new parameters instead.", flush=True)
            print(f"[generate:{prefix}] {out_path} exists ({existing_n} sims), skipping (resumable).",
                  flush=True)
            n_done += existing_n
            batch_idx += 1
            continue

        K_batch = K_FIT * k_factor_batch

        masks = np.zeros((this_batch, n_gal, n_los), dtype=np.uint8)
        n_bubbles = np.zeros(this_batch, dtype=np.int64)
        for j in range(this_batch):
            mrng = np.random.default_rng(mock_seeds[j])
            masks[j], n_bubbles[j] = generate_one_mock(
                mrng, x_ion_batch[j], K_batch[j], geom, k_grid, mean_vol_grid,
                x_gal, y_gal, z_gal, z_end_offset, n_gal, n_los)

        np.savez(out_path, x_ion=x_ion_batch, k_factor=k_factor_batch, K=K_batch,
                n_bubbles=n_bubbles, masks=masks, realized=masks.mean(axis=(1, 2)))
        print(f"[generate:{prefix}] batch {batch_idx} ({this_batch} sims) saved to {out_path} "
              f"-- realized x_ion mean={masks.mean():.3f}", flush=True)
        n_done += this_batch
        batch_idx += 1


def generate_split_v2(n_sim, batch_size, seed, output_dir, prefix, geom, cat, n_los, bsd_calibration,
                      task_id=0, n_tasks=1):
    """As generate_split, for the v2 prior (bubble_prior_v2.py): redshift-dependent ionized fraction, bubble sizes
    calibrated to Lu+24 as a function of x_ion, and optional host bubbles around the galaxies. Same resumability
    rules: every batch's random numbers are drawn from rng_master whether or not the batch is computed.

    Parallel: n_tasks independent processes (a SLURM array) split the batches round-robin, task task_id computing
    the batches with batch_idx % n_tasks == task_id. Every task draws the random numbers of ALL batches from
    rng_master, so a batch's mocks are identical whichever task computes it and however many tasks there are."""
    import bubble_prior_v2 as bp
    bsd = bp.BSD(bsd_calibration)
    los_z = bp.LOSRedshift(cat["z0"], cat["z_mean"])
    x_gal, y_gal, z_gal = cat["x_gal"], cat["y_gal"], cat["z_gal"]
    n_gal = len(x_gal)
    rng_master = np.random.default_rng(seed)
    n_done, batch_idx = 0, 0
    while n_done < n_sim:
        this_batch = min(batch_size, n_sim - n_done)
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")
        hist = np.array([bp.draw_history(rng_master) for _ in range(this_batch)])    # (z_mid, dz, x_hi_7, z_end)
        zeta = rng_master.uniform(*bp.ZETA_RANGE, size=this_batch)
        f_host = rng_master.uniform(*bp.F_HOST_RANGE, size=this_batch)
        mock_seeds = rng_master.integers(0, 2**31 - 1, size=this_batch)

        if batch_idx % n_tasks != task_id:            # another array task's batch: random numbers drawn, skip it
            n_done += this_batch
            batch_idx += 1
            continue
        if os.path.exists(out_path):
            existing_n = len(np.load(out_path)["x_ion"])
            print(f"[generate_v2:{prefix}] {out_path} exists ({existing_n} sims), skipping (resumable).", flush=True)
            n_done += existing_n
            batch_idx += 1
            continue

        masks = np.zeros((this_batch, n_gal, n_los), dtype=np.uint8)
        n_bubbles = np.zeros(this_batch, dtype=np.int64)
        hosted = np.zeros((this_batch, n_gal), dtype=bool)
        realized_z = np.zeros((this_batch, len(bp.Z_SLICES) - 1))
        for j in range(this_batch):
            masks[j], n_bubbles[j], hosted[j], realized_z[j] = bp.generate_one_mock_v2(
                np.random.default_rng(mock_seeds[j]), hist[j, 0], hist[j, 1], zeta[j], f_host[j], bsd, los_z,
                geom, x_gal, y_gal, z_gal, geom["z_end_offset"], n_los)
        # x_ion: ionized fraction of the tanh history at the field-mean redshift z0 (the key add_mock_observations
        # uses to count sims; also the natural "global" parameter to compare with v1's x_ion)
        x_ion = 1 - bp.x_hi_of_z(cat["z0"], hist[:, 0], hist[:, 1])
        tau = np.array([bp.tau_reionization(zm, d) for zm, d in hist[:, :2]])
        np.savez(out_path, x_ion=x_ion, z_mid=hist[:, 0], dz=hist[:, 1], x_hi_7=hist[:, 2], z_end=hist[:, 3],
                 tau_e=tau, zeta=zeta, f_host=f_host,
                 hosted=hosted, n_bubbles=n_bubbles, masks=masks, realized=masks.mean(axis=(1, 2)),
                 realized_z=realized_z, z_slices=bp.Z_SLICES)
        print(f"[generate_v2:{prefix}] batch {batch_idx} ({this_batch} sims) saved to {out_path} "
              f"-- realized ionized fraction mean={masks.mean():.3f}", flush=True)
        n_done += this_batch
        batch_idx += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_sim", type=int, default=15000, help="Training mocks.")
    p.add_argument("--n_sim_val", type=int, default=None,
                   help="Held-out validation mocks (default: 20%% of --n_sim).")
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_los", type=int, default=75)
    p.add_argument("--ionized_frac_lo", type=float, default=0.1)
    p.add_argument("--ionized_frac_hi", type=float, default=0.9)
    p.add_argument("--k_factor_lo", type=float, default=0.3)
    p.add_argument("--k_factor_hi", type=float, default=3.0)
    p.add_argument("--z_hi", type=float, default=7.3, help="Real-catalog redshift window upper edge.")
    p.add_argument("--prior", type=str, default="v1", choices=["v1", "v2"],
                   help="v1: one x_ion and BSD per mock (default, unchanged). v2: bubble_prior_v2.py.")
    p.add_argument("--bsd_calibration", type=str, default=None,
                   help="v2 only: bsd_lu24_calibration.npz from calibrate_bsd_lu24.py.")
    p.add_argument("--task_id", type=int, default=0,
                   help="v2 only: this process's index among --n_tasks (e.g. $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--n_tasks", type=int, default=1,
                   help="v2 only: number of parallel processes splitting the batches round-robin.")
    p.add_argument("--repo", type=str, default=REPO_DEFAULT,
                   help="Path to the Lyman-alpha-bubbles repo (for tb_lya.txt/sample_nirspec_properties.txt).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cat = load_real_catalog(args.repo, args.z_hi)
    n_gal = len(cat["x_gal"])
    geom = build_geometry(cat)
    print(f"[setup] {n_gal} real galaxies, z<={args.z_hi}. Evaluation box: "
          f"x_half={geom['x_half']:.2f} y_half={geom['y_half']:.2f} Mpc, "
          f"z in [{geom['z_lo']:.2f},{geom['z_hi']:.2f}] Mpc", flush=True)

    n_sim_val = args.n_sim_val if args.n_sim_val is not None else max(1, args.n_sim // 5)
    if args.prior == "v2":
        import bubble_prior_v2 as bp
        if args.bsd_calibration is None:
            raise ValueError("--prior v2 needs --bsd_calibration")
        if not 0 <= args.task_id < args.n_tasks:
            raise ValueError(f"--task_id {args.task_id} must be in [0, --n_tasks {args.n_tasks})")
        if args.task_id == 0:                         # one writer, so parallel tasks do not race on the file
            with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
                json.dump(dict(prior="v2", n_gal=n_gal, n_los=args.n_los, z_hi=args.z_hi,
                               bsd_calibration=os.path.abspath(args.bsd_calibration), z_ref=bp.Z_REF,
                               x_hi_7_range=bp.X_HI_7_RANGE, z_end_range=bp.Z_END_RANGE, x_hi_end=bp.X_HI_END,
                               zeta_range=bp.ZETA_RANGE,
                               f_host_range=bp.F_HOST_RANGE, x_ion_cap=bp.X_ION_CAP, slab=bp.SLAB,
                               n_sim=args.n_sim, n_sim_val=n_sim_val, batch_size=args.batch_size,
                               seed=args.seed, n_tasks=args.n_tasks), f, indent=2)
        for n, s, pre in ((args.n_sim, args.seed, "train"), (n_sim_val, args.seed + 1_000_000, "val")):
            generate_split_v2(n, args.batch_size, s, args.output_dir, pre, geom, cat, args.n_los,
                              args.bsd_calibration, task_id=args.task_id, n_tasks=args.n_tasks)
        print(f"[done] v2 prior, task {args.task_id}/{args.n_tasks}: its share of {args.n_sim} train + "
              f"{n_sim_val} val mocks in {args.output_dir}", flush=True)
        return

    k_grid, mean_vol_grid = build_mean_vol_lookup(args.k_factor_lo, args.k_factor_hi)
    print(f"[setup] mean_vol(K) lookup table built, K in [{k_grid.min():.3f},{k_grid.max():.3f}]", flush=True)

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(dict(n_gal=n_gal, n_los=args.n_los, z_hi=args.z_hi,
                       ionized_frac_range=[args.ionized_frac_lo, args.ionized_frac_hi],
                       k_factor_range=[args.k_factor_lo, args.k_factor_hi],
                       K_fit=K_FIT, loc=LOC, scale=SCALE, r_lo=R_LO, r_hi=R_HI,
                       n_sim=args.n_sim, batch_size=args.batch_size, seed=args.seed), f, indent=2)

    n_sim_val = args.n_sim_val if args.n_sim_val is not None else max(1, args.n_sim // 5)
    common = dict(geom=geom, k_grid=k_grid, mean_vol_grid=mean_vol_grid,
                 x_gal=cat["x_gal"], y_gal=cat["y_gal"], z_gal=cat["z_gal"],
                 z_end_offset=geom["z_end_offset"], n_gal=n_gal, n_los=args.n_los,
                 ionized_frac_range=(args.ionized_frac_lo, args.ionized_frac_hi),
                 k_factor_range=(args.k_factor_lo, args.k_factor_hi))

    generate_split(args.n_sim, args.batch_size, args.seed, args.output_dir, "train", **common)
    generate_split(n_sim_val, args.batch_size, args.seed + 1_000_000, args.output_dir, "val", **common)
    print(f"[done] {args.n_sim} train + {n_sim_val} val mocks in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
