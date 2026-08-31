"""
Add the noisy observational `x` to the synthetic bubble mock dataset
(`generate_bubble_mocks.py`'s output -- theta-only, binary per-galaxy LOS
masks), turning it into real `(theta, x)` training pairs for
`sbi_pixel_field.py`'s NRE. Output files are written in EXACTLY
`sbi_pixel_field.py`'s own `simulate`-output schema (`theta`, `x`, `n_gal`,
`n_los` in one `.npz` per batch), so `sbi_pixel_field.load_sims`/
`run_train_nre`/`run_infer` can consume them completely unmodified -- this
script's whole job is producing a drop-in-compatible sims_dir from a
synthetic (not 21cmFAST) theta source.

Requires the full py21cmfast-dependent stack (`real_data_run.py` ->
`speed_up.py` -> `galaxy_prop.py`), same as `sbi_pixel_field.py simulate`
itself -- CANNOT run in the lightweight local env used for the mock
generator/diagnostics; run this on the cluster.

Two real conventions had to be reconciled between the mock generator and
`sbi_pixel_field.py`'s theta convention (verified numerically, not assumed --
see the session's derivation before this file was written):

1. Ionized vs. neutral: the mock's `masks` array (from
   `generate_bubble_mocks.py`) is 1=IONIZED (bubble hit). The physics here
   (`lf.segments_to_tau`) wants `x_HI`, i.e. 1=NEUTRAL. Flipped via `1 -
   masks` below.
2. LOS bin order: the mock's `masks[..., k]` axis runs index 0 = nearest
   z_end (observer side) -> index n_los-1 = nearest the source (built via
   `np.linspace(z_end_offset, z_gal[i], n_los)`, see
   `generate_bubble_mocks.py::generate_one_mock`). `lf.bin_z_edges` (and
   therefore `lf.segments_to_tau`, and `sbi_pixel_field.py`'s own theta
   convention) is the OPPOSITE: index 0 = nearest the source, index
   n_los-1 = nearest z_end. Confirmed numerically (bin_z_edges gave
   z_b[0]=6.909 matching the galaxy's own z=6.908, z_e[-1]=5.300=z_end).
   Flipped via `[..., ::-1]` below -- getting this backwards would silently
   pair each LOS bin's ionization state with the wrong redshift, corrupting
   every tau/EW prediction without any obvious symptom.

Output theta is written in `sbi_pixel_field.py`'s OWN convention (1=neutral,
index 0=source) -- i.e. the flip/invert happens ONCE, here, and everything
downstream (training, inference, the real x_obs comparison) never needs to
know the mock generator used a different bookkeeping convention internally.

Resumable per output batch file, and driven by whatever
`{prefix}_batch_*.npz` files currently exist in `--mocks_dir` (NOT a
`--n_sim` you have to keep in sync) -- safe to run now against a partial
mock dataset and re-run later to top up new batches after
`generate_bubble_mocks.py` is scaled up, with no wasted recomputation on
already-done batches either time.

Usage
-----
python add_mock_observations.py --mocks_dir bubble_mocks_run1 \\
    --output_dir bubble_mocks_run1_x --ew_model exponential
"""
import argparse
import glob
import os
import re

import numpy as np

import real_data_run as rdr
from lyabubbles import lightcone_field as lf
from sbi_real_data import _build_x, _per_galaxy_sigma
from sbi_pixel_field import _add_catalog_args, _precompute_bin_z_edges


def _mock_theta_to_canonical(masks):
    """`(this_batch, n_gal, n_los)` mock masks (1=ionized, index0=z_end-side)
    -> `x_HI` in `sbi_pixel_field.py`'s own convention (1=neutral,
    index0=source-side). See module docstring for why both flips are needed."""
    return 1.0 - masks[:, :, ::-1]


def _compute_tau_and_x_for_batch(masks, meta, z_b_edges, z_e_edges, n_los,
                                 main_dir, ew_model, rng):
    """One mock batch's `masks` -> `(theta_out, x_out)` in
    `sbi_pixel_field.py`'s schema. Mirrors
    `sbi_pixel_field._compute_pixel_theta_and_tau_batch` +
    `_generate_split`'s per-sim x-building, with the (already precomputed)
    mock theta swapped in for the real-field ray trace."""
    s = rdr._S
    n_gal = len(s.x_gal)
    this_batch = masks.shape[0]
    wave_em_vals = rdr.wave_em.value

    x_hi = _mock_theta_to_canonical(masks)              # (this_batch, n_gal, n_los)
    theta_batch = np.ascontiguousarray(x_hi.transpose(1, 0, 2))   # (n_gal, this_batch, n_los)

    tau_batch = np.empty((n_gal, this_batch, len(wave_em_vals)), dtype=np.float64)
    for g in range(n_gal):
        for k in range(this_batch):
            tau_batch[g, k, :] = lf.segments_to_tau(
                z_b_edges[g], z_e_edges[g], theta_batch[g, k, :], s.redshifts[g], wave_em_vals,
            )

    rdr._refresh_mc_state(meta['muv'], meta['redshifts'], meta['x_gal'], meta['y_gal'],
                          meta['z_gal'], meta['beta'], meta['z0'], this_batch, main_dir,
                          ew_model=ew_model)
    rdr._S.pixel_tau = tau_batch

    sigma = _per_galaxy_sigma()
    x_dim = 2 * n_gal
    x_out = np.empty((this_batch, x_dim), dtype=np.float32)
    for k in range(this_batch):
        # Same trapz/exp(-tau) combination sbi_pixel_field._ew_pred_from_pixel_tau
        # uses -- inlined rather than imported since that helper reads
        # rdr._S.pixel_tau via module-global state exactly as set two lines
        # above, so calling it here is equivalent, just spelled out for clarity.
        tau_now  = rdr._S.pixel_tau[:, k, :]
        j_s_k    = rdr._S.j_s[:, k, :]
        weighted = j_s_k * rdr._S.tau_cgm * np.exp(-tau_now)
        numerator = np.trapz(weighted, wave_em_vals, axis=1)
        t_in = numerator / rdr._S.j_s_trapz_denom[:, k]
        ew_pred = rdr._S.ew_int[:, k] * t_in
        x_out[k] = _build_x(ew_pred, sigma, rng)

    theta_out = theta_batch.transpose(1, 0, 2).reshape(this_batch, n_gal * n_los)
    return theta_out, x_out


def process_split(meta, mocks_dir, output_dir, prefix, n_los, z_b_edges, z_e_edges,
                  main_dir, ew_model, seed):
    paths = sorted(glob.glob(os.path.join(mocks_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        print(f"[add_x:{prefix}] no {prefix}_batch_*.npz files found in {mocks_dir}, skipping.",
              flush=True)
        return
    os.makedirs(output_dir, exist_ok=True)
    rng_master = np.random.default_rng(seed)

    for path in paths:
        batch_idx = int(re.search(r"_batch_(\d+)\.npz$", path).group(1))
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")
        mock = np.load(path)
        this_batch = len(mock["x_ion"])

        # Advance rng_master identically whether we skip or compute, same
        # reasoning as generate_bubble_mocks.py's own resumability fix --
        # keeps re-running this script over a growing --mocks_dir equivalent
        # to one long run, not a source of duplicate/misaligned noise draws.
        batch_rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        if os.path.exists(out_path):
            existing_n = len(np.load(out_path)["theta"])
            if existing_n != this_batch:
                print(f"[add_x:{prefix}] WARNING: {out_path} exists with {existing_n} sims, "
                      f"not the {this_batch} in the source mock batch {path} -- delete it first "
                      f"if you want it rebuilt.", flush=True)
            print(f"[add_x:{prefix}] {out_path} exists ({existing_n} sims), skipping (resumable).",
                  flush=True)
            continue

        theta_out, x_out = _compute_tau_and_x_for_batch(
            mock["masks"], meta, z_b_edges, z_e_edges, n_los, main_dir, ew_model, batch_rng)

        n_gal = mock["masks"].shape[1]
        np.savez(out_path, theta=theta_out, x=x_out, n_gal=n_gal, n_los=n_los)
        print(f"[add_x:{prefix}] batch {batch_idx}: {this_batch} sims -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_catalog_args(p)
    p.add_argument("--mocks_dir", type=str, required=True,
                   help="--output_dir from a prior generate_bubble_mocks.py run.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the noise-injection RNG only (theta itself is already fixed by "
                        "the mock files) -- independent of generate_bubble_mocks.py's --seed.")
    args = p.parse_args()

    meta = rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    z_b_edges, z_e_edges = _precompute_bin_z_edges(meta, args.n_los)
    print(f"[add_x] {len(meta['x_gal'])} galaxies, n_los={args.n_los}, ew_model={args.ew_model}",
          flush=True)

    # Sanity check up front, not silently: the mock files' n_gal/n_los must
    # match what --lya_catalog/--properties_catalog/--z_hi/etc. (the SAME
    # filters generate_bubble_mocks.py used) actually produce here -- a
    # mismatch would otherwise misalign every galaxy's ionization mask with
    # the wrong redshift/EW-prior entry, silently.
    any_mock = sorted(glob.glob(os.path.join(args.mocks_dir, "train_batch_*.npz")) or
                      glob.glob(os.path.join(args.mocks_dir, "val_batch_*.npz")))
    if any_mock:
        probe = np.load(any_mock[0])
        mock_n_gal, mock_n_los = probe["masks"].shape[1], probe["masks"].shape[2]
        if mock_n_gal != len(meta['x_gal']) or mock_n_los != args.n_los:
            raise ValueError(
                f"Mock/catalog mismatch: {any_mock[0]} has n_gal={mock_n_gal}, n_los={mock_n_los}, "
                f"but this run's catalog args give n_gal={len(meta['x_gal'])} and --n_los={args.n_los}. "
                f"Check --lya_catalog/--properties_catalog/--z_hi/--z_min/--muv_max/--prefer match "
                f"what generate_bubble_mocks.py used.")

    process_split(meta, args.mocks_dir, args.output_dir, "train", args.n_los,
                  z_b_edges, z_e_edges, args.main_dir, args.ew_model, args.seed)
    process_split(meta, args.mocks_dir, args.output_dir, "val", args.n_los,
                  z_b_edges, z_e_edges, args.main_dir, args.ew_model, args.seed + 1_000_000)
    print(f"[add_x] done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()