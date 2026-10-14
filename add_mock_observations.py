"""
add_mock_observations.py -- turn synthetic ionization maps into (theta, x) training pairs.

WHAT THIS SCRIPT DOES, IN ONE PARAGRAPH
---------------------------------------
`generate_bubble_mocks.py` produces synthetic ionization maps ("mocks"): for every
galaxy of the real GOODS-N catalog, a binary ionized/neutral mask sampled along its
sightline, from the galaxy down to z_end = 5.3. These masks are the parameter `theta`
of the pixel-field SBI. To train a neural ratio estimator we also need, for every
mock, a simulated observation `x` -- what the 43 measured Lya equivalent widths (EWs)
would look like if the Universe had exactly that ionization map. This script computes
that `x` with the same forward model used everywhere else in the pipeline, and writes
`(theta, x)` pairs in the file format `sbi_pixel_field.py train_nre/infer` reads.

THE FORWARD MODEL, STEP BY STEP (for one mock, one galaxy g)
------------------------------------------------------------
 1. IGM optical depth from the mask. Every neutral sightline bin is a segment of fully
    neutral gas; its Lya damping-wing optical depth tau(lambda) on the rest-frame
    wavelength grid 1214-1225 A (= -412 ... +2301 km/s around Lya) is the analytic
    Miralda-Escude (1998) solution, summed over all neutral segments
    (`lf.segments_to_tau`; see `_compute_tau_and_x_for_batch` for the equation).
 2. Galaxy properties, drawn at random for every (mock, galaxy) pair
    (`rdr._refresh_mc_state`), given the galaxy's catalog Muv:
      - intrinsic (emergent) Lya EW, from the distribution chosen with `--ew_model`
        (default: Mason et al. 2018 exponential model; alternatives Tang et al. 2024,
        Gagnon-Hartman et al. 2026 -- see lyabubbles/galaxy_prop.py::p_EW);
      - Lya line profile J(lambda): a Gaussian in velocity, centred on a velocity
        offset dv drawn from the Mason et al. 2018 dv(Muv) relation;
      - CGM transmission: a step function, 0 blueward and 1 redward of the halo
        circular velocity (Mason et al. 2018 model; deterministic given Muv).
 3. IGM transmission  T = Int J * T_CGM * exp(-tau) dlambda / Int J dlambda,
    and the observable EW_obs = EW_intrinsic * T.
 4. Observation model (`sbi_real_data._build_x`): Gaussian noise with the galaxy's own
    catalog noise level sigma_g, then a 3-sigma detection cut. Non-detections are
    reported as the 3-sigma upper-limit value, exactly like the real catalog.
 5. x = [43 reported EWs, 43 upper-limit flags]  (86 numbers).

Only step 1 depends on theta. Steps 2-4 are the same stochastic model as in the
21cmFAST-based simulator (`sbi_pixel_field.py simulate`), so the two training sets
differ ONLY in where theta comes from.

Requires the full py21cmfast-dependent stack (`real_data_run.py` -> `speed_up.py` ->
`galaxy_prop.py`), unlike `generate_bubble_mocks.py`. Run it on the cluster.

CONVENTIONS THAT MUST BE RECONCILED (verified numerically when the script was written)
--------------------------------------------------------------------------------------
1. Ionized vs neutral. `generate_bubble_mocks.py` stores masks with 1 = IONIZED (the
   sample point lies inside a bubble). The physics (`lf.segments_to_tau`) and the SBI
   theta convention use the neutral fraction x_HI, i.e. 1 = NEUTRAL. Flipped with
   `1 - masks`.
2. Order along the sightline. In the mock masks, index 0 is the sample point at z_end
   (observer side) and index n_los-1 is the point at the galaxy (masks are sampled at
   `np.linspace(z_end_offset, z_gal[i], n_los)` in `generate_bubble_mocks.py`).
   `lf.bin_z_edges`, `lf.segments_to_tau` and the SBI theta all use the OPPOSITE
   order: index 0 next to the galaxy, index n_los-1 next to z_end (checked:
   z_b[0] = 6.909 equals that galaxy's z = 6.908; z_e[-1] = 5.300 = z_end).
   Flipped with `[..., ::-1]`. Getting this backwards would pair each bin with the
   wrong redshift and corrupt every tau without any visible symptom.
The flips are done ONCE, here; the written theta is in the SBI convention (1 = neutral,
index 0 = galaxy side), so training, inference and the real x_obs never see the mock
generator's internal convention.

KNOWN APPROXIMATIONS (documented, not hidden)
---------------------------------------------
a. Point sample vs bin. The mock stores the ionization state at n_los POINTS spaced
   L/(n_los-1) apart (L = path length from the galaxy to z_end), but tau is computed
   for n_los BINS of width L/n_los. After the flip, point k sits at k*L/(n_los-1) from
   the galaxy, which always lies inside bin k = [k*L/n_los, (k+1)*L/n_los] (at the
   galaxy-side edge for small k, drifting towards the far edge for large k). The whole
   bin is assigned the state of that single point.
b. Absolute distance offset. `generate_bubble_mocks.py` places galaxy g at comoving
   distance d_C(z0) + z_gal[g] + z_mean, whereas `lf.bin_z_edges` starts the sightline
   at d_C(z0) + z_gal[g] (no z_mean). z_mean = mean(d_C(z_i)) - d_C(mean z_i) is a
   non-linearity offset of -0.25 cMpc for the 43-galaxy GOODS-N sample, i.e. 2.6% of a
   ~9.8 cMpc bin (z shift ~0.0008). Negligible, but the two files are not identical.
c. Reproducibility. `--seed` controls only the observational noise (step 4). The EW,
   velocity-offset and line-profile draws (step 2) use numpy's global random state
   inside `get_content`/`p_EW`/`get_js` (run in joblib worker processes), which is not
   seeded here: rerunning the script gives statistically equivalent but not identical x.

References
----------
Miralda-Escude J., 1998, ApJ, 501, 15        -- damping-wing optical depth, function I(x)
Gunn J. E., Peterson B. A., 1965, ApJ, 142, 1633 -- Gunn-Peterson optical depth
Mason C. A. et al., 2018, ApJ, 856, 2         -- EW(Muv) model, dv(Muv) relation, CGM model
Tang M. et al., 2024, arXiv:2402.06070 / 2408.01507 -- log-normal EW distributions
Gagnon-Hartman S. et al., 2026, arXiv:2602.13389    -- joint L_Lya / dv model
Oke J. B., Gunn J. E., 1983, ApJ, 266, 713    -- AB magnitudes (M_UV <-> L_nu)
Planck Collaboration, 2020, A&A, 641, A6      -- cosmology (astropy Planck18)

Usage
-----
python add_mock_observations.py --mocks_dir bubble_mocks_run1 \\
    --output_dir bubble_mocks_run1_x --ew_model exponential
"""
# --- standard library ---
import argparse   # command-line interface
import glob       # find the mock batch files on disk
import json       # read the mock directory's run_config.json (completeness check)
import os         # path handling, directory creation
import re         # parse the batch index out of a file name

# --- third party ---
import numpy as np

# --- this repository ---
# real_data_run: catalog loading (`_load_catalog_and_priors`), the stochastic galaxy
# model (`_refresh_mc_state`), the rest-frame wavelength grid (`wave_em`) and the
# module-level state object `_S` in which both store their results.
import real_data_run as rdr
# lightcone_field: sightline binning (`bin_z_edges`) and the damping-wing optical depth
# of a sequence of neutral/ionized segments (`segments_to_tau`).
from lyabubbles import lightcone_field as lf
# sbi_real_data: the observation model. `_per_galaxy_sigma` gives each galaxy's noise
# level from the catalog; `_build_x` adds noise, applies the 3-sigma cut and packs x.
from sbi_real_data import _build_x, _per_galaxy_sigma
# sbi_pixel_field: the shared CLI arguments (catalog paths, z-window, n_los, ew_model)
# and the per-galaxy redshift edges of the n_los sightline bins.
from sbi_pixel_field import _add_catalog_args, _precompute_bin_z_edges


def _mock_theta_to_canonical(masks):
    """Convert mock masks to the SBI theta convention (see module docstring,
    "CONVENTIONS"). Input: (this_batch, n_gal, n_los) uint8 masks with 1 = ionized and
    index 0 at z_end. Output: same shape, float, x_HI with 1 = neutral and index 0 at
    the galaxy."""
    # masks[:, :, ::-1] reverses the sightline axis (z_end-first -> galaxy-first);
    # 1.0 - (...) turns "ionized" (1) into "neutral" (1), and also casts uint8 to float.
    return 1.0 - masks[:, :, ::-1]


def _compute_tau_and_x_for_batch(masks, meta, z_b_edges, z_e_edges, n_los,
                                 main_dir, ew_model, rng):
    """Forward-model one batch of mocks.

    Inputs
      masks      (this_batch, n_gal, n_los) uint8, mock convention (1 = ionized, index 0 at z_end)
      meta       dict returned by rdr._load_catalog_and_priors (catalog Muv, z, positions, ...)
      z_b_edges, z_e_edges  (n_gal, n_los) redshifts of the near (b = "begin", galaxy side)
                 and far (e = "end", observer side) edge of every sightline bin
      n_los      number of sightline bins per galaxy
      main_dir   repository root (data tables for the CGM model are read from it)
      ew_model   intrinsic Lya EW distribution (see real_data_run.EW_MODEL_CHOICES)
      rng        numpy Generator for the observational noise of this batch
    Returns
      theta_out  (this_batch, n_gal * n_los) x_HI maps, flattened galaxy-major
      x_out      (this_batch, 2 * n_gal) simulated observations

    Mirrors what `sbi_pixel_field.py simulate` does per simulation, with the mock theta
    taking the place of the 21cmFAST ray trace."""
    s = rdr._S                                   # shared state filled by _load_catalog_and_priors
    n_gal = len(s.x_gal)                         # number of galaxies in the catalog window (43)
    this_batch = masks.shape[0]                  # number of mocks in this batch (normally 2000)
    wave_em_vals = rdr.wave_em.value             # rest-frame wavelength grid [A]: 100 points, 1214-1225 A

    # theta in the SBI convention, shape (this_batch, n_gal, n_los)
    x_hi = _mock_theta_to_canonical(masks)              # (this_batch, n_gal, n_los)
    # Same array reordered to (n_gal, this_batch, n_los) so that the inner loop below runs over
    # mocks for a fixed galaxy; ascontiguousarray makes each [g, k, :] row contiguous in memory.
    theta_batch = np.ascontiguousarray(x_hi.transpose(1, 0, 2))   # (n_gal, this_batch, n_los)

    # ---- Step 1: IGM damping-wing optical depth of every (galaxy, mock) sightline ----
    # tau_batch[g, k, :] = tau(lambda) on the 100-point wavelength grid.
    tau_batch = np.empty((n_gal, this_batch, len(wave_em_vals)), dtype=np.float64)
    for g in range(n_gal):
        for k in range(this_batch):
            # Each sightline bin j is a segment from z_b[j] (galaxy side) to z_e[j] (observer
            # side) with neutral fraction x_HI,j in {0, 1}. For a source at z_s and rest-frame
            # wavelength lambda, the photon redshifts into Lya resonance at
            # 1 + z_lambda = (lambda / 1215.67 A) (1 + z_s). segments_to_tau returns
            #   tau(lambda) = (tau_GP R_alpha / pi) * sum_j x_HI,j * r_b,j^1.5 * [I(r_b,j) - I(r_e,j)],
            #   r_{b,e} = (1 + z_{b,e}) / (1 + z_lambda),
            #   tau_GP = 7.16e5 ((1 + z_s)/10)^1.5   (Gunn & Peterson 1965 optical depth of a fully
            #            neutral IGM at mean density, evaluated at the source redshift),
            #   R_alpha = Lambda_alpha / (4 pi nu_alpha), Lambda_alpha = 6.25e8 s^-1,
            #   I(x) = x^4.5/(1-x) + 9/7 x^3.5 + 9/5 x^2.5 + 3 x^1.5 + 9 x^0.5
            #          - ln|(1 + x^0.5)/(1 - x^0.5)|        (Miralda-Escude 1998).
            # I(x) has a pole at x = 1 (a wavelength resonating exactly at a segment edge); the
            # resulting negative per-segment terms are numerical artefacts and are clipped to 0
            # inside segments_to_tau. Ionized bins contribute nothing (x_HI = 0). There is no gas
            # below z_end = 5.3 in this model.
            tau_batch[g, k, :] = lf.segments_to_tau(
                z_b_edges[g], z_e_edges[g], theta_batch[g, k, :], s.redshifts[g], wave_em_vals,
            )

    # ---- Step 2: stochastic galaxy properties, one independent draw per (galaxy, mock) ----
    # `_refresh_mc_state(..., n_inside_tau=this_batch)` draws `this_batch` realisations per galaxy,
    # so draw k is paired with mock k. Through get_content (lyabubbles/speed_up.py) it fills:
    #   rdr._S.j_s[g, k, :]          Lya line profile on the wavelength grid: a Gaussian in velocity,
    #                                centre dv and width sigma_v = dv (fwhm_true=False), normalised
    #                                to unit integral over the grid. log10(dv / km s^-1) ~
    #                                N(0.32 gamma (Muv + 20 + 0.26 z) + 2.34, 0.24), gamma = -0.3 for
    #                                Muv >= -20 - 0.26 z, else -0.7 (Mason et al. 2018). NOTE: z here
    #                                is the field-mean redshift z0, not each galaxy's own z.
    #   rdr._S.j_s_trapz_denom[g, k] Int J dlambda (= 1 by construction; kept for clarity).
    #   rdr._S.ew_int[g, k]          intrinsic Lya EW [A], drawn from the `ew_model` distribution
    #                                given Muv (lyabubbles/galaxy_prop.py::p_EW). Recovered exactly
    #                                from the drawn Lya luminosity, L_Lya = EW * C * L_UV, with
    #                                L_UV,nu = 10^(-0.4 (Muv - 51.6)) erg/s/Hz (AB system, Oke & Gunn
    #                                1983) and C = (c / 1216 A^2) (1500/1216)^(-beta-2) converting
    #                                L_nu at 1500 A to L_lambda at 1216 A for a UV slope beta = -2.
    #   rdr._S.tau_cgm[g, :]         CGM TRANSMISSION (despite the name: values are 0 or 1), set in
    #                                _load_catalog_and_priors: 1 for velocities >= v_c, 0 below, with
    #                                v_c = (10 G M_h H(z=7.5))^(1/3) the halo circular velocity and
    #                                M_h interpolated from a Muv -> M_h table (lyabubbles/data/
    #                                Muv.npy, mh.npy) (Mason et al. 2018). Deterministic given Muv.
    # The same call also draws "outside bubble" sightline geometry (get_bubbles /
    # calculate_taus_prep) used by the parametric-bubble likelihood. Those fields (tau_prec,
    # ew_pred_outside, t_outside, ...) are NOT used here: in this script the IGM is entirely
    # described by theta (step 1). They cost compute but do not affect x.
    rdr._refresh_mc_state(meta['muv'], meta['redshifts'], meta['x_gal'], meta['y_gal'],
                          meta['z_gal'], meta['beta'], meta['z0'], this_batch, main_dir,
                          ew_model=ew_model)
    # Store the theta-dependent tau next to the galaxy draws, in the same place the 21cmFAST
    # simulator stores it (sbi_pixel_field._ew_pred_from_pixel_tau reads `_S.pixel_tau`); here it is
    # only read back a few lines below.
    rdr._S.pixel_tau = tau_batch

    # ---- Step 4 inputs: per-galaxy noise level from the real catalog ----
    # sigma_g = reported 1-sigma EW error for detected galaxies, or (reported 3-sigma upper
    # limit) / 3 for non-detections. A fixed property of the real observation of galaxy g.
    sigma = _per_galaxy_sigma()
    x_dim = 2 * n_gal                            # x = [n_gal EWs, n_gal upper-limit flags]
    x_out = np.empty((this_batch, x_dim), dtype=np.float32)
    for k in range(this_batch):
        # ---- Step 3: transmission and observed EW for mock k, all galaxies at once ----
        # (The same combination as sbi_pixel_field._ew_pred_from_pixel_tau, written out here.)
        tau_now  = rdr._S.pixel_tau[:, k, :]          # (n_gal, n_wave) IGM optical depth of mock k
        j_s_k    = rdr._S.j_s[:, k, :]                # (n_gal, n_wave) line profile, draw k
        # integrand J(lambda) * T_CGM(lambda) * exp(-tau_IGM(lambda)); tau_cgm broadcasts over k
        weighted = j_s_k * rdr._S.tau_cgm * np.exp(-tau_now)
        # Int over wavelength (trapezoidal rule on the 100-point grid), one value per galaxy
        numerator = np.trapz(weighted, wave_em_vals, axis=1)
        # T = Int J T_CGM e^-tau dlambda / Int J dlambda: fraction of the emitted Lya line that
        # reaches the observer (also the model's prediction for the Lya escape fraction)
        t_in = numerator / rdr._S.j_s_trapz_denom[:, k]
        # observed EW = intrinsic EW x transmission (the UV continuum is not attenuated)
        ew_pred = rdr._S.ew_int[:, k] * t_in
        # Guard: a non-finite EW would silently become a 3-sigma upper limit in _build_x (NaN
        # comparisons are False), making x identical for every mock -- the network then learns
        # nothing while SBC still passes. This happened with the Tang EW models before the
        # p_Tang W=0 fix (2026-10-09), so fail loudly instead.
        if not np.all(np.isfinite(ew_pred)):
            raise FloatingPointError(
                f"non-finite predicted EW for galaxies {np.where(~np.isfinite(ew_pred))[0].tolist()} "
                f"(ew_model={ew_model}); check the intrinsic EW sampler.")
        # ---- Step 4: observation model ----
        # EW_sim = ew_pred + N(0, sigma_g); detected if EW_sim > 3 sigma_g, in which case EW_sim is
        # reported, otherwise the 3 sigma_g upper-limit value is reported with flag 1.
        # x_out[k] = [reported EWs (n_gal), upper-limit flags (n_gal)].
        x_out[k] = _build_x(ew_pred, sigma, rng)

    # Back to (this_batch, n_gal, n_los), then flatten each map galaxy-major
    # (element g * n_los + j = bin j of galaxy g), the layout sbi_pixel_field.load_sims expects.
    theta_out = theta_batch.transpose(1, 0, 2).reshape(this_batch, n_gal * n_los)
    return theta_out, x_out


def process_split(meta, mocks_dir, output_dir, prefix, n_los, z_b_edges, z_e_edges,
                  main_dir, ew_model, seed, task_id=0, n_tasks=1):
    """Forward-model every mock batch file `{prefix}_batch_*.npz` ("train" or "val") found in
    `mocks_dir` and write one output file per batch to `output_dir`.

    Resumable: a batch whose output file already exists is skipped. The noise seed of each
    batch is drawn from one master generator in a fixed order whether or not the batch is
    skipped, so a resumed run assigns every batch the same noise seed as an uninterrupted
    run (but see module docstring, approximation c: the galaxy draws are not seeded).

    Parallel: n_tasks independent processes (a SLURM array) split the batches round-robin, task
    task_id processing the batches with batch_idx % n_tasks == task_id. Every task still draws
    the noise seeds of ALL batches, so a batch gets the same noise seed whichever task runs it."""
    # all input batch files of this split, sorted so batch order is deterministic
    paths = sorted(glob.glob(os.path.join(mocks_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        print(f"[add_x:{prefix}] no {prefix}_batch_*.npz files found in {mocks_dir}, skipping.",
              flush=True)
        return
    os.makedirs(output_dir, exist_ok=True)            # create the output directory if needed
    rng_master = np.random.default_rng(seed)          # master generator: one noise seed per batch

    for path in paths:
        # batch index from the file name, e.g. ".../train_batch_00003.npz" -> 3
        batch_idx = int(re.search(r"_batch_(\d+)\.npz$", path).group(1))
        # output keeps the same split name and batch index
        out_path = os.path.join(output_dir, f"{prefix}_batch_{batch_idx:05d}.npz")
        mock = np.load(path)                          # keys: masks, x_ion, k_factor, K, n_bubbles, realized
        this_batch = len(mock["x_ion"])               # number of mocks in this batch

        # Draw this batch's noise seed BEFORE deciding whether to skip, so the master generator
        # advances identically in a fresh and in a resumed run (same reasoning as the
        # resumability logic of generate_bubble_mocks.py).
        batch_rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        if batch_idx % n_tasks != task_id:            # another array task's batch: seed drawn, skip it
            continue
        if os.path.exists(out_path):
            # Already done: check the stored size matches the input, warn if not (an output from a
            # run with different batch sizes would otherwise be silently reused).
            existing_n = len(np.load(out_path)["theta"])
            if existing_n != this_batch:
                print(f"[add_x:{prefix}] WARNING: {out_path} exists with {existing_n} sims, "
                      f"not the {this_batch} in the source mock batch {path} -- delete it first "
                      f"if you want it rebuilt.", flush=True)
            print(f"[add_x:{prefix}] {out_path} exists ({existing_n} sims), skipping (resumable).",
                  flush=True)
            continue

        # forward model the whole batch (steps 1-5 of the module docstring)
        theta_out, x_out = _compute_tau_and_x_for_batch(
            mock["masks"], meta, z_b_edges, z_e_edges, n_los, main_dir, ew_model, batch_rng)

        n_gal = mock["masks"].shape[1]                # galaxies per mock (43)
        # Same schema as `sbi_pixel_field.py simulate` output: theta, x, n_gal, n_los
        np.savez(out_path, theta=theta_out, x=x_out, n_gal=n_gal, n_los=n_los)
        print(f"[add_x:{prefix}] batch {batch_idx}: {this_batch} sims -> {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # catalog paths, z-window, Muv cut, n_los, ew_model, main_dir (shared with sbi_pixel_field.py)
    _add_catalog_args(p)
    p.add_argument("--mocks_dir", type=str, required=True,
                   help="--output_dir from a prior generate_bubble_mocks.py run.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds the noise-injection RNG only (theta itself is already fixed by "
                        "the mock files) -- independent of generate_bubble_mocks.py's --seed.")
    p.add_argument("--task_id", type=int, default=0,
                   help="This process's index among --n_tasks (e.g. $SLURM_ARRAY_TASK_ID).")
    p.add_argument("--n_tasks", type=int, default=1,
                   help="Number of parallel processes splitting the batches round-robin.")
    args = p.parse_args()
    if not 0 <= args.task_id < args.n_tasks:
        raise ValueError(f"--task_id {args.task_id} must be in [0, --n_tasks {args.n_tasks})")

    # Completeness check (v2 mock directories, whose run_config.json records both split sizes).
    # Noise seeds are assigned to the batch files in sorted order, so a missing file -- e.g. a
    # generate_bubble_mocks.py array task that has not finished yet -- would shift the seeds of
    # all later batches between runs. Refuse to start until every mock batch exists.
    cfg_path = os.path.join(args.mocks_dir, "run_config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if "n_sim_val" in cfg:
        for prefix, n in (("train", cfg["n_sim"]), ("val", cfg["n_sim_val"])):
            n_expected = -(-n // cfg["batch_size"])           # ceil(n / batch_size)
            n_found = len(glob.glob(os.path.join(args.mocks_dir, f"{prefix}_batch_*.npz")))
            if n_found != n_expected:
                raise RuntimeError(f"{args.mocks_dir}: {n_found} {prefix} batch files, expected "
                                   f"{n_expected} -- is generate_bubble_mocks.py still running?")

    # Load the real catalog: deduplicate, apply the z-window and Muv cut, convert RA/Dec/z to
    # comoving coordinates centred on the sample, and fill the catalog-fixed fields of rdr._S
    # (redshifts, observed EWs and errors, upper-limit flags, CGM transmission, ...).
    meta = rdr._load_catalog_and_priors(
        args.lya_catalog, args.properties_catalog, args.z_lo, args.z_hi,
        args.z_min, args.muv_max, args.main_dir, r_max=args.r_max, prefer=args.prefer,
        legacy_catalog_path=args.legacy_catalog,
    )
    # Redshift edges of the n_los equal-comoving-width bins of every galaxy's sightline, from the
    # galaxy (d_C(z0) + z_gal) down to z_end = 5.3. Fixed by the catalog, so computed once.
    z_b_edges, z_e_edges = _precompute_bin_z_edges(meta, args.n_los)
    print(f"[add_x] {len(meta['x_gal'])} galaxies, n_los={args.n_los}, ew_model={args.ew_model}",
          flush=True)

    # Consistency check: the mocks must have been generated for the SAME galaxies (same catalog
    # filters) and the same n_los. Otherwise galaxy g's mask would be paired with another galaxy's
    # redshift, Muv and noise level without any error being raised.
    any_mock = sorted(glob.glob(os.path.join(args.mocks_dir, "train_batch_*.npz")) or
                      glob.glob(os.path.join(args.mocks_dir, "val_batch_*.npz")))
    if any_mock:
        probe = np.load(any_mock[0])
        # a `KeyError: masks` here means --mocks_dir points at an already processed (theta, x)
        # directory instead of the raw generate_bubble_mocks.py output
        mock_n_gal, mock_n_los = probe["masks"].shape[1], probe["masks"].shape[2]
        if mock_n_gal != len(meta['x_gal']) or mock_n_los != args.n_los:
            raise ValueError(
                f"Mock/catalog mismatch: {any_mock[0]} has n_gal={mock_n_gal}, n_los={mock_n_los}, "
                f"but this run's catalog args give n_gal={len(meta['x_gal'])} and --n_los={args.n_los}. "
                f"Check --lya_catalog/--properties_catalog/--z_hi/--z_min/--muv_max/--prefer match "
                f"what generate_bubble_mocks.py used.")

    # Training and validation splits. The validation noise seed is offset by 1e6 so the two splits
    # never share a noise stream.
    process_split(meta, args.mocks_dir, args.output_dir, "train", args.n_los,
                  z_b_edges, z_e_edges, args.main_dir, args.ew_model, args.seed,
                  task_id=args.task_id, n_tasks=args.n_tasks)
    process_split(meta, args.mocks_dir, args.output_dir, "val", args.n_los,
                  z_b_edges, z_e_edges, args.main_dir, args.ew_model, args.seed + 1_000_000,
                  task_id=args.task_id, n_tasks=args.n_tasks)
    print(f"[add_x] task {args.task_id}/{args.n_tasks} done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
