"""
Sanity-check the pipeline's fiducial astro_params (the Map_parm block in
lyabubbles/galaxy_prop.py's UV-LF snippet, ~line 392) against two
independent physical constraints, before committing to the full ~40-50
lightcone campaign (see cluster-storage-expansion memory):

1. UV luminosity function (p21c.compute_luminosity_function) -- purely
   analytic (halo mass function + astro_params), no lightcone needed.
2. CMB optical depth tau_e (p21c.compute_tau) -- needs a REAL reionization
   history, so this section runs the first actual lightcone of the
   campaign: z=5.3->20, wide enough to capture the full x_HI: 0->1
   transition tau_e needs. Wider (and slower) than a typical ~z=6.5-8.3
   training lightcone -- a one-off per astro_params combination for
   calibration, not the per-seed campaign shape.

VERSION: confirmed py21cmfast 4.2 (user-reported) -- this rewrite targets
the v4 API (InputParameters/AstroOptions/OutputCache), replacing the v3-
style plain-dict-kwargs version this script started as.

IMPORTANT DESIGN CHOICE, read before running: v4 restructured FlagOptions
into AstroOptions/MatterOptions/SimulationOptions, and several of the OLD
flag_options this pipeline used (USE_MASS_DEPENDENT_ZETA, EVOLVING_R_BUBBLE_
MAX) don't exist in v4 at all; PHOTON_CONS became the string enum
PHOTON_CONS_TYPE; INHOMO_RECO is deprecated in favor of RECOMB_MODEL. Rather
than guess a translation (a physics-correctness risk, not just a syntax
one), this script builds `inputs` from a v4 template ('simple') and ONLY
overrides the astro_params/cosmo_params/box values that must match the
existing pipeline -- every flag/option field is left at the template's own
default. Section 0 below prints those resolved defaults so you can eyeball
whether they're reasonable for this science case (mass-dependent SFR/fesc
scaling, inhomogeneous recombinations, photon conservation) before trusting
the tau_e/UVLF numbers that follow. If any of those defaults look wrong for
this project, tell me and we'll override them explicitly instead of
accepting the template.

Still UNVERIFIED (no py21cmfast available in this session to actually run
this against): (a) whether HII_DIM/BOX_LEN are still the correct
SimulationOptions field names in v4 (if the from_template() call below
errors on these specific kwargs, that's the tell), (b) whether
run_lightcone(...) returns the final LightCone directly from a plain
assignment or needs .exhaust_lightcone() -- handled defensively below by
checking for a `global_xHI` attribute and falling back if absent, (c) the
z_step_factor=1.05 node-redshift spacing is a first guess, not tuned.

Needs: py21cmfast 4.x on the cluster. Does NOT need the real galaxy catalog
-- this only checks the astro_params in isolation.
"""
import os

import numpy as np
import py21cmfast as p21c

# ---- fiducial astro_params/cosmo_params, exactly matching lyabubbles/galaxy_prop.py ----
# Field names confirmed unchanged from v3 in AstroParams/CosmoParams (per
# the v3->v4 migration docs: "CosmoParams: same as v3" / "AstroParams: same
# as v3 with some additions").
FIDUCIAL_OVERRIDES = dict(
    F_STAR10=-1.21,
    ALPHA_STAR=0.50,
    M_TURN=8.65,
    t_STAR=0.55,
    F_ESC10=-0.971704175935387049,
    ALPHA_ESC=-0.498862738992503996,
    hlittle=0.6688,
    OMm=0.321,
    OMb=0.04952,
    POWER_INDEX=0.9626,
    SIGMA_8=0.8118,
    # Confirmed valid SimulationOptions override kwargs (HII_DIM via the
    # official 'Qin20' template example; N_THREADS via its own docs entry).
    HII_DIM=256,
    BOX_LEN=384,
    # SimulationOptions default is N_THREADS=1 -- MUST match --cpus-per-task
    # in the #SBATCH line (the real one, not just what you meant to request
    # -- see the ##SBATCH silent-comment trap in cluster-workflow-notes
    # memory). ADJUST to your actual allocation.
    N_THREADS=16,
)

# ---- where things get saved ----
# CACHE_DIR: the RAW 21cmFAST outputs (every field for every one of the
# ~25 node redshifts below: density/velocity/ionization/brightness_temp
# etc, whatever this run configuration produces) -- OutputCache decides the
# on-disk layout/filenames itself, hashed by input params, same convention
# as the existing snapshot table's path in lyabubbles/lightcone_field.py.
# This is a MUCH bigger footprint than the ~260 MB/lightcone estimate from
# the campaign-planning conversation -- that number was for the 2 fields
# (x_HI + density) worth keeping for actual SBI training use, not
# everything py21cmfast writes internally per timestep. Budget a few GB for
# this one pilot run; ADJUST the path to somewhere on your 2.5 TB.
CACHE_DIR = '/lustre/astro/ivannik/21cmFAST_cache/tau_uvlf_check/'  # ADJUST
# OUT_DIR: the two small .npz result files this script writes (KB-scale).
# Made explicit/absolute rather than left as relative paths (which would've
# landed wherever the job's working directory happened to be) so they're
# easy to find afterward.
OUT_DIR = '/lustre/astro/ivannik/21cmFAST_cache/tau_uvlf_check_results/'  # ADJUST
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

Z_MIN = 5.3     # matches Z_END_DEFAULT in lyabubbles/lightcone_field.py
Z_MAX = 20.0    # generous margin past full neutrality -- CHECK below, may need raising

node_z = p21c.wrapper.inputs.get_logspaced_redshifts(
    min_redshift=Z_MIN, z_step_factor=1.05, max_redshift=Z_MAX,
)
# z_step_factor=1.05 over z=5.3->20 gives ~25 node redshifts (log((1+Z_MAX)
# /(1+Z_MIN)) / log(z_step_factor) = log(21/6.3)/log(1.05) =~ 25) -- a
# modest number of coeval-timestep evaluations, not the dominant cost
# driver; HII_DIM=256 per-timestep cost matters more. Still unbenchmarked
# end-to-end, hence the generous walltime request below.

inputs = p21c.InputParameters.from_template(
    ['simple'],
    node_redshifts=node_z,
    random_seed=0,
    **FIDUCIAL_OVERRIDES,
)

# ============================== 0. sanity-print the flags we did NOT set ===
print("[inputs] astro_options (from 'simple' template, not explicitly overridden):")
print(inputs.astro_options)
print("[inputs] matter_options:")
print(inputs.matter_options)
print("[inputs] resolved cell size:", inputs.simulation_options.cell_size)
print("--- eyeball the above against what this project actually wants "
      "(mass-dependent SFR/fesc scaling, inhomogeneous recombinations, "
      "photon conservation) before trusting tau_e/UVLF below ---\n")

# ============================== 1. UV luminosity function ===================
# No lightcone/simulation needed -- fast. Redshifts bracket the real
# catalog's z~6.9-7.3 with margin either side to see the trend.
UVLF_REDSHIFTS = [6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]

Muv, Mh, lf = p21c.compute_luminosity_function(
    redshifts=UVLF_REDSHIFTS,
    inputs=inputs,
)
# Muv, lf shapes: (n_z, n_Muv_bins).

uvlf_out = os.path.join(OUT_DIR, 'uvlf_fiducial_check.npz')
np.savez(uvlf_out, Muv=Muv, Mh=Mh, lf=lf,
         redshifts=UVLF_REDSHIFTS, astro_overrides=FIDUCIAL_OVERRIDES)
print(f"[UVLF] saved {uvlf_out}")
for i, z in enumerate(UVLF_REDSHIFTS):
    # Quick sanity print only -- no observational UV LF table is wired into
    # this repo yet, so eyeball this by hand against e.g. Bouwens+2021 /
    # Finkelstein+2022 (peak phi should land roughly Muv~-21 to -20 at
    # z~6-7 for a realistic model). Say if you want an actual overlay --
    # I don't have a verified observational table to add on my own.
    valid = np.isfinite(lf[i]) & (lf[i] > -90)
    if valid.any():
        i_peak = np.argmax(lf[i][valid])
        print(f"  z={z}: Muv range [{Muv[i][valid].min():.1f}, {Muv[i][valid].max():.1f}], "
              f"phi peak near Muv={Muv[i][valid][i_peak]:.2f}")

# ============================== 2. tau_e (needs a real lightcone) ===========
cache = p21c.OutputCache(CACHE_DIR)

lcn = p21c.RectilinearLightconer.between_redshifts(
    min_redshift=Z_MIN,
    max_redshift=Z_MAX,
    quantities=("brightness_temp", "xH_box"),
    resolution=inputs.simulation_options.cell_size,
)

lightcone = p21c.run_lightcone(lightconer=lcn, inputs=inputs, cache=cache, progressbar=True)
if not hasattr(lightcone, "global_xHI"):
    # run_lightcone is documented as a generator in some usages; a direct
    # assignment SHOULD already give the final LightCone (per the official
    # tutorial's own example), but this is a defensive fallback in case it
    # doesn't in this version -- UNVERIFIED which branch actually fires.
    lightcone = lightcone.exhaust_lightcone()

z_hist   = np.asarray(lightcone.node_redshifts)
xHI_hist = np.asarray(lightcone.global_xHI)

i_lo, i_hi = np.argmin(z_hist), np.argmax(z_hist)
print(f"\n[lightcone] x_HI(z={z_hist[i_lo]:.2f}) = {xHI_hist[i_lo]:.4f}  (should be ~0, fully ionized)")
print(f"[lightcone] x_HI(z={z_hist[i_hi]:.2f}) = {xHI_hist[i_hi]:.4f}  (should be ~1 -- "
      f"if well below 1, raise Z_MAX and rerun; tau_e would otherwise be biased LOW, "
      f"missing the high-z neutral-IGM contribution)")

tau_e = p21c.compute_tau(redshifts=z_hist, global_xHI=xHI_hist, inputs=inputs)

TAU_PLANCK, TAU_PLANCK_ERR = 0.0544, 0.0073   # Planck 2018 TT,TE,EE+lowE+lensing
n_sigma = (tau_e - TAU_PLANCK) / TAU_PLANCK_ERR
print(f"\n[tau_e] fiducial astro_params give tau_e = {tau_e:.4f}")
print(f"[tau_e] Planck 2018: {TAU_PLANCK} +/- {TAU_PLANCK_ERR}  "
      f"({n_sigma:+.2f} sigma from fiducial)")

tau_out = os.path.join(OUT_DIR, 'tau_fiducial_check.npz')
np.savez(tau_out, z_hist=z_hist, xHI_hist=xHI_hist,
         tau_e=tau_e, astro_overrides=FIDUCIAL_OVERRIDES)
print(f"[tau_e] saved {tau_out}")

# This lightcone's 5.3-20 span covers the ~6.5-8.3 SBI training window
# (z_end..z_hi) within it -- it can double as the first of the ~40-50
# campaign lightcones once lyabubbles/lightcone_field.py + sbi_pixel_field.py
# are updated to draw from a per-seed lightcone list instead of the current
# single 13-snapshot table (not done yet, see cluster-storage-expansion
# memory's "next concrete step"). One seed's tau_e is a reasonable starting
# point for F_ESC10 calibration -- seed-to-seed tau_e scatter at this box
# size should be modest but isn't exactly zero; average across 2-3 seeds
# instead if the calibration needs to be tighter than that.
