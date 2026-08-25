"""
Sanity-check the pipeline's fiducial astro_params (the Map_parm block in
lyabubbles/galaxy_prop.py's UV-LF snippet, ~line 392) against two
independent physical constraints, before committing to the full ~40-50
lightcone campaign (see cluster-storage-expansion memory):

1. UV luminosity function (p21c.compute_luminosity_function) -- purely
   analytic (halo mass function + astro_params), no lightcone needed.
   CONFIRMED WORKING on the cluster (first run, 2026-08-25).
2. CMB optical depth tau_e (p21c.compute_tau), fed by p21c.run_global_
   evolution -- needs a REAL reionization history spanning z=5.3->20 (wide
   enough to capture the full x_HI: 0->1 transition), wider than the
   ~z=6.5-8.3 training window the SBI sightlines actually use. A one-off
   per astro_params combination for calibration, not the per-seed campaign
   shape.
3. The actual spatial lightcone (density + ionization cube), generated
   separately in section 2b -- not needed for tau_e itself, but this is
   what can double as the first of the ~40-50 campaign lightcones later.

VERSION: confirmed py21cmfast 4.2. v4 restructured FlagOptions into
AstroOptions/MatterOptions/SimulationOptions; several of this pipeline's
OLD flag_options don't exist as-named (USE_MASS_DEPENDENT_ZETA gone
entirely -- mass-dependent SFR/fesc now appears to be the only supported
mode, given AstroParams unconditionally carries F_STAR10/F_ESC10/etc.;
EVOLVING_R_BUBBLE_MAX also gone, no confirmed v4 equivalent found -- left
at the template default, not guessed, since it's a spatial bubble-
morphology setting more relevant to later pixel-field structure than to
this tau_e check specifically). Two flags DO have confirmed v4 equivalents
(docs quote the physical description, not just a name change) and are
explicitly overridden in FIDUCIAL_OVERRIDES rather than left at the
'simple' template's non-matching defaults (RECOMB_MODEL='none',
PHOTON_CONS_TYPE='no-photoncons', printed by the first run -- did NOT
match old intent):
  - RECOMB_MODEL='inhomogeneous' == old INHOMO_RECO=True (same Sobacchi &
    Mesinger 2014 model, confirmed from the docs' own description).
  - PHOTON_CONS_TYPE='z-photoncons' == old PHOTON_CONS=True's default
    behavior (redshift-recalibration correction, Park+22).
Section 0 below still prints the resolved astro_options/matter_options so
you can eyeball everything NOT explicitly overridden.

CONFIRMED WORKING on the cluster (2026-08-25): HII_DIM/BOX_LEN/N_THREADS as
SimulationOptions override kwargs, 'simple' template + from_template()
construction, the whole UV-LF section (two informational, non-fatal
warnings: USE_MINI_HALOS=False -> ACG-only LFs, matches pre-existing
intent; USE_TS_FLUCT=False -> brightness_temp inaccurate before Ts
saturates at high z, but that's brightness_temp not x_HI, so likely fine
for tau_e/morphology purposes -- not 100% certain there's zero feedback
onto ionization, flagged rather than dismissed).

FIXED across two runs: (a) RectilinearLightconer.between_redshifts() built
its own default cosmology instead of the custom one from
FIDUCIAL_OVERRIDES -- fixed with cosmo=inputs.cosmo_params.cosmo. (b) the
lightconer's `quantities=(...)` used the old v3 field name 'xH_box', which
doesn't exist in v4 -- crashed with a ValueError that helpfully listed the
correct name, 'neutral_fraction', among the valid outputs for these
inputs -- fixed. (c) switched tau_e's data source from run_lightcone's
lightcone object to run_global_evolution(inputs=inputs) directly -- a
dedicated function for exactly this (confirmed from the 21cmFAST source,
py21cmfast/drivers/global_evolution.py: returns a GlobalEvolution object
with `.quantities['neutral_fraction']` and `.node_redshifts`), and it's
what run_lightcone was already calling internally anyway (visible in the
first run's own warning line) -- cleaner than depending on a LightCone
object's global-quantity attribute name, which was never confirmed.

Still UNVERIFIED (section 2b only, since 2a no longer depends on this):
whether run_lightcone(...) returns the final LightCone directly from a
plain assignment or needs .exhaust_lightcone() -- handled defensively by
checking for a `node_redshifts` attribute and falling back if absent; the
z_step_factor=1.05 node-redshift spacing is a first guess, not tuned or
benchmarked for cost.

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
    # v4 equivalents of this pipeline's old flag_options (which don't exist
    # as-named in v4 -- see module docstring). The 'simple' template's own
    # defaults for these (RECOMB_MODEL='none', PHOTON_CONS_TYPE=
    # 'no-photoncons', printed by the first cluster run) do NOT match the
    # old intent (INHOMO_RECO=True, PHOTON_CONS=True), so overriding
    # explicitly rather than accepting the template here:
    #   RECOMB_MODEL='inhomogeneous' -- docs: "recombination rate calculated
    #   locally at every cell (Sobacchi & Mesinger 2014)" -- this IS the old
    #   INHOMO_RECO=True model, same physics/citation.
    #   PHOTON_CONS_TYPE='z-photoncons' -- docs: "adjusting the redshift of
    #   the N_ion source field (Park+22)" -- the classic photon-conservation
    #   correction, matching old PHOTON_CONS=True's default behavior.
    # NOT set: old EVOLVING_R_BUBBLE_MAX has no equally clear v4 equivalent
    # (possibly folded into USE_EXP_FILTER/HII_FILTER, possibly removed) --
    # left at the template default rather than guessed. It's a spatial
    # bubble-morphology setting, more relevant to later pixel-field
    # structure than to this tau_e check -- lower priority to resolve.
    RECOMB_MODEL='inhomogeneous',
    PHOTON_CONS_TYPE='z-photoncons',
    # 'simple' template's R_BUBBLE_MAX=15.0 triggered a UserWarning once
    # RECOMB_MODEL was changed from 'none' ("You are setting R_BUBBLE_MAX
    # != 50 when RECOMB_MODEL != 'none'... non-standard... usually occurs
    # upon manual update of RECOMB_MODEL") -- the warning itself names 50
    # as the standard pairing, so setting it explicitly rather than leaving
    # the now-mismatched template default in place.
    R_BUBBLE_MAX=50.0,
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
# Raised from an original 20.0: RECOMB_MODEL='inhomogeneous' requires
# max(node_redshifts) to be ABOVE Z_HEAT_MAX (=35.0, SimulationOptions
# default, not overridden here) -- 20.0 was below that and raised a
# ValueError on construction. 40.0 clears it with margin (also physically
# fine/better -- higher z_max just means a more complete reionization
# history for tau_e, not a compromise).
Z_MAX = 40.0    # generous margin past full neutrality AND past Z_HEAT_MAX

node_z = p21c.wrapper.inputs.get_logspaced_redshifts(
    min_redshift=Z_MIN, z_step_factor=1.05, max_redshift=Z_MAX,
)
# z_step_factor=1.05 over z=5.3->40 gives ~38 node redshifts (log((1+Z_MAX)
# /(1+Z_MIN)) / log(z_step_factor) = log(41/6.3)/log(1.05) =~ 38, up from
# ~25 when Z_MAX was 20 before the Z_HEAT_MAX fix) -- still a modest number
# of coeval-timestep evaluations, not the dominant cost driver; HII_DIM=256
# per-timestep cost matters more. Still unbenchmarked
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

# ============================== 2a. tau_e via run_global_evolution =========
# Dedicated function for exactly this (source: py21cmfast/drivers/
# global_evolution.py) -- returns a GlobalEvolution object with a
# `quantities` dict keyed by field name (confirmed 'neutral_fraction' is
# the correct v4 key, from the crash traceback's own list of valid output
# names) and a `node_redshifts` property proxying inputs.node_redshifts.
# No lightconer/cosmology-matching issue here since it only takes `inputs`
# directly -- cleaner than fighting run_lightcone for something it wasn't
# the most direct tool for. This is ALSO exactly what run_lightcone calls
# internally under the hood (visible in the first run's warning line,
# "global_evolution = run_global_evolution(inputs=inputs)"), so nothing is
# lost by calling it explicitly instead.
global_evolution = p21c.run_global_evolution(inputs=inputs, progressbar=True)

z_hist   = np.asarray(global_evolution.node_redshifts)
xHI_hist = np.asarray(global_evolution.quantities['neutral_fraction'])

i_lo, i_hi = np.argmin(z_hist), np.argmax(z_hist)
print(f"\n[global_evolution] x_HI(z={z_hist[i_lo]:.2f}) = {xHI_hist[i_lo]:.4f}  (should be ~0, fully ionized)")
print(f"[global_evolution] x_HI(z={z_hist[i_hi]:.2f}) = {xHI_hist[i_hi]:.4f}  (should be ~1 -- "
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

# ============================== 2b. the actual spatial lightcone ===========
# run_global_evolution above is enough for tau_e alone -- this section
# generates the real 3D lightcone (density + ionization structure), which
# is the part that can double as the first of the ~40-50 campaign
# lightcones once lyabubbles/lightcone_field.py + sbi_pixel_field.py are
# updated to draw from a per-seed lightcone list instead of the current
# single 13-snapshot table (not done yet, see cluster-storage-expansion
# memory's "next concrete step"). Comment this section out if you only
# wanted the tau_e/UVLF check for now and don't need the cached cube yet --
# it's the most expensive part of the script.
cache = p21c.OutputCache(CACHE_DIR)

lcn = p21c.RectilinearLightconer.between_redshifts(
    min_redshift=Z_MIN,
    max_redshift=Z_MAX,
    # 'xH_box' (old v3 name) doesn't exist in v4 -- confirmed correct name
    # 'neutral_fraction' from the crash traceback's own list of valid
    # output arrays for these inputs.
    quantities=("brightness_temp", "neutral_fraction"),
    resolution=inputs.simulation_options.cell_size,
    # Without this, between_redshifts() builds its own default cosmology,
    # which doesn't match the custom one set via FIDUCIAL_OVERRIDES
    # (hlittle/OMm/OMb/...) -- caused the "lightconer.cosmo is not the same
    # as inputs.cosmo_params.cosmo" ValueError on the first run.
    cosmo=inputs.cosmo_params.cosmo,
)

lightcone = p21c.run_lightcone(lightconer=lcn, inputs=inputs, cache=cache, progressbar=True)
if not hasattr(lightcone, "node_redshifts"):
    # run_lightcone is documented as a generator in some usages; a direct
    # assignment SHOULD already give the final LightCone (per the official
    # tutorial's own example), but this is a defensive fallback in case it
    # doesn't in this version -- UNVERIFIED which branch actually fires.
    lightcone = lightcone.exhaust_lightcone()

print(f"\n[lightcone] generated and cached in {CACHE_DIR} -- "
      f"{len(lightcone.node_redshifts)} node redshifts, "
      f"quantities={('brightness_temp', 'neutral_fraction')}")

# One seed's tau_e (from 2a above) is a reasonable starting point for
# F_ESC10 calibration -- seed-to-seed tau_e scatter at this box size should
# be modest but isn't exactly zero; average across 2-3 seeds instead if the
# calibration needs to be tighter than that.
