"""
Full 3D bubble fields of posterior samples, for figures.

The v2 mock pool stores only each mock's sightline masks, not its spheres. But every mock is reproducible:
generate_bubble_mocks.generate_split_v2 draws, per batch, the histories, zeta, f_host and one integer seed per mock
from rng_master, and the mock itself uses only np.random.default_rng(that seed). So we
  1. draw posterior samples from the pool with the exact-likelihood weights w (exact_pixel_likelihood_chunked.py
     combine output),
  2. replay rng_master to recover each sample's (z_mid, dz, zeta, f_host, seed),
  3. rerun bubble_prior_v2.generate_one_mock_v2 for it, keeping the spheres.
Cheap: the replay only draws the per-batch numbers; just the selected mocks are regenerated.

Check: the regenerated n_bubbles, hosted, z_mid, dz, zeta, f_host must equal the pool's stored hyper_* values,
otherwise the replay is not the cluster's mock (different BSD calibration file, seed or numpy stream) and the
script stops.

    python posterior_bubble_fields.py --exact exact_v2_exponential.npz \
        --bsd_calibration bsd_lu24_calibration.npz --n_samples 4 --output posterior_bubbles_exponential.npz

With --n_fog N it also voxelises N posterior and N prior draws (prior: the pool itself, with the tau prior
weights if the combine used --tau_prior) on a grid around the galaxies and saves P(ionized) per voxel for both:
the "probability fog" of the 3D figure (lightcone_3d_posterior_cells.txt). ~1 s per draw on one core.

Output: per sample the pool index, weight, hyperparameters, sightline mask (n_gal, n_los; 1 = ionized, column 0 =
z_end end) and spheres (ragged, concatenated: centers, radii, sample_of_sphere, is_host); the catalogue (ra, dec,
redshift, mock-frame x/y/z of the galaxies); the frame -> redshift map; p_full (P(neutral), exact).
"""
import argparse
import sys
import time

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
from scipy.spatial import cKDTree

import bubble_prior_v2 as bp
from generate_bubble_mocks import load_real_catalog, build_geometry, REPO_DEFAULT

SPLIT_SEED_OFFSET = {"train": 0, "val": 1_000_000}   # generate_bubble_mocks.main: val seed = --seed + 1_000_000
MEDOID_BAND = (0.45, 0.55)  # medoid candidates: box neutral fraction between these posterior quantiles


def pool_location(idx, pool_files, pool_counts):
    """Pool row idx -> (split, batch index, row within the batch)."""
    ends = np.cumsum(pool_counts)
    f = int(np.searchsorted(ends, idx, side="right"))
    name = str(pool_files[f])                            # e.g. 'train_batch_00012.npz'
    split, batch = name.split("_batch_")[0], int(name.split("_batch_")[1].split(".")[0])
    return split, batch, int(idx - (ends[f] - pool_counts[f]))


def replay(split, wanted, seed, batch_size, n_split):
    """The rng_master draws of generate_split_v2 for one split: {(batch, row): (z_mid, dz, x_hi_7, z_end, zeta,
    f_host, mock_seed)} for the (batch, row) pairs in wanted. Identical draw order to generate_split_v2."""
    rng_master = np.random.default_rng(seed + SPLIT_SEED_OFFSET[split])
    last = max(b for b, _ in wanted)
    out, n_done, batch_idx = {}, 0, 0
    while batch_idx <= last:
        this_batch = min(batch_size, n_split - n_done)
        hist = np.array([bp.draw_history(rng_master) for _ in range(this_batch)])
        zeta = rng_master.uniform(*bp.ZETA_RANGE, size=this_batch)
        f_host = rng_master.uniform(*bp.F_HOST_RANGE, size=this_batch)
        mock_seeds = rng_master.integers(0, 2**31 - 1, size=this_batch)
        for b, j in wanted:
            if b == batch_idx:
                out[(b, j)] = (*hist[j], zeta[j], f_host[j], mock_seeds[j])
        n_done += this_batch
        batch_idx += 1
    return out


def regenerate(idx, ex, args, cat, geom, bsd, los_z):
    """Regenerate the pool mocks idx (with repeats allowed) with their spheres, checking each against the pool.
    Yields (k, pool row, mask, hosted, (x_hi_7, z_end, z_mid, dz, zeta, f_host), centers, radii)."""
    loc = [pool_location(i, ex["pool_files"], ex["pool_counts"]) for i in idx]
    draws = {}
    for split, n_split in (("train", args.n_train), ("val", args.n_val)):
        wanted = sorted({(b, j) for s, b, j in loc if s == split})
        if wanted:
            draws.update({(split, b, j): v for (b, j), v in
                          replay(split, wanted, args.pool_seed, args.batch_size, n_split).items()})
    for k, (i, (s, b, j)) in enumerate(zip(idx, loc)):
        z_mid, dz, x7, z_end, zeta, f_host, mseed = draws[(s, b, j)]
        mask, n_b, hosted, _, c, r = bp.generate_one_mock_v2(
            np.random.default_rng(mseed), z_mid, dz, zeta, f_host, bsd, los_z, geom,
            cat["x_gal"], cat["y_gal"], cat["z_gal"], geom["z_end_offset"], args.n_los, return_bubbles=True)
        # the replay must reproduce the pool's stored mock exactly
        stored = dict(z_mid=ex["hyper_z_mid"][i], dz=ex["hyper_dz"][i], zeta=ex["hyper_zeta"][i],
                      f_host=ex["hyper_f_host"][i], n_bubbles=ex["hyper_n_bubbles"][i])
        mine = dict(z_mid=z_mid, dz=dz, zeta=zeta, f_host=f_host, n_bubbles=n_b)
        bad = [key for key in stored if not np.isclose(stored[key], mine[key], rtol=1e-10, atol=0)]
        if bad or not np.array_equal(hosted, ex["hyper_hosted"][i]):
            raise RuntimeError(f"pool row {i} ({s} batch {b} row {j}): replay differs from the pool in "
                               f"{bad or ['hosted']} -- wrong --bsd_calibration, --pool_seed or batch layout?")
        yield k, i, (s, b, j), mask, hosted, (x7, z_end, z_mid, dz, zeta, f_host), c, r


def voxel_grid(geom, los_z, z_lo, voxel):
    """Voxel centres (mock frame, cMpc) covering the galaxies' transverse box and the line of sight from redshift
    z_lo to just behind the furthest galaxy."""
    zc = np.linspace(geom["z_end_offset"], geom["z_hi"], 20000)
    zc_lo = zc[np.searchsorted(los_z(zc), z_lo)]            # los_z is increasing in zc
    gx = np.arange(-geom["x_half"], geom["x_half"] + 1e-9, voxel)
    gy = np.arange(-geom["y_half"], geom["y_half"] + 1e-9, voxel)
    gz = np.arange(zc_lo, geom["z_hi"] + 1e-9, voxel)
    return gx, gy, gz


def ionized_voxels(tree, n_vox, centers, radii, lo, hi):
    """Boolean (n_vox,): voxel centres inside any sphere. Only spheres that reach the grid's bounding box."""
    near = np.all((centers + radii[:, None] >= lo) & (centers - radii[:, None] <= hi), axis=1)
    inside = np.zeros(n_vox, bool)
    for ids in tree.query_ball_point(centers[near], radii[near]):
        inside[ids] = True
    return inside


def medoid(fields, draw, ex, args, cat, geom, bsd, los_z):
    """The most representative posterior draw: the medoid, i.e. the draw whose ionization field (voxel grid) has the
    smallest mean Hamming distance (fraction of voxels with a different state) to all other posterior draws. The
    draws are already posterior-distributed (weighted resampling), so a plain mean is the posterior expectation.
    Candidates are restricted to draws whose neutral fraction in the box lies within MEDOID_BAND (posterior quantiles)
    of the median: an unrestricted Hamming medoid is biased towards almost fully ionized fields (they differ from the
    others only where those are neutral), i.e. it would under-represent the neutral gas.
    Returns its pool row, mean distance, hyperparameters, sightline mask and spheres."""
    f = fields.astype(np.float32)
    n_on = f.sum(axis=1)
    # Hamming(i, j) = |a_i| + |a_j| - 2 a_i . a_j  (a = ionized-voxel indicator)
    ham = (n_on[:, None] + n_on[None, :] - 2 * (f @ f.T)) / fields.shape[1]
    mean_d = ham.mean(axis=1)
    x_hi_box = 1 - n_on / fields.shape[1]
    lo, hi = np.quantile(x_hi_box, MEDOID_BAND)
    cand = np.where((x_hi_box >= lo) & (x_hi_box <= hi))[0]
    m = int(cand[np.argmin(mean_d[cand])])
    print(f"[medoid] box neutral fraction of the posterior draws: median {np.median(x_hi_box):.3f}, "
          f"{len(cand)} candidates in [{lo:.3f}, {hi:.3f}]; chosen {x_hi_box[m]:.3f}", flush=True)
    row = draw[m]
    (_, i, _, mask, hosted, h, c, r), = regenerate(np.array([row]), ex, args, cat, geom, bsd, los_z)
    print(f"[medoid] pool row {row}: mean voxel disagreement with the other posterior draws {mean_d[m]:.3f} "
          f"(median draw {np.median(mean_d):.3f}); x_hi_7 {h[0]:.2f}, z_end {h[1]:.2f}, zeta {h[4]:.2f}, "
          f"f_host {h[5]:.2f}, {len(r)} spheres", flush=True)
    return dict(medoid_row=row, medoid_mean_dist=mean_d[m], post_mean_dist=mean_d, post_x_hi_box=x_hi_box,
                medoid_hyper=np.array([*h, ex["hyper_x_ion"][i], ex["hyper_tau_e"][i]]),
                medoid_mask=mask, medoid_hosted=hosted, medoid_centers=c, medoid_radii=r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exact", required=True, help="combine output of exact_pixel_likelihood_chunked.py")
    p.add_argument("--bsd_calibration", required=True, help="the SAME bsd_lu24_calibration.npz the pool used")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--pick", default="sir", choices=["sir", "top"],
                   help="sir: posterior draws (weighted resampling, default); top: the highest-weight maps")
    p.add_argument("--seed", type=int, default=1, help="seed of the posterior resampling")
    p.add_argument("--pool_seed", type=int, default=0, help="--seed of generate_bubble_mocks.py (pool default 0)")
    p.add_argument("--batch_size", type=int, default=2000, help="--batch_size of generate_bubble_mocks.py")
    p.add_argument("--n_train", type=int, default=1_000_000, help="--n_sim of generate_bubble_mocks.py")
    p.add_argument("--n_val", type=int, default=200_000, help="--n_sim_val of generate_bubble_mocks.py")
    p.add_argument("--n_los", type=int, default=75)
    p.add_argument("--z_hi", type=float, default=7.3)
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--n_fog", type=int, default=0,
                   help="posterior and prior draws to voxelise for the P(ionized) fog (0 = no fog)")
    p.add_argument("--fog_z_lo", type=float, default=6.6, help="low-redshift end of the fog grid")
    p.add_argument("--fog_voxel", type=float, default=1.5, help="fog voxel size [cMpc]")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ex = np.load(args.exact, allow_pickle=True)
    w = ex["w"]
    if args.pick == "sir":
        idx = np.random.default_rng(args.seed).choice(len(w), size=args.n_samples, replace=False, p=w / w.sum())
    else:
        idx = np.argsort(w)[::-1][:args.n_samples]
    idx = np.sort(idx)

    cat = load_real_catalog(args.repo, args.z_hi)
    # the same catalogue again, for the Lya EWs (load_real_catalog keeps only positions); same cuts and order
    rdmod = sys.modules["real_data_standalone"]
    full = rdmod.load_catalog_v2(f"{args.repo}/tb_lya.txt", f"{args.repo}/sample_nirspec_properties.txt",
                                 z_min=5.0, muv_max=-18.0, prefer="grating")
    keep = full.redshift <= args.z_hi
    rd = argparse.Namespace(ew=full.ew[keep], ew_err=full.ew_err[keep], is_upper_limit=full.is_upper_limit[keep])
    assert np.allclose(full.ra[keep], cat["ra"]) and np.allclose(full.redshift[keep], cat["redshift"])
    geom = build_geometry(cat)
    bsd = bp.BSD(args.bsd_calibration)
    los_z = bp.LOSRedshift(cat["z0"], cat["z_mean"])
    n_gal = len(cat["x_gal"])

    # ---- 1. individual samples, with their spheres ----
    masks, hyper, centers, radii, owner, is_host = [], [], [], [], [], []
    for k, i, (s, b, j), mask, hosted, h, c, r in regenerate(idx, ex, args, cat, geom, bsd, los_z):
        masks.append(mask)
        hyper.append([*h, ex["hyper_x_ion"][i], ex["hyper_tau_e"][i]])
        centers.append(c)
        radii.append(r)
        owner.append(np.full(len(r), k))
        is_host.append(np.arange(len(r)) >= len(r) - hosted.sum())    # host spheres are appended last
        print(f"[sample {k}] pool row {i} ({s} batch {b} row {j}), w/max(w) = {w[i] / w.max():.3f}: "
              f"x_hi_7 {h[0]:.2f}, z_end {h[1]:.2f}, zeta {h[4]:.2f}, f_host {h[5]:.2f}, "
              f"{len(r)} spheres ({hosted.sum()} hosts) -- replay verified", flush=True)

    # ---- 2. probability fog: P(ionized) per voxel, posterior and prior ----
    fog = {}
    if args.n_fog > 0:
        gx, gy, gz = voxel_grid(geom, los_z, args.fog_z_lo, args.fog_voxel)
        vox = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        tree = cKDTree(vox)
        lo, hi = vox.min(axis=0), vox.max(axis=0)
        rng = np.random.default_rng(args.seed + 1)
        prior_p = np.exp(ex["log_prior_tau"] - ex["log_prior_tau"].max())  # all equal without --tau_prior
        for name, pw in (("post", w), ("prior", prior_p)):
            draw = np.sort(rng.choice(len(pw), size=args.n_fog, replace=True, p=pw / pw.sum()))
            acc = np.zeros(len(vox), np.float32)
            fields = np.zeros((args.n_fog, len(vox)), bool) if name == "post" else None
            t0 = time.perf_counter()
            for k, i, _, mask, hosted, h, c, r in regenerate(draw, ex, args, cat, geom, bsd, los_z):
                ion = ionized_voxels(tree, len(vox), c, r, lo, hi)
                acc += ion
                if fields is not None:
                    fields[k] = ion
                if (k + 1) % 50 == 0:
                    print(f"[fog {name}] {k + 1}/{args.n_fog} draws ({time.perf_counter() - t0:.0f} s)", flush=True)
            fog[f"fog_{name}"] = (acc / args.n_fog).reshape(len(gx), len(gy), len(gz))
            fog[f"fog_{name}_rows"] = draw
            if fields is not None:
                fog.update(medoid(fields, draw, ex, args, cat, geom, bsd, los_z))
        fog.update(fog_x=gx, fog_y=gy, fog_zc=gz, fog_z=los_z(gz))
        print(f"[fog] grid {len(gx)} x {len(gy)} x {len(gz)} voxels of {args.fog_voxel} cMpc, "
              f"z {los_z(gz[0]):.2f}-{los_z(gz[-1]):.2f}; mean P(ionized) posterior "
              f"{fog['fog_post'].mean():.3f}, prior {fog['fog_prior'].mean():.3f}", flush=True)

    # mock-frame LOS coordinate -> redshift table (for plotting with redshift on the axis)
    zc_grid = np.linspace(geom["gz_lo"] - 200, geom["gz_hi"] + 200, 4000)
    np.savez(args.output, pool_index=idx, weight=w[idx], w_max=w.max(),
             hyper=np.array(hyper),
             hyper_names=np.array(["x_hi_7", "z_end", "z_mid", "dz", "zeta", "f_host", "x_ion", "tau_e"]),
             masks=np.array(masks), centers=np.concatenate(centers), radii=np.concatenate(radii),
             sample_of_sphere=np.concatenate(owner), is_host=np.concatenate(is_host),
             ra=cat["ra"], dec=cat["dec"], redshift=cat["redshift"],
             x_gal=cat["x_gal"], y_gal=cat["y_gal"], z_gal=cat["z_gal"],
             z_end_offset=geom["z_end_offset"], x_half=geom["x_half"], y_half=geom["y_half"],
             zc_grid=zc_grid, z_of_zc=los_z(zc_grid),
             x_mean=cat["x_mean"], y_mean=cat["y_mean"], ra0=cat["ra0"], dec0=cat["dec0"],
             d_c0=Cosmo.comoving_distance(cat["z0"]).to(u.Mpc).value, z_mean=cat["z_mean"],
             ew=rd.ew, ew_err=rd.ew_err, is_upper_limit=rd.is_upper_limit,
             p_full=ex["p_full"], ew_model=ex["ew_model"], n_gal=n_gal, n_los=args.n_los, **fog)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
