"""Compare real-data pixel-field posteriors obtained with different intrinsic Lya EW models
(sbi_runs/pixel_mock_<model>/pixel_infer.npz from submit_ew_model_runs.sh). numpy/matplotlib only.

The first entry is the reference (normally `exponential`). theta (bubble mocks) is identical across
runs, so the prior is the same and any difference is due to the EW model alone.

Figure 1 (ew_model_posteriors.png):
  a  posterior of the near-source neutral fraction (first N_NEAR bins of all sightlines; Lya-informed)
  b  posterior of the global neutral fraction (mostly prior structure, see sbi_pixel_sbc.py)
  c  per-galaxy near-source P(neutral): each model vs the reference
Figure 2 (ew_model_maps.png): marginal P(neutral) maps per model + difference to the reference,
  x = LOS bin index.
Figure 3 (ew_model_maps_z.png, only with catalog=...): the same maps with x = REDSHIFT. Each galaxy's
  row is drawn at its own bins' redshifts (from z_end = 5.3 up to the galaxy), so structure at a common
  redshift lines up vertically instead of appearing as a diagonal. Bin k of galaxy g spans comoving
  distance d_C(z_g) - [k, k+1] * Delta_g, Delta_g = (d_C(z_g) - d_C(5.3)) / n_los (Planck18), the same
  equal-comoving-width bins as lyabubbles.lightcone_field.bin_z_edges (which starts the sightline at
  d_C(z0) + z_gal, i.e. 0.25 cMpc from d_C(z_g) for GOODS-N -- negligible at this resolution).

Usage: python compare_ew_models.py exponential=path/pixel_infer.npz tang_muv=... gagnon_hartman=... \
           [catalog=catalog.npz] [zmin=6.5]
  catalog.npz must hold the galaxy redshifts (key `z`) in the SAME order as the posterior rows, i.e.
  the order of rdr._S.redshifts (the GOODS-N catalog is z-sorted; ~/Downloads/catalog43.npz matches).
  zmin sets the low-redshift edge of figure 3 (default: z_end = 5.3).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

N_NEAR = 5
COLORS = {'exponential': '#2a78d6', 'tang_muv': '#eb6834', 'gagnon_hartman': '#1baf7a'}
LABELS = {'exponential': 'exponential (fiducial)', 'tang_muv': 'Tang+24 (Muv-dependent, z~5-6)',
          'gagnon_hartman': 'Gagnon-Hartman'}
FALLBACK = ['#2a78d6', '#eb6834', '#1baf7a', '#8a6fd6']
INK, MUTED = '#0b0b0b', '#52514e'

runs = []
z_gal = None          # galaxy redshifts in posterior-row order (enables figure 3)
zmin = 5.3            # low-redshift edge of figure 3
for i, arg in enumerate(sys.argv[1:]):
    name, path = arg.split('=', 1)
    if name == 'catalog':
        z_gal = np.asarray(np.load(path, allow_pickle=True)['z'], dtype=float)
        continue
    if name == 'zmin':
        zmin = float(path)
        continue
    d = np.load(path)
    js = d['joint_samples']
    runs.append(dict(name=name, label=LABELS.get(name, name), c=COLORS.get(name, FALLBACK[i % 4]),
                     m=d['marginal_map'], ess=float(d['ess']),
                     near=js[:, :, :N_NEAR].mean(axis=(1, 2)), glob=js.mean(axis=(1, 2)),
                     near_g=d['marginal_map'][:, :N_NEAR].mean(axis=1)))
if len(runs) < 2:
    sys.exit(__doc__)
ref = runs[0]

print(f"{'model':>16s} {'ESS':>6s} {'near (16/50/84)':>22s} {'global (16/50/84)':>22s} {'max px':>7s}")
for r in runs:
    q = lambda a: '/'.join(f'{v:.3f}' for v in np.percentile(a, [16, 50, 84]))
    print(f"{r['name']:>16s} {r['ess']:6.0f} {q(r['near']):>22s} {q(r['glob']):>22s} {r['m'].max():7.3f}")
for r in runs[1:]:
    dn = r['near'].mean() - ref['near'].mean()
    sd = np.hypot(r['near'].std(), ref['near'].std())
    print(f"{r['name']} - {ref['name']}: near-source mean shift {dn:+.3f} ({dn / sd:+.2f} combined sigma); "
          f"per-galaxy max |dP| {np.abs(r['near_g'] - ref['near_g']).max():.3f}")

plt.rcParams.update({'font.size': 11})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
bins = np.linspace(0, 1, 41)
for r in runs:
    for ax, key in ((axs[0], 'near'), (axs[1], 'glob')):
        ax.hist(r[key], bins=bins, histtype='step', lw=2, color=r['c'], density=True,
                label=f"{r['label']}  (ESS {r['ess']:.0f})")
axs[0].set_xlabel(f'near-source neutral fraction (first {N_NEAR} bins)')
axs[1].set_xlabel('global neutral fraction (all bins)')
for ax in axs[:2]:
    ax.set_ylabel('posterior density'); ax.set_yticks([])
axs[0].legend(frameon=False, fontsize=8.5, loc='upper right')
lim = max(r['near_g'].max() for r in runs) * 1.08
axs[2].plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=':')
for r in runs[1:]:
    axs[2].scatter(ref['near_g'], r['near_g'], s=26, color=r['c'], edgecolor='white', lw=0.6,
                   label=r['label'], zorder=3)
axs[2].set_xlim(0, lim); axs[2].set_ylim(0, lim); axs[2].set_aspect('equal')
axs[2].set_xlabel(f"near-source P(neutral), {ref['name']}")
axs[2].set_ylabel('near-source P(neutral), other model')
axs[2].legend(frameon=False, fontsize=8.5, loc='upper left')
for ax in axs:
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('ew_model_posteriors.png', dpi=170)

n = len(runs)
fig, axs = plt.subplots(2, n, figsize=(4.4 * n, 7), squeeze=False, layout='constrained')
vmax = max(r['m'].max() for r in runs)
dmax = max(np.abs(r['m'] - ref['m']).max() for r in runs[1:])
for j, r in enumerate(runs):
    im = axs[0, j].imshow(r['m'], aspect='auto', cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
    axs[0, j].set_title(r['label'], fontsize=10, color=r['c'])
    if j == 0:
        axs[1, j].axis('off')
        axs[1, j].text(0.5, 0.5, 'reference', ha='center', va='center', color=MUTED, transform=axs[1, j].transAxes)
        continue
    imd = axs[1, j].imshow(r['m'] - ref['m'], aspect='auto', cmap='RdBu_r', vmin=-dmax, vmax=dmax,
                           interpolation='nearest')
    axs[1, j].set_title(f"minus {ref['name']}", fontsize=10)
for ax in axs.flat:
    if ax.axison:
        ax.set_xlabel('LOS bin (0 = galaxy)'); ax.set_ylabel('galaxy')
fig.colorbar(im, ax=axs[0, :].tolist(), label='P(neutral)')
fig.colorbar(imd, ax=axs[1, :].tolist(), label='Δ P(neutral)')
fig.savefig('ew_model_maps.png', dpi=170, bbox_inches='tight')
print('saved ew_model_posteriors.png, ew_model_maps.png')

# ---------------- Figure 3: maps on a redshift axis ----------------
if z_gal is not None:
    from astropy.cosmology import Planck18 as Cosmo
    n_gal, n_los = ref['m'].shape
    if len(z_gal) != n_gal:
        sys.exit(f"catalog has {len(z_gal)} redshifts, posterior has {n_gal} rows")
    Z_END = 5.3
    zg = np.linspace(Z_END - 0.1, z_gal.max() + 0.1, 4000)          # d_C(z) table, inverted by interp
    dg = Cosmo.comoving_distance(zg).value
    d_gal = np.interp(z_gal, zg, dg)
    width = (d_gal - np.interp(Z_END, zg, dg)) / n_los                 # bin width per galaxy [cMpc]
    d_edges = d_gal[:, None] - np.arange(n_los + 1)[None, :] * width[:, None]   # (n_gal, n_los + 1)
    z_edges = np.interp(d_edges, dg, zg)                               # edge 0 = galaxy, edge n_los = z_end
    print(f"bin widths {width.min():.2f}-{width.max():.2f} cMpc")

    def zmap(ax, m, **kw):
        for g in range(n_gal):     # one row per galaxy, each with its own redshift edges
            qm = ax.pcolormesh(z_edges[g][::-1], [g - 0.5, g + 0.5], m[g][::-1][None, :],
                               shading='flat', rasterized=True, **kw)
        ax.plot(z_gal, np.arange(n_gal), 'o', ms=2.2, color=INK, zorder=3)   # galaxy positions
        ax.set_xlim(zmin, z_gal.max() + 0.02); ax.set_ylim(n_gal - 0.5, -0.5)
        yt = np.linspace(0, n_gal - 1, 6).round().astype(int)
        ax.set_yticks(yt); ax.set_yticklabels([f'{z_gal[t]:.2f}' for t in yt])
        ax.set_xlabel('redshift'); ax.set_ylabel('galaxy (labelled by its z)')
        return qm

    fig, axs = plt.subplots(2, n, figsize=(4.6 * n, 7), squeeze=False, layout='constrained')
    for j, r in enumerate(runs):
        im = zmap(axs[0, j], r['m'], cmap='Blues', vmin=0, vmax=vmax)
        axs[0, j].set_title(r['label'], fontsize=10, color=r['c'])
        if j == 0:
            axs[1, j].axis('off')
            axs[1, j].text(0.5, 0.5, 'reference', ha='center', va='center', color=MUTED,
                           transform=axs[1, j].transAxes)
            continue
        imd = zmap(axs[1, j], r['m'] - ref['m'], cmap='RdBu_r', vmin=-dmax, vmax=dmax)
        axs[1, j].set_title(f"minus {ref['name']}", fontsize=10)
    fig.colorbar(im, ax=axs[0, :].tolist(), label='P(neutral)')
    fig.colorbar(imd, ax=axs[1, :].tolist(), label='Δ P(neutral)')
    fig.savefig('ew_model_maps_z.png', dpi=170, bbox_inches='tight')
    print('saved ew_model_maps_z.png')
