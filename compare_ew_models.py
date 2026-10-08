"""Compare real-data pixel-field posteriors obtained with different intrinsic Lya EW models
(sbi_runs/pixel_mock_<model>/pixel_infer.npz from submit_ew_model_runs.sh). numpy/matplotlib only.

The first entry is the reference (normally `exponential`). theta (bubble mocks) is identical across
runs, so the prior is the same and any difference is due to the EW model alone.

Figure 1 (ew_model_posteriors.png):
  a  posterior of the near-source neutral fraction (first N_NEAR bins of all sightlines; Lya-informed)
  b  posterior of the global neutral fraction (mostly prior structure, see sbi_pixel_sbc.py)
  c  per-galaxy near-source P(neutral): each model vs the reference
Figure 2 (ew_model_maps.png): marginal P(neutral) maps per model + difference to the reference.

Usage: python compare_ew_models.py exponential=path/pixel_infer.npz tang_muv=... gagnon_hartman=...
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
for i, arg in enumerate(sys.argv[1:]):
    name, path = arg.split('=', 1)
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
