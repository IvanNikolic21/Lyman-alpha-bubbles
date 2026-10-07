"""Plot SBC results of the pixel-field SBI (output of sbi_pixel_sbc.py). numpy/matplotlib only.

Figure 1 (pixel_sbc_ranks.png): per statistic, ECDF(rank) - uniform with a 95% simultaneous band,
plus 68%/95% coverage in the panel title.
Figure 2 (pixel_sbc_tarp.png): TARP expected coverage vs credibility level (diagonal = calibrated).

Reading the ECDF-difference panels:
  curve inside the band              -> calibrated
  S-shape: up then down (or vice versa) -> posterior too narrow / too wide (over/under-confident)
  one-sided bump above/below zero       -> biased (posterior systematically high / low)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'pixel_sbc.npz'
d = np.load(path)
ranks, names, n = d['ranks'], [str(s) for s in d['stat_names']], int(d['n_queries'])
groups = [str(g) for g in d['stat_groups']] if 'stat_groups' in d.files else ['?'] * len(names)
# per-galaxy damping-wing statistic: ranks pooled over galaxies x queries (Lya-informed, "one number")
if 'ranks_dw' in d.files:
    ranks = np.column_stack([np.full(n, np.nan), ranks])
    names, groups = ['dw_g (pooled)'] + names, ['lya'] + groups
order = sorted(range(len(names)), key=lambda i: groups[i] != 'lya')      # Lya-informed panels first
names, groups = [names[i] for i in order], [groups[i] for i in order]
ranks = ranks[:, order]
GROUP_LABEL = {'lya': 'Lya-informed', 'prior': 'prior structure (not constrained by Lya)', '?': ''}

INK, BAND, CURVE = '#000000', '#d9dce3', '#2a78d6'
plt.rcParams.update({'font.size': 11})

# 95% simultaneous band for ECDF - uniform via Monte Carlo of uniform ranks (same n, same grid)
grid = np.linspace(0, 1, 201)
rng = np.random.default_rng(0)
sim = np.array([np.searchsorted(np.sort(rng.random(n)), grid, side='right') / n - grid for _ in range(2000)])
dev = np.abs(sim).max(axis=1)
band = np.quantile(dev, 0.95)          # simultaneous (Kolmogorov-type) half-width, one statistic

k = len(names); ncol = 4; nrow = int(np.ceil(k / ncol))
band_k = np.quantile(dev, 1 - 0.05 / k)  # Bonferroni: family-wise 95% over all k statistics (PASS/FAIL gate)
fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow), squeeze=False)
for i, name in enumerate(names):
    ax = axs.flat[i]
    r = d['ranks_dw'].ravel() if name == 'dw_g (pooled)' else ranks[:, i]
    m = len(r)
    ecdf = np.searchsorted(np.sort(r), grid, side='right') / m - grid
    # bands scale with the number of ranks in this panel (pooled panel has n_gal x n_queries ranks,
    # NOT independent across galaxies of one instance -> its band is optimistic; read it as indicative)
    b1, bk = band * np.sqrt(n / m), band_k * np.sqrt(n / m)
    ax.fill_between(grid, -bk, bk, color=BAND, lw=0, alpha=0.5)
    ax.fill_between(grid, -b1, b1, color=BAND, lw=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(grid, ecdf, color=CURVE, lw=2)
    c68 = np.mean(np.abs(r - 0.5) < 0.34); c95 = np.mean(np.abs(r - 0.5) < 0.475)
    passed = np.abs(ecdf).max() <= bk       # family-wise gate; inner band = single-statistic 95%
    ax.set_title(f"{name}  {'PASS' if passed else 'FAIL'}   [{GROUP_LABEL[groups[i]].split(' (')[0]}]\n"
                 f"68%: {c68:.2f}   95%: {c95:.2f}", fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(-max(3 * band, 0.12), max(3 * band, 0.12))
    ax.set_xlabel('fractional rank'); ax.set_ylabel('ECDF - uniform')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in list(axs.flat)[k:]:
    ax.axis('off')
fig.suptitle(f'Pixel-field SBC: {n} held-out instances, {int(d["n_post"])} posterior samples each, '
             f'median ESS {np.median(d["ess"]):.0f}', fontsize=12)
fig.tight_layout(); fig.savefig('pixel_sbc_ranks.png', dpi=170)

# TARP
alpha = np.linspace(0, 1, 101)
ecp = np.array([np.mean(d['tarp_f'] < a) for a in alpha])
boot = np.array([[np.mean(rng.choice(d['tarp_f'], n) < a) for a in alpha] for _ in range(300)])
lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.fill_between(alpha, lo, hi, color=BAND, lw=0)
ax.plot([0, 1], [0, 1], color=INK, ls='--', lw=1)
ax.plot(alpha, ecp, color=CURVE, lw=2)
ax.set_xlabel('credibility level'); ax.set_ylabel('expected coverage (TARP)')
ax.set_title('TARP, Lya-informed block\n(first bins of every sightline)', fontsize=11)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig('pixel_sbc_tarp.png', dpi=170)
print(f'saved pixel_sbc_ranks.png, pixel_sbc_tarp.png; 95% band half-width {band:.4f} (single), {band_k:.4f} (family-wise, k={k})')
