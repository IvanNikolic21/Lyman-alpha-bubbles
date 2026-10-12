"""Plot the galaxy attribution of the pixel-field posterior (output of sbi_pixel_attribution.py).
numpy / matplotlib / astropy only.

influence_S(g', k) = P_full(g', k) - P_without_S(g', k): how much more neutral (red) or ionized (blue)
pixel k of sightline g' is BECAUSE of the observations of the removed set S.

Figure 1 (attribution_summary.png)
  left   one row per removal set (single galaxies sorted by redshift, then the --groups), x = REDSHIFT.
         Colour = influence averaged over all sightlines that pass through that redshift (sightline g'
         covers z_end = 5.3 ... z_g'). Marker = the removed galaxy's own redshift: filled = detection,
         open = upper limit.
  right  total influence of each set: sum over all pixels of |influence| (in units of pixels)
Figure 2 (attribution_maps.png): the full posterior map and the influence maps of the --groups (or, if
  none were run, of the most influential single galaxies), each galaxy's row drawn at its own bin
  redshifts.

Bin k of galaxy g spans comoving distance d_C(z_g) - [k, k+1] * Delta_g, Delta_g = (d_C(z_g) -
d_C(5.3)) / n_los (Planck18), the equal-comoving-width bins of lyabubbles.lightcone_field.bin_z_edges.

Usage: python plot_pixel_attribution.py pixel_attribution.npz [zmin] [label]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as Cosmo

path = sys.argv[1] if len(sys.argv) > 1 else 'pixel_attribution.npz'
zmin = float(sys.argv[2]) if len(sys.argv) > 2 else 6.6
label = sys.argv[3] if len(sys.argv) > 3 else ''
d = np.load(path)
n_gal, n_los = int(d['n_gal']), int(d['n_los'])
z_gal = d['redshifts']                                       # galaxy redshifts in posterior-row order
ew_obs, is_ul = d['x_obs'][:n_gal], d['x_obs'][n_gal:].astype(bool)
infl, labels, members = d['influence'], [str(s) for s in d['set_labels']], d['set_members']
Z_END = 5.3
INK, MUTED = '#0b0b0b', '#52514e'

# ---- redshift edges of every galaxy's bins ----
zg = np.linspace(Z_END - 0.1, z_gal.max() + 0.1, 4000)
dg = Cosmo.comoving_distance(zg).value                       # d_C(z) table, inverted by interpolation
d_gal = np.interp(z_gal, zg, dg)
width = (d_gal - np.interp(Z_END, zg, dg)) / n_los           # bin width per galaxy [cMpc]
z_edges = np.interp(d_gal[:, None] - np.arange(n_los + 1)[None, :] * width[:, None], dg, zg)

# ---- influence averaged over sightlines on a common redshift grid ----
z_grid = np.linspace(zmin, z_gal.max(), 400)
d_grid = np.interp(z_grid, zg, dg)
k_of = np.floor((d_gal[:, None] - d_grid[None, :]) / width[:, None]).astype(int)   # (n_gal, n_z) bin index
valid = (k_of >= 0) & (k_of < n_los)                         # sightline g' passes through that redshift
k_c = np.clip(k_of, 0, n_los - 1)
prof = np.array([np.where(valid, I[np.arange(n_gal)[:, None], k_c], np.nan) for I in infl])
prof = np.nanmean(prof, axis=1)                              # (n_sets, n_z)
total = np.abs(infl).sum(axis=(1, 2))                        # total influence of each set

n_sets = len(labels)
singles = [i for i in range(n_sets) if (members[i] >= 0).sum() == 1]
groups = [i for i in range(n_sets) if (members[i] >= 0).sum() > 1]
order = sorted(singles, key=lambda i: z_gal[members[i][0]]) + groups

plt.rcParams.update({'font.size': 10})
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 0.17 * n_sets + 2.2), sharey=True,
                             gridspec_kw={'width_ratios': [4, 1]}, layout='constrained')
vmax = np.nanmax(np.abs(prof[order]))
im = a0.pcolormesh(z_grid, np.arange(len(order)), prof[order],      # row centres; 'nearest' builds edges
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest', rasterized=True)
ytl = []
for row, i in enumerate(order):
    mem = members[i][members[i] >= 0]
    if len(mem) == 1:
        g = mem[0]
        a0.plot(z_gal[g], row, 'o', ms=4.5, mfc=INK if not is_ul[g] else 'white', mec=INK, mew=0.9)
        ytl.append(f"{g:2d}  z={z_gal[g]:.3f}  {'DET' if not is_ul[g] else 'UL<'}{ew_obs[g]:5.1f}")
    else:
        a0.plot(z_gal[mem], np.full(len(mem), row), '|', ms=7, color=INK)
        ytl.append(f"group {labels[i]}")
if groups:
    a0.axhline(len(singles) - 0.5, color=INK, lw=0.8)
a0.set_yticks(range(len(order))); a0.set_yticklabels(ytl, fontsize=7, family='monospace')
a0.set_ylim(len(order) - 0.5, -0.5); a0.set_xlim(zmin, z_gal.max() + 0.01)
a0.set_xlabel('redshift'); a0.set_ylabel('removed galaxy / group')
a0.set_title(f'{label}  influence of each galaxy, averaged over sightlines '
             '(red: its data make that redshift more neutral)', fontsize=10)
fig.colorbar(im, ax=a0, label='Δ P(neutral)', shrink=0.6)
a1.barh(range(len(order)), total[order], color=MUTED, height=0.7)
a1.set_xlabel('total |influence| [pixels]')
for ax in (a0, a1):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.savefig('attribution_summary.png', dpi=170)

# ---- Figure 2: maps on a redshift axis ----
show = groups if groups else sorted(singles, key=lambda i: -total[i])[:5]
fig, axs = plt.subplots(1, len(show) + 1, figsize=(4.2 * (len(show) + 1), 4.6), squeeze=False,
                        layout='constrained')
axs = axs[0]

def zmap(ax, m, **kw):
    for g in range(n_gal):                                   # one row per galaxy, own redshift edges
        qm = ax.pcolormesh(z_edges[g][::-1], [g - 0.5, g + 0.5], m[g][::-1][None, :], shading='flat',
                           rasterized=True, **kw)
    ax.scatter(z_gal, np.arange(n_gal), s=9, c=np.where(is_ul, 'white', INK), edgecolors=INK, lw=0.7,
               zorder=3)                                     # galaxies: filled = detection, open = UL
    ax.set_xlim(zmin, z_gal.max() + 0.02); ax.set_ylim(n_gal - 0.5, -0.5)
    yt = np.linspace(0, n_gal - 1, 6).round().astype(int)
    ax.set_yticks(yt); ax.set_yticklabels([f'{z_gal[t]:.2f}' for t in yt])
    ax.set_xlabel('redshift')
    return qm

qp = zmap(axs[0], d['p_full'], cmap='Blues', vmin=0, vmax=d['p_full'].max())
axs[0].set_title(f'{label} full posterior P(neutral)', fontsize=10); axs[0].set_ylabel('galaxy (labelled by z)')
fig.colorbar(qp, ax=axs[0], shrink=0.8)
dm = max(np.abs(infl[i]).max() for i in show)
for ax, i in zip(axs[1:], show):
    mem = members[i][members[i] >= 0]
    qd = zmap(ax, infl[i], cmap='RdBu_r', vmin=-dm, vmax=dm)
    ax.scatter(z_gal[mem], mem, s=40, facecolors='none', edgecolors='#d81b60', lw=1.4, zorder=4)
    ax.set_title(f'influence of {labels[i]}  (ESS {d["ess_minus"][i]:.0f})', fontsize=10)
fig.colorbar(qd, ax=axs[1:].tolist(), label='Δ P(neutral)', shrink=0.8)
fig.savefig('attribution_maps.png', dpi=170, bbox_inches='tight')

print(f"full-data ESS {float(d['ess_full']):.0f}; M = {int(d['n_draws'])} replacement draws per set")
print(f"{'set':>12s} {'total|infl|':>11s} {'max infl':>9s} {'min infl':>9s} {'ESS':>6s}")
for i in sorted(range(n_sets), key=lambda i: -total[i])[:15]:
    print(f"{labels[i]:>12s} {total[i]:11.2f} {infl[i].max():9.3f} {infl[i].min():9.3f} {d['ess_minus'][i]:6.0f}")
print('saved attribution_summary.png, attribution_maps.png')
