"""Plot the mock test of exact_vs_nn_mocks.py: exact-likelihood vs network posteriors against the known truth.
numpy / matplotlib / astropy only. Only pixels at z > zmin are used (default 6.8: the redshifts where the
Lya data are informative; farther pixels follow the prior for both methods).

Figure exact_vs_nn_mocks.png
 (a) reliability: pixels binned by predicted P(neutral); y = fraction truly neutral. A correct posterior
     lies on the diagonal. Marker area ~ number of pixels.
 (b) proper scores relative to the prior (skill = 1 - score / score_prior; higher is better): log-loss and
     Brier. The true posterior has the best expected score, so the exact likelihood must not lose to the
     network. Log-loss clips P to [1e-3, 1 - 1e-3] (a finite pool can give P = 0 exactly).
 (c) per mock: mean |P_network - P_exact| over the z > zmin pixels; the real observation is marked if
     real=exact_attribution.npz:pixel_infer.npz is given (computed with the full 10k pool there).
 (d) per mock: log-loss(network) - log-loss(exact); positive = exact closer to the truth.
Figure exact_vs_nn_mocks_examples.png: truth / exact / network for the mocks at the 50th, 90th percentile
and the maximum of the disagreement (c), on a redshift axis.

Usage: python plot_exact_vs_nn_mocks.py exact_vs_nn_mocks.npz [zmin=6.8] [real=<exact.npz>:<pixel_infer.npz>]
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as Cosmo

path = sys.argv[1]
opts = dict(a.split('=', 1) for a in sys.argv[2:])
zmin = float(opts.get('zmin', 6.8))
d = np.load(path)
n_gal, n_los = int(d['n_gal']), int(d['n_los'])
truth, pe, pn = d['true_grid'].astype(float), d['p_exact'].astype(float), d['p_nn'].astype(float)
prior = np.broadcast_to(d['prior_p'].astype(float), truth.shape)
z_gal, nq = d['redshifts'], len(truth)
Z_END, INK, MUTED = 5.3, '#0b0b0b', '#8a8986'
C_EX, C_NN = '#2a78d6', '#d81b60'

# ---- bin redshifts (equal comoving width from the galaxy down to z_end, as lightcone_field.bin_z_edges) ----
zg = np.linspace(Z_END - 0.1, z_gal.max() + 0.1, 4000)
dg = Cosmo.comoving_distance(zg).value
d_gal = np.interp(z_gal, zg, dg)
width = (d_gal - np.interp(Z_END, zg, dg)) / n_los
z_edges = np.interp(d_gal[:, None] - np.arange(n_los + 1)[None, :] * width[:, None], dg, zg)
z_mid = 0.5 * (z_edges[:, 1:] + z_edges[:, :-1])
sel = z_mid > zmin                                            # (n_gal, n_los) pixels used
print(f'{sel.sum()} of {sel.size} pixels per mock have z > {zmin}')


def logloss(p, t):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return -(t * np.log(p) + (1 - t) * np.log(1 - p))


T, PE, PN, PP = truth[:, sel], pe[:, sel], pn[:, sel], prior[:, sel]  # (nq, n_pix)
scores = {}
for name, P in (('prior', PP), ('network', PN), ('exact', PE)):
    scores[name] = (logloss(P, T).mean(), ((P - T) ** 2).mean())
print(f"{'':>8s} {'log-loss':>9s} {'Brier':>8s}")
for k, (ll, br) in scores.items():
    print(f'{k:>8s} {ll:9.4f} {br:8.4f}')

dis = np.abs(PN - PE).mean(axis=1)                            # per-mock mean disagreement
dll = logloss(PN, T).mean(axis=1) - logloss(PE, T).mean(axis=1)
print(f'mean |P_nn - P_exact| per mock: median {np.median(dis):.3f}, 90% {np.percentile(dis, 90):.3f}, '
      f'max {dis.max():.3f}; exact has lower log-loss in {np.mean(dll > 0):.0%} of mocks')

real = None
if 'real' in opts:
    re_path, rn_path = (os.path.expanduser(s) for s in opts['real'].split(':'))
    real = np.abs(np.load(rn_path)['marginal_map'] - np.load(re_path)['p_full'])[sel].mean()
    print(f'real data: mean |P_nn - P_exact| = {real:.3f} (percentile among mocks: {np.mean(dis < real):.0%})')

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(1, 4, figsize=(17, 4), layout='constrained')
edges = np.linspace(0, 1, 11)
for P, col, lab in ((PE, C_EX, 'exact'), (PN, C_NN, 'network')):
    idx = np.clip(np.digitize(P.ravel(), edges) - 1, 0, 9)
    n = np.bincount(idx, minlength=10)
    frac = np.bincount(idx, weights=T.ravel(), minlength=10) / np.maximum(n, 1)
    pm = np.bincount(idx, weights=P.ravel(), minlength=10) / np.maximum(n, 1)
    ok = n > 50
    ax[0].plot(pm[ok], frac[ok], '-', color=col, lw=1.5, label=lab)
    ax[0].scatter(pm[ok], frac[ok], s=8 + 60 * n[ok] / n.max(), color=col, zorder=3)
ax[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls='--')
ax[0].set(xlabel='predicted P(neutral)', ylabel='fraction truly neutral', title=f'(a) reliability, z > {zmin}')
ax[0].legend(frameon=False)

for i, (k, j) in enumerate((('log-loss', 0), ('Brier', 1))):
    for o, (name, col) in enumerate((('network', C_NN), ('exact', C_EX))):
        skill = 1 - scores[name][j] / scores['prior'][j]
        ax[1].bar(i + (o - 0.5) * 0.36, skill, width=0.34, color=col, label=name if i == 0 else None)
        ax[1].text(i + (o - 0.5) * 0.36, skill, f'{skill:.3f}', ha='center', va='bottom', fontsize=8)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['log-loss', 'Brier'])
ax[1].set(ylabel='skill vs prior (higher = better)', title='(b) proper scores vs truth'); ax[1].legend(frameon=False)

ax[2].hist(dis, bins=30, color=MUTED)
if real is not None:
    ax[2].axvline(real, color=INK, lw=1.5); ax[2].text(real, ax[2].get_ylim()[1] * 0.9, ' real data', fontsize=9)
ax[2].set(xlabel='mean |P_network − P_exact|', ylabel='mocks', title='(c) disagreement per mock')

ax[3].hist(dll, bins=30, color=MUTED); ax[3].axvline(0, color=INK, lw=0.8)
ax[3].set(xlabel='log-loss(network) − log-loss(exact)', ylabel='mocks',
          title=f'(d) exact closer to truth in {np.mean(dll > 0):.0%} of mocks')
for a in ax:
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
fig.savefig('exact_vs_nn_mocks.png', dpi=160)

# ---- examples on a redshift axis ----
pick = [int(np.argsort(dis)[nq // 2]), int(np.argsort(dis)[int(0.9 * nq)]), int(np.argmax(dis))]
fig, axs = plt.subplots(3, 3, figsize=(13, 11), layout='constrained')
for r, qi in enumerate(pick):
    for c, (m, t) in enumerate(((truth[qi], 'truth'), (pe[qi], f'exact (ESS {d["ess_exact"][qi]:.0f})'),
                                (pn[qi], f'network (ESS {d["ess_nn"][qi]:.0f})'))):
        a = axs[r, c]
        for g in range(n_gal):
            qm = a.pcolormesh(z_edges[g][::-1], [g - 0.5, g + 0.5], m[g][::-1][None, :], shading='flat',
                              cmap='Blues', vmin=0, vmax=1, rasterized=True)
        ul = d['x_q'][qi, n_gal:] == 1
        a.scatter(z_gal, np.arange(n_gal), s=9, c=np.where(ul, 'white', INK), edgecolors=INK, lw=0.6, zorder=3)
        a.set_xlim(zmin, z_gal.max() + 0.02); a.set_ylim(n_gal - 0.5, -0.5)
        yt = np.linspace(0, n_gal - 1, 6).round().astype(int)
        a.set_yticks(yt); a.set_yticklabels([f'{z_gal[k]:.2f}' for k in yt]); a.set_xlabel('redshift')
        a.set_title(f'mock {int(d["query_idx"][qi])}: {t}' + (f'\nmean |ΔP| = {dis[qi]:.3f}' if c == 0 else ''),
                    fontsize=9)
fig.colorbar(qm, ax=axs.ravel().tolist(), shrink=0.5, label='P(neutral)  (truth: 1 = neutral)')
fig.savefig('exact_vs_nn_mocks_examples.png', dpi=140)
print('saved exact_vs_nn_mocks.png, exact_vs_nn_mocks_examples.png')
