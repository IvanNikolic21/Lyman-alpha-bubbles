"""
plot_pencil_prior.py -- figures for the pencil-beam prior (esf_lightcone.py + make_pencil_pool.py) and its
exact-likelihood posterior, compared with v2.

  concept    pencil_concept.png    one wide lightcone, the survey footprint carved out at several places, and the
                                   local neutral fraction x_HI(z) of each pencil vs the global history
  prior      pencil_prior.png      local (z >= 6.8) vs global neutral fraction, v2 vs pencils; example masks
  posterior  pencil_posterior.png  P(neutral) maps (x = redshift) and hyperparameter posteriors, v2 vs pencils,
                                   with and without the Lyman-alpha-forest cut z_end < 6.05; evidence

Usage (JWST_inference env):
python plot_pencil_prior.py concept
python plot_pencil_prior.py prior --v2_smoke ~/Downloads/v2_smoke
python plot_pencil_prior.py posterior --pencil ~/Documents/Lya_bubbles/esf_pencils/exact_pencil_exponential.npz \\
    --v2 ~/Downloads/exact_v2_exponential.npz
"""
import os
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import esf_lightcone as esf
import make_pencil_pool as mpp

# Data root: the laptop folder by default; on the cluster set   export LYA_DATA=/lustre/astro/ivannik/Lya_bubbles/esf
DATA_ROOT = os.path.expanduser(os.environ.get('LYA_DATA', '~/Documents/Lya_bubbles'))
LC_DIR = os.path.join(DATA_ROOT, 'esf_lightcones')
PEN_DIR = os.path.join(DATA_ROOT, 'esf_pencils')
ION, NEU = '#4a86c8', '#e8873a'                      # deck convention: ionized blue, neutral orange
C_V2, C_PEN = '#7a8194', '#c2410c'
CM_P = LinearSegmentedColormap.from_list('p', ['#1f5fa8', '#7fb0e6', '#e6e9ee', '#f2a66b', '#c2410c'])
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': ['Arial', 'DejaVu Sans']})


def geometry():
    return dict(np.load(os.path.join(LC_DIR, 'geometry.npz')))


def point_redshifts(g):
    ell = np.stack([np.linspace(float(g['z_end_offset']), zz, mpp.N_LOS) for zz in g['z_gal']])
    return esf.los_redshift(ell, float(g['z0']), float(g['z_mean']))              # (43, 75)


def wquant(x, w, qs):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c /= c[-1]
    return [x[o][min(np.searchsorted(c, q), len(x) - 1)] for q in qs]


# ---------------------------------------------------------------------------------------------------------------
def fig_concept(a):
    g = geometry()
    d = np.load(os.path.join(LC_DIR, 'lc_003.npz'))
    j = 1                                                       # M_min = 1e9
    logF, qtab, z_los, ell, keep, dx = d['logF'][j].astype(np.float32), d['qtab'][j], d['z_los'], d['ell'], \
        d['keep'], float(d['dx'])
    i7 = int(np.argmin(np.abs(z_los - 7)))
    q = np.linspace(0, 1, esf.N_QUANT)
    x7, alpha = 0.5, 0.0
    lz7 = -np.interp(x7, q, qtab[i7])
    ion = logF + alpha * np.log10((1 + z_los) / 8)[None, None] >= -lz7
    hist = esf.xhi_history(qtab, z_los, lz7, alpha)
    n_t = ion.shape[0]
    wx = int(np.ceil((g['x_gal'].max() - g['x_gal'].min()) / dx))
    wy = int(np.ceil((g['y_gal'].max() - g['y_gal'].min()) / dx))

    fig = plt.figure(figsize=(16, 8.2))
    ax = fig.add_axes([0.05, 0.47, 0.62, 0.46])
    sl = n_t // 2
    zk = z_los[keep]
    img = ion[:, sl, :][:, keep]
    ax.imshow(img, origin='lower', aspect='auto', cmap=ListedColormap([NEU, ION]), interpolation='nearest',
              extent=[0, keep.sum() * dx, 0, n_t * dx])
    ticks_z = [5.5, 6.0, 6.5, 7.0, 7.4]
    ell_k = ell[keep] - ell[keep][0]
    ax.set_xticks(np.interp(ticks_z, zk, ell_k))
    ax.set_xticklabels([f'{t}' for t in ticks_z])
    ax.set_xlabel('Redshift')
    ax.set_ylabel('transverse [cMpc]')
    ax.set_title(f'One wide lightcone ({n_t * dx:.0f} cMpc across), global $x_{{\\rm HI}}(z{{=}}7)$ = {x7}',
                 loc='left', fontsize=14)
    rng = np.random.default_rng(4)
    starts = rng.choice(np.arange(5, n_t - wx - 5), 4, replace=False)
    cols = ['#111111', '#6b21a8', '#0f766e', '#a16207']
    for s0, c in zip(starts, cols):
        ax.add_patch(plt.Rectangle((0, s0 * dx), keep.sum() * dx, wx * dx, fill=False, ec=c, lw=1.8))
    ax.invert_xaxis()

    # local histories of many pencils vs the global one
    ax2 = fig.add_axes([0.73, 0.47, 0.25, 0.46])
    for k in range(60):
        ox, oy = rng.integers(0, n_t - max(wx, wy), 2)
        loc = 1 - ion[ox:ox + wx, oy:oy + wy, :].mean(axis=(0, 1))
        ax2.plot(z_los[keep], loc[keep], color='#e0a27a', lw=0.7, alpha=0.6)
    for s0, c in zip(starts, cols):
        loc = 1 - ion[s0:s0 + wx, sl - wy // 2:sl + wy // 2, :].mean(axis=(0, 1))
        ax2.plot(z_los[keep], loc[keep], color=c, lw=1.6)
    ax2.plot(z_los[keep], hist[keep], color='k', lw=3, ls='--', label='global (whole lightcone)')
    ax2.plot([], [], color='#e0a27a', label='inside one 30 cMpc pencil')
    ax2.set_xlabel('Redshift')
    ax2.set_ylabel('neutral fraction $x_{\\rm HI}$')
    ax2.set_xlim(5.3, 7.4)
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, fontsize=11, loc='upper left')
    ax2.set_title('local history of each pencil', loc='left', fontsize=14)

    # the 43 sightlines of the four pencils drawn on the left (top row of masks)
    zp = point_redshifts(g)
    il = np.clip(np.round((np.stack([np.linspace(float(g['z_end_offset']), zz, mpp.N_LOS) for zz in g['z_gal']])
                           - ell[0]) / dx).astype(int), 0, len(ell) - 1)
    order = np.argsort(g['z_gal'])
    for k, (s0, c) in enumerate(zip(starts, cols)):
        axm = fig.add_axes([0.05 + k * 0.235, 0.07, 0.215, 0.29])
        ix = (np.floor((g['x_gal'] - g['x_gal'].min()) / dx).astype(int) + s0) % n_t
        iy = (np.floor((g['y_gal'] - g['y_gal'].min()) / dx).astype(int) + sl - wy // 2) % n_t
        m = ion[ix[:, None], iy[:, None], il]
        for r, gi in enumerate(order):
            zz = zp[gi]
            for b in range(mpp.N_LOS - 1):
                axm.fill_between([zz[b], zz[b + 1]], r - 0.45, r + 0.45, color=ION if m[gi, b] else NEU, lw=0)
        axm.set_xlim(7.35, 5.3)
        axm.set_ylim(-1, len(order))
        axm.set_yticks([])
        axm.set_xlabel('Redshift')
        for sp in axm.spines.values():
            sp.set_edgecolor(c)
        axm.spines['left'].set_visible(True)
        axm.set_title(f'pencil {k + 1}: our 43 sightlines', fontsize=12, color=c, loc='left')
    fig.text(0.05, 0.395, 'Carved pencil beams (same global history, different places in the lightcone):',
             fontsize=13)
    fig.savefig(a.out or 'pencil_concept.png', dpi=150)
    print('saved', a.out or 'pencil_concept.png')


# ---------------------------------------------------------------------------------------------------------------
def fig_prior(a):
    g = geometry()
    zp = point_redshifts(g)
    near = zp >= 6.8
    # v2: local vs global over the same points
    import bubble_prior_v2 as bp
    v2l, v2g = [], []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.v2_smoke), '*_batch_*.npz'))):
        m = np.load(f)
        for i in range(len(m['masks'])):
            v2l.append(1 - m['masks'][i][near].mean())
            v2g.append(bp.x_hi_of_z(zp[near], m['z_mid'][i], m['dz'][i]).mean())
    v2l, v2g = np.array(v2l), np.array(v2g)
    pl, pg, mm = [], [], []
    for f in sorted(glob.glob(os.path.join(PEN_DIR, 'mocks', '*_batch_*.npz')))[:5]:
        m = np.load(f)
        pl.append(m['xhi_local'])
        pg.append(m['xhi_global_local'])
        mm.append(m['m_min'])
    pl, pg, mm = map(np.concatenate, (pl, pg, mm))

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (l, gg, lab, c) in zip(axs[:2], [(v2l, v2g, 'v2 prior (bubbles drawn in the footprint)', C_V2),
                                             (pl, pg, 'pencil beams from wide lightcones', C_PEN)]):
        ax.scatter(gg, l, s=3, alpha=0.25, color=c, rasterized=True)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('global $x_{\\rm HI}$  (whole universe, same redshifts)')
        ax.set_ylabel('local $x_{\\rm HI}$ along our sightlines, $z \\geq 6.8$')
        ax.set_title(lab, fontsize=13, loc='left')
        ax.text(0.97, 0.05, f'scatter = {np.std(l - gg):.3f}', transform=ax.transAxes, ha='right', fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    b = np.linspace(-0.6, 0.6, 61)
    axs[2].hist(v2l - v2g, b, density=True, histtype='step', lw=2, color=C_V2, label='v2')
    for mv, ls in zip(esf.M_MIN, ['-', '--', ':']):
        axs[2].hist((pl - pg)[mm == mv], b, density=True, histtype='step', lw=2, ls=ls, color=C_PEN,
                    label=f'pencils, M$_{{\\min}}$ = 1e{np.log10(mv):.0f}')
    axs[2].set_xlabel('local $-$ global $x_{\\rm HI}$')
    axs[2].legend(frameon=False, fontsize=11)
    axs[2].set_title('how "messy" the pencil is', fontsize=13, loc='left')
    fig.tight_layout()
    fig.savefig(a.out or 'pencil_prior.png', dpi=150)
    print('saved', a.out or 'pencil_prior.png', f'v2 scatter {np.std(v2l - v2g):.3f}, pencil {np.std(pl - pg):.3f}')


# ---------------------------------------------------------------------------------------------------------------
def load_pencil_hyper(pen_npz, pen_dir=None):
    pen_dir = pen_dir or PEN_DIR
    files = [str(f) for f in pen_npz['pool_files']]
    keys = ('x_hi_7', 'z_end', 'tau_e', 'alpha', 'm_min', 'xhi_local', 'xhi_global_local', 'realization')
    out = {k: [] for k in keys}
    for f in files:
        m = np.load(os.path.join(pen_dir, 'mocks', f))
        for k in keys:
            out[k].append(m[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def fig_posterior(a):
    g = geometry()
    zp = point_redshifts(g)
    P = np.load(os.path.expanduser(a.pencil), allow_pickle=True)
    V = np.load(os.path.expanduser(a.v2), allow_pickle=True)
    h = load_pencil_hyper(P)
    w = P['w']
    L = P['L']
    cut = h['z_end'] < 6.05
    maps = {'v2': V['p_full'], 'pencils': P['p_full']}
    if a.pencil_forest:
        # dedicated pool whose histories all end at 5.3 <= z_end <= 6.05 (make_pencil_pool.py --z_end_max 6.05)
        Fz = np.load(os.path.expanduser(a.pencil_forest), allow_pickle=True)
        hc = load_pencil_hyper(Fz, os.path.dirname(os.path.expanduser(a.pencil_forest)))
        wc = Fz['w']
        maps['pencils, z_end < 6.05'] = Fz['p_full']
        lz_c = float(Fz['log_evidence'])
    else:
        # reweight the main pool (few effective samples if the data disfavour the cut)
        hc = h
        wc = np.where(cut, np.exp(L - L.max()), 0.0)
        wc /= wc.sum()
        acc = np.zeros(P['p_full'].shape)
        a0 = 0
        for f, n in zip(P['pool_files'], P['pool_counts']):
            th = np.load(os.path.join(PEN_DIR, 'pool', str(f)))['theta'].astype(np.float64)
            acc += (wc[a0:a0 + n] @ th).reshape(acc.shape)
            a0 += n
        maps['pencils, z_end < 6.05'] = acc
        lz_c = float(P['log_evidence']) + np.log(np.mean(np.exp(L - L.max()) * cut) / np.mean(np.exp(L - L.max()))) \
            - np.log(cut.mean()) if cut.any() else np.nan
    ess = lambda ww: 1 / np.sum(ww ** 2)

    order = np.argsort(g['z_gal'])
    fig = plt.figure(figsize=(17, 10))
    for k, (lab, mp) in enumerate(maps.items()):
        ax = fig.add_axes([0.05 + k * 0.315, 0.55, 0.29, 0.38])
        for r, gi in enumerate(order):
            zz = zp[gi][::-1]                                    # theta order: from the galaxy outwards
            for b in range(mpp.N_LOS - 1):
                ax.fill_between([zz[b], zz[b + 1]], r - 0.5, r + 0.5, color=CM_P(mp[gi, b] / 0.6), lw=0)
        ax.set_xlim(7.35, 5.3)
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Redshift')
        near = np.concatenate([mp[gi][zp[gi][::-1] >= 6.8] for gi in range(len(mp))]).mean()
        ax.set_title(f'{lab}\nmean P(neutral) at z ≥ 6.8: {near:.2f}', fontsize=13, loc='left')
    cax = fig.add_axes([0.975, 0.55, 0.008, 0.38])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CM_P, norm=plt.Normalize(0, 0.6)), cax=cax)
    cb.set_label('P(neutral)')

    # hyperparameter posteriors
    ax = fig.add_axes([0.05, 0.08, 0.27, 0.32])
    b = np.linspace(0, 1, 31)
    ax.hist(V['hyper_x_hi_7'], b, weights=V['w'], density=True, histtype='step', lw=2.2, color=C_V2, label='v2')
    ax.hist(h['x_hi_7'], b, weights=w, density=True, histtype='step', lw=2.2, color=C_PEN, label='pencils')
    ax.hist(hc['x_hi_7'], b, weights=wc, density=True, histtype='step', lw=2.2, ls='--', color=C_PEN,
            label='pencils, z_end<6.05')
    ax.hist(h['x_hi_7'], b, density=True, histtype='stepfilled', color='#eeeeee', zorder=0, label='prior')
    ax.set_xlabel('global $x_{\\rm HI}(z=7)$')
    ax.legend(frameon=False, fontsize=11)
    q = lambda x, ww: '{:.2f} [{:.2f}, {:.2f}]'.format(*np.array(wquant(x, ww, [0.5, 0.16, 0.84])))
    txt = (f"v2: {q(V['hyper_x_hi_7'], V['w'])}\npencils: {q(h['x_hi_7'], w)}\n"
           f"pencils, z_end<6.05: {q(hc['x_hi_7'], wc)}")
    ax.set_title(txt, fontsize=11, loc='left')

    ax = fig.add_axes([0.38, 0.08, 0.27, 0.32])
    d = h['xhi_local'] - h['xhi_global_local']
    b = np.linspace(-0.6, 0.6, 41)
    ax.hist(d, b, density=True, histtype='stepfilled', color='#eeeeee', label='prior')
    ax.hist(d, b, weights=w, density=True, histtype='step', lw=2.2, color=C_PEN, label='posterior')
    dc = hc['xhi_local'] - hc['xhi_global_local']
    ax.hist(dc, b, weights=wc, density=True, histtype='step', lw=2.2, ls='--', color=C_PEN,
            label='posterior, z_end<6.05')
    ax.set_xlabel('local $-$ global $x_{\\rm HI}$ around our galaxies ($z\\geq6.8$)')
    ax.legend(frameon=False, fontsize=11)
    ax.set_title(f"is our field more ionized than average?  posterior mean\n{np.sum(w * d):+.3f} (pencils), "
                 f"{np.sum(wc * dc):+.3f} (z_end<6.05); prior {d.mean():+.3f}", fontsize=11, loc='left')

    ax = fig.add_axes([0.71, 0.08, 0.27, 0.32])
    labs = [f'1e{np.log10(m):.0f}' for m in esf.M_MIN]
    pr = [np.mean(h['m_min'] == m) for m in esf.M_MIN]
    po = [w[h['m_min'] == m].sum() for m in esf.M_MIN]
    pc = [wc[hc['m_min'] == m].sum() for m in esf.M_MIN]
    xx = np.arange(3)
    ax.bar(xx - 0.27, pr, 0.25, color='#dddddd', label='prior')
    ax.bar(xx, po, 0.25, color=C_PEN, label='posterior')
    ax.bar(xx + 0.27, pc, 0.25, color=C_PEN, alpha=0.45, label='posterior, z_end<6.05')
    ax.set_xticks(xx)
    ax.set_xticklabels(labs)
    ax.set_xlabel('M$_{\\min}$ of ionizing sources [M$_\\odot$]  (larger = bigger bubbles)')
    ax.legend(frameon=False, fontsize=11)
    lz_p, lz_v = float(P['log_evidence']), float(V['log_evidence'])
    ax.set_title(f"ln Z: v2 {lz_v:.2f}, pencils {lz_p:.2f}, pencils+cut {lz_c:.2f}\n"
                 f"ESS: pencils {ess(w):.0f} / {len(w)}, with cut {ess(wc):.0f}", fontsize=11, loc='left')
    fig.savefig(a.out or 'pencil_posterior.png', dpi=150, bbox_inches='tight')
    print('saved', a.out or 'pencil_posterior.png')
    print(txt.replace('\n', ' | '), '| lnZ v2', lz_v, 'pencil', lz_p, 'pencil+cut', lz_c, 'ESS', ess(w), ess(wc))
    for m, a1, a2, a3 in zip(labs, pr, po, pc):
        print(f'  M_min {m}: prior {a1:.2f} post {a2:.2f} post+cut {a3:.2f}')
    print('  tau_e posterior', q(h['tau_e'], w), ' z_end posterior', q(h['z_end'], w))
    print(f"  local-global: pencils {np.sum(w * d):+.3f}, z_end<6.05 {np.sum(wc * dc):+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', choices=['concept', 'prior', 'posterior'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--v2_smoke', default='~/Downloads/v2_smoke')
    ap.add_argument('--pencil', default=os.path.join(PEN_DIR, 'exact_pencil_exponential.npz'))
    ap.add_argument('--pencil_forest', default=None, help='exact_pencil_forest_exponential.npz (optional)')
    ap.add_argument('--v2', default='~/Downloads/exact_v2_exponential.npz')
    a = ap.parse_args()
    {'concept': fig_concept, 'prior': fig_prior, 'posterior': fig_posterior}[a.which](a)


if __name__ == '__main__':
    main()
