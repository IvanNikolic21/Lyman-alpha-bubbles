"""Where along the sightlines does Lya actually constrain the ionization field? numpy/matplotlib only.

Inputs: pixel_sbc.npz from sbi_pixel_sbc.py (depth-sweep version), and optionally the real-data
posterior pixel_infer.npz (marginal_map + joint_samples) for the REALISED curve.

Calibration (TARP) is necessary but not sufficient: a prior-like posterior is perfectly calibrated.
"Precise" = calibrated AND informative ("maximize sharpness subject to calibration"). Information
per sightline bin k, pooled over galaxies:

  uncertainty coefficient  U_k = I(theta_k; x) / H(theta_k)                     [0 = prior, 1 = known]
      I = H(prior) - E[log-loss of posterior P(neutral) vs truth]   (expected; a LOWER bound on the
      mutual information if the posterior is miscalibrated)
      truth-free variant: H(prior) - E[H(posterior)]  (equals I for a calibrated posterior;
      the only version available for real data, where the truth is unknown)
  AUC_k   P(posterior ranks a truly neutral pixel above a truly ionized one); 0.5 = no information
  contraction_k  1 - var_post / var_prior of the across-galaxy neutral fraction in bin k

Expected curves average over the prior (held-out simulations): where the network is precise for
TYPICAL data. The realised curve uses the actual GOODS-N posterior: where THIS data set is informative.

Usage: python plot_pixel_sbc_depth.py pixel_sbc.npz [pixel_infer.npz]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'pixel_sbc.npz'
real_path = sys.argv[2] if len(sys.argv) > 2 else None
U_MIN = 0.05                # uncertainty coefficient below this = "no better than the prior"
EPS = 1e-4
d = np.load(path)
need = ('tarp_cum', 'tarp_shell', 'post_p', 'true_grid', 'prior_p_bin')
if not all(k in d.files for k in need):
    sys.exit(f"{path} has no depth-sweep arrays; rerun sbi_pixel_sbc.py (version with N_SWEEP).")


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def auc(score, label):
    """Mann-Whitney AUC with average ranks for ties."""
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


n = int(d['n_queries']); n_cut = d['n_sweep']; n_q, n_gal, n_los = d['post_p'].shape
bw = float(np.mean(d['bin_width']))                       # mean comoving bin width [cMpc]
r_bin = (np.arange(n_los) + 0.5) * bw                     # bin-centre distance from the galaxy
r_cut = n_cut * bw
r_shell_mid = 0.5 * (np.concatenate([[0], n_cut[:-1]]) * bw + r_cut)

# ---------------- expected information (held-out simulations) ----------------
t = d['true_grid'].astype(bool); p = np.clip(d['post_p'].astype(np.float64), EPS, 1 - EPS)
p0 = d['prior_p_bin']
H0 = entropy(p0)                                                          # (n_los,) bits
logloss = -(t * np.log2(p) + (~t) * np.log2(1 - p)).mean(axis=(0, 1))
U_exp = (H0 - logloss) / H0                                               # truth-based (lower bound)
U_exp_ent = (H0 - entropy(p).mean(axis=(0, 1))) / H0                      # truth-free
AUC = np.array([auc(p[:, :, k].ravel(), t[:, :, k].ravel()) for k in range(n_los)])
contr_exp = 1 - (d['post_std_bin'] ** 2).mean(0) / np.maximum(d['prior_std_bin'] ** 2, 1e-12)
cov68 = np.mean(np.abs(d['ranks_bin'] - 0.5) < 0.34, axis=0)

# ---------------- realised information (GOODS-N posterior) ----------------
U_real = contr_real = None
if real_path:
    rr = np.load(real_path)
    if (int(rr['n_gal']), int(rr['n_los'])) != (n_gal, n_los):
        sys.exit(f"{real_path} is {int(rr['n_gal'])}x{int(rr['n_los'])}, SBC is {n_gal}x{n_los}.")
    U_real = (H0 - entropy(rr['marginal_map']).mean(axis=0)) / H0
    col = rr['joint_samples'].mean(axis=1)                                # (n_samp, n_los)
    contr_real = 1 - col.var(0) / np.maximum(d['prior_std_bin'] ** 2, 1e-12)
    print(f"realised posterior: {real_path} (ESS {float(rr['ess']):.0f})")

# ---------------- calibration vs depth (TARP) ----------------
alpha = np.linspace(0, 1, 101)
def tarp_dev(f):
    return np.abs(np.array([np.mean(f < a) for a in alpha]) - alpha).max()
dev_cum = np.array([tarp_dev(d['tarp_cum'][:, j]) for j in range(len(n_cut))])
dev_shell = np.array([tarp_dev(d['tarp_shell'][:, j]) for j in range(len(n_cut))])
rng = np.random.default_rng(0)
null = np.array([tarp_dev(rng.random(n)) for _ in range(1000)])
band, band_k = np.quantile(null, 0.95), np.quantile(null, 1 - 0.05 / len(n_cut))

# ---------------- figure ----------------
INK, BAND, C1, C2, C3 = '#000000', '#d9dce3', '#2a78d6', '#d81b60', '#1a9e77'
plt.rcParams.update({'font.size': 11})
fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True,
                                 gridspec_kw={'height_ratios': [2.2, 1.5, 1.0]})

a0.axhline(0, color=INK, lw=0.6)
a0.plot(r_bin, U_exp, color=C1, lw=2.2, label='expected (log-loss vs truth)')
a0.plot(r_bin, U_exp_ent, color=C1, lw=1.2, ls='--', label='expected (posterior entropy)')
if U_real is not None:
    a0.plot(r_bin, U_real, color=C2, lw=2.2, label='realised, GOODS-N (posterior entropy)')
a0.axhline(U_MIN, color=INK, lw=0.6, ls=':')
inf = U_exp >= U_MIN
r_inf = r_bin[np.argmin(inf)] if not inf.all() else r_bin[-1]
if inf[0]:
    a0.axvspan(0, r_inf, color=C1, alpha=0.06, lw=0)
    a0.text(r_inf, U_MIN + 0.03, f' informative to ~{r_inf:.0f} cMpc', color=C1, fontsize=9, va='bottom')
a0.set_ylabel('fraction of prior uncertainty\nremoved per pixel  $I/H$')
a0.set_title('What Lya constrains along the sightline', fontsize=11)
a0.legend(frameon=False, fontsize=9, loc='upper right')

ax2 = a1.twinx()
a1.plot(r_bin, contr_exp, color=C3, lw=2, label='contraction, expected')
if contr_real is not None:
    a1.plot(r_bin, contr_real, color=C3, lw=1.5, ls='--', label='contraction, realised')
ax2.plot(r_bin, AUC, color=INK, lw=1.5, label='AUC, expected')
ax2.axhline(0.5, color=INK, lw=0.6, ls=':')
a1.set_ylabel(r'$1 - \sigma^2_{\rm post}/\sigma^2_{\rm prior}$', color=C3)
ax2.set_ylabel('AUC  (0.5 = none)')
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc='lower left')

a2.axhspan(0, band_k, color=BAND, alpha=0.5, lw=0); a2.axhspan(0, band, color=BAND, lw=0)
a2.plot(r_cut, dev_cum, 'o-', color=C1, lw=1.5, ms=4, label='first N bins')
a2.plot(r_shell_mid, dev_shell, 's--', color=C2, lw=1.2, ms=4, label='shells')
a2.set_ylabel('TARP\nmax |ECP − α|'); a2.legend(frameon=False, fontsize=9, ncol=2, loc='upper right')
a2.set_xlabel('comoving distance from galaxy along the sightline [cMpc]')

for ax in (a0, a1, a2):
    ax.set_xscale('log'); ax.set_xlim(0.4 * bw, n_los * bw * 1.1)
    for sp in ('top', 'right'):
        if not (ax is a1 and sp == 'right'):
            ax.spines[sp].set_visible(False)
ax2.spines['top'].set_visible(False)
fig.tight_layout(); fig.savefig('pixel_sbc_depth.png', dpi=170)

print(f"mean bin width {bw:.2f} cMpc; TARP band {band:.3f} (single), {band_k:.3f} (family-wise, k={len(n_cut)})")
hdr = f"{'bins':>7s} {'r[cMpc]':>8s} {'U_exp':>6s} {'U_ent':>6s} {'U_real':>6s} {'AUC':>5s} {'contr':>6s} {'cov68':>6s} {'TARPsh':>7s}"
print(hdr)
for j, N in enumerate(n_cut):
    lo = 0 if j == 0 else n_cut[j - 1]; sl = slice(lo, N)
    ur = f"{U_real[sl].mean():6.3f}" if U_real is not None else f"{'-':>6s}"
    print(f"{lo:3d}-{N - 1:<3d} {r_cut[j]:8.1f} {U_exp[sl].mean():6.3f} {U_exp_ent[sl].mean():6.3f} {ur} "
          f"{np.nanmean(AUC[sl]):5.2f} {contr_exp[sl].mean():6.2f} {cov68[sl].mean():6.2f} "
          f"{dev_shell[j]:6.3f}{'*' if dev_shell[j] > band_k else ' '}")
print(f"U_exp >= {U_MIN} out to ~{r_inf:.0f} cMpc  (* = TARP shell outside family-wise band)")
print('saved pixel_sbc_depth.png')
