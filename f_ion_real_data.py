"""
Compute and plot the ionized fraction posterior from a real_data_run model
comparison result.

Usage
-----
python f_ion_real_data.py mc_real_data_zmin-7.30.npz
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from analyze_bias_test import f_ion_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='model-comparison .npz from real_data_run.py')
    parser.add_argument('--n_mc', type=int, default=50000)
    parser.add_argument('--out',  type=str, default=None)
    args = parser.parse_args()

    d = dict(np.load(args.npz, allow_pickle=False))
    prior_lo = d['prior_lo']
    prior_hi = d['prior_hi']

    models = []
    if 'posterior_samples' in d:
        models.append((1, d['posterior_samples'], d.get('logz', np.nan)))
    if 'posterior_samples_m2' in d:
        models.append((2, d['posterior_samples_m2'], d.get('logz_m2', np.nan)))
    if 'posterior_samples_m3' in d:
        models.append((3, d['posterior_samples_m3'], d.get('logz_m3', np.nan)))

    if not models:
        sys.exit('No posterior_samples found in the file.')

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {1: 'C0', 2: 'C1', 3: 'C2'}

    print(f'\nIonized fraction f_ion within the prior box:')
    print(f'  Volume: {np.prod(prior_hi[:3] - prior_lo[:3]):.1f} Mpc^3')
    print()

    for n_bub, post, logz in models:
        f_ions = f_ion_samples(post, prior_lo, prior_hi, n_bub,
                               n_mc=args.n_mc, seed=n_bub)
        med = np.median(f_ions)
        p16 = np.percentile(f_ions, 16)
        p84 = np.percentile(f_ions, 84)
        print(f'  M{n_bub}: f_ion = {med:.3f}  [{p16:.3f}, {p84:.3f}]  '
              f'(log Z = {float(logz):.2f})')

        ax.hist(f_ions, bins=40, density=True, alpha=0.55,
                color=colors[n_bub], label=f'M{n_bub}  median={med:.3f}')
        ax.axvline(med, color=colors[n_bub], lw=1.5)
        ax.axvline(p16, color=colors[n_bub], lw=0.8, ls='--')
        ax.axvline(p84, color=colors[n_bub], lw=0.8, ls='--')

    ax.set_xlabel('Ionized fraction $f_{\\rm ion}$')
    ax.set_ylabel('Posterior density')
    ax.set_title('Ionized fraction posterior — real data')
    ax.legend()

    out = args.out or args.npz.replace('.npz', '_f_ion.png')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'\nPlot saved to {out}')


if __name__ == '__main__':
    main()