"""
Summary statistics of a pixel-field map theta (n_gal x n_los, 1 = neutral) used for simulation-based calibration,
shared by sbi_pixel_sbc.py (network posterior) and exact_pixel_sbc.py (exact-likelihood posterior), so both test
exactly the same quantities. Pure numpy. Their meaning is explained in sbi_pixel_sbc.py's docstring.
"""
import numpy as np

N_NEAR = 5
SLAB_EDGES = (0, 19, 38, 57, 75)
N_SWEEP = (1, 2, 3, 5, 8, 12, 19, 28, 38, 57, 75)   # cumulative TARP blocks; shells between entries
R_V_CMPC = 2.0      # ~200 km/s Lya velocity offset at z~7 in comoving distance (softens 1/r^2 at r -> 0)


def dw_weights(bin_width_cmpc, n_los):
    """Damping-wing weights per galaxy and bin: w_k ~ 1/(r_k + R_V)^2, normalized per galaxy.
    bin_width_cmpc: (n_gal,) bin width of each galaxy's sightline (equal-comoving bins)."""
    r = (np.arange(n_los)[None, :] + 0.5) * bin_width_cmpc[:, None]
    w = 1.0 / (r + R_V_CMPC) ** 2
    return w / w.sum(axis=1, keepdims=True)                               # (n_gal, n_los)


def dw_per_galaxy(theta_grid, W):
    return (theta_grid * W[None]).sum(axis=2)                             # (n, n_gal)


def statistics(theta_grid, pairs, W):
    """theta_grid: (n, n_gal, n_los) binary (1 = neutral). Returns (n, n_stat) array
    (order = STAT_NAMES). Per-galaxy dw_g is returned separately by dw_per_galaxy()."""
    near = theta_grid[:, :, :N_NEAR].mean(axis=2)                      # (n, n_gal)
    cols = [dw_per_galaxy(theta_grid, W).mean(axis=1), near.mean(axis=1)]
    for a, b in zip(SLAB_EDGES[:-1], SLAB_EDGES[1:]):
        cols.append(theta_grid[:, :, a:b].mean(axis=(1, 2)))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        cols.append(1.0 - np.abs(near[:, i] - near[:, j]).mean(axis=1))
    else:
        cols.append(np.zeros(len(theta_grid)))
    cols.append(theta_grid.mean(axis=(1, 2)))                             # glob (prior-structure)
    return np.column_stack(cols)


STAT_NAMES = (['dw_mean', 'near'] + [f'slab_{k}' for k in range(len(SLAB_EDGES) - 1)]
              + ['pair_agree', 'glob'])
STAT_GROUP = {'dw_mean': 'lya', 'near': 'lya', 'pair_agree': 'lya', 'glob': 'prior',
              'slab_0': 'lya', 'slab_1': 'prior', 'slab_2': 'prior', 'slab_3': 'prior'}
