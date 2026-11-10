"""
los_grid.py -- the sightline bins of the pixel likelihood, defined ONCE for both sides of the calculation:

  * the pool (resample_pencil_pool.py): which cell of a mock's lightcone decides whether bin k is ionized
    (the cell at the bin CENTRE), and
  * the likelihood (exact_pixel_likelihood.GalaxyModel.transmission): which stretch of the sightline bin k
    covers when it is neutral (its two EDGES).

Every galaxy's sightline runs from the galaxy (distance 0) towards the observer, down to z = 5.3. A grid is given
by a short text "spec" and turned into bin edges in comoving distance FROM THE GALAXY [cMpc], increasing,
0 = the galaxy, last edge = the full sightline length:

  'uniform:N'        N equal-width bins (the 75-bin default is 'uniform:75', ~10 cMpc per bin)
  'varying:W,C,M'    bins of width W cMpc for the first C cMpc in front of the galaxy (where the damping wing is most
                     sensitive), then M bins whose edges are spaced evenly in log(distance) from C to the end, so
                     the bins widen steadily with distance (e.g. 'varying:2,30,50': 15 bins of 2 cMpc, then 50 bins
                     widening from ~2 to ~47 cMpc; 65 bins in total)

The number of bins is the same for every galaxy (the pool stores (n_mocks, n_galaxies, n_bins) arrays); only the
bin widths of the far, widening part differ slightly between galaxies, because their sightlines differ in length.

The OLD pools (make_pencil_pool.py) have no spec: they sample 75 points spaced evenly INCLUDING both ends (74 gaps
for 75 bins). Point k lies inside likelihood bin k, but drifts from the bin's galaxy-side edge (k = 0: exactly at
the galaxy) to its far edge (k = 74: exactly at z = 5.3) -- never at the centre. 'legacy:N' reproduces those points
exactly, for checks.
"""
import numpy as np


def parse(spec):
    """'uniform:300' -> ('uniform', [300.0]); 'varying:2,30,50' -> ('varying', [2.0, 30.0, 50.0])."""
    kind, _, numbers = spec.partition(':')
    values = [float(v) for v in numbers.split(',')] if numbers else []
    expected = {'uniform': 1, 'legacy': 1, 'varying': 3}
    if kind not in expected or len(values) != expected[kind]:
        raise ValueError(f"bad sightline grid spec {spec!r}: use 'uniform:N', 'legacy:N' or 'varying:W,C,M'")
    return kind, values


def n_bins(spec):
    """Number of bins of a grid (the same for every galaxy)."""
    kind, values = parse(spec)
    if kind == 'varying':
        width, core, n_far = values
        return int(round(core / width)) + int(n_far)
    return int(values[0])


def bin_edges(spec, length):
    """Bin edges [cMpc from the galaxy], increasing, (n_bins + 1,), for a sightline of `length` cMpc."""
    kind, values = parse(spec)
    if kind in ('uniform', 'legacy'):
        return np.linspace(0.0, length, int(values[0]) + 1)
    width, core, n_far = values
    n_core = int(round(core / width))
    if not np.isclose(n_core * width, core):
        raise ValueError(f"{spec!r}: the core length C must be a whole number of bin widths W")
    if length <= core:
        raise ValueError(f"{spec!r}: sightline of {length:.1f} cMpc is shorter than the core {core} cMpc")
    core_edges = np.linspace(0.0, core, n_core + 1)
    far_edges = core * (length / core) ** (np.arange(1, int(n_far) + 1) / n_far)     # log-spaced, ends at length
    edges = np.concatenate([core_edges, far_edges])
    edges[-1] = length                                       # exact end despite rounding
    return edges


def sample_distances(spec, length):
    """Where each bin's ionization state is read off the lightcone [cMpc from the galaxy], (n_bins,), nearest the
    galaxy first: the bin centres -- except 'legacy', which reproduces the old pools' points (spaced length / (N - 1),
    the first exactly at the galaxy, the last exactly at the end)."""
    kind, values = parse(spec)
    edges = bin_edges(spec, length)
    if kind == 'legacy':
        n = int(values[0])
        return length - np.linspace(0.0, length, n)[::-1]    # = old np.linspace(z_end, z_gal, n), from the galaxy
    return 0.5 * (edges[:-1] + edges[1:])
