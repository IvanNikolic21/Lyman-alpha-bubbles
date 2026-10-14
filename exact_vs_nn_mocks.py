"""
exact_vs_nn_mocks.py -- is the exact pixel likelihood correct, and how close is the network to it?
Tested on MOCKS, where the true ionization map is known.

WHAT IT DOES
------------
sbi_pixel_sbc.py drew n_queries mock observations (theta*_q, x_q) from the val split and, for each, stored
the network's posterior P(neutral) map (post_p), computed by importance weighting the REMAINING val maps
(the "pool"; the queries are excluded from it). Here we compute the exact-likelihood posterior for the
same queries with the same pool:

    w_qi  ~  exp( sum_g l_g(theta_i; x_q) ),      P_exact,q = sum_i w_qi theta_i,

with l_g = log p(x_g | theta_g) from exact_pixel_likelihood.loglik_quadrature (numerical integration over
the intrinsic EW and the velocity offset; no Monte Carlo noise).
Saved: the true maps, the exact and the network posterior maps, the exact ESS, for plot_exact_vs_nn_mocks.py.

WHY THIS TESTS CORRECTNESS
--------------------------
For mocks, theta*_q is drawn from the prior and x_q from the simulator. If l_g is the simulator's true
likelihood, P_exact is the true posterior (up to the finite pool), and then
 (a) it is CALIBRATED: among pixels with P_exact ~ p, a fraction p is truly neutral;
 (b) no other method can beat it on average under a proper score (log-loss, Brier): the true posterior
     minimises the expected score. If the network scored better, the exact likelihood would be wrong.
Neither test uses the real data.

COST
----
Per galaxy, the transmission matrix T (n_pool x n_dv) is computed once. Upper limits: one threshold per
galaxy (3 sigma_g, catalog sigma) -> one log-likelihood vector shared by all queries. Detections: one
quadrature (n_pool x n_dv x n_y) per query and detected galaxy (~5 per mock). 200 queries: a few minutes.

Usage
-----
python exact_vs_nn_mocks.py --sbc pixel_sbc.npz --pool_dir ~/Documents/Lya_bubblesx --ew_model exponential \\
    --n_queries 200 --output exact_vs_nn_mocks_exponential.npz
The --pool_dir x must have been simulated with --ew_model, and --sbc must come from the network trained on
it (only exponential is available locally).
"""
import os
import argparse

import numpy as np

import exact_pixel_likelihood as epl     # registers the lyabubbles stub; GalaxyModel, loglik_from_mu, ...


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sbc', required=True, help='pixel_sbc.npz of the network (query_idx, post_p, true_grid).')
    ap.add_argument('--pool_dir', required=True, help='the (theta, x) dataset the SBC was run on.')
    ap.add_argument('--pool_split', default='val', choices=['train', 'val'])
    ap.add_argument('--ew_model', required=True, choices=list(epl.EW_MODEL_FLAGS))
    ap.add_argument('--n_queries', type=int, default=200, help='first N SBC queries to compute.')
    ap.add_argument('--n_dv', type=int, default=32, help='velocity-offset quantiles (loglik_quadrature).')
    ap.add_argument('--n_y', type=int, default=64, help='cells of the EW integral (loglik_quadrature).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lya_catalog', default=os.path.join(epl.REPO, 'tb_lya.txt'))
    ap.add_argument('--properties_catalog', default=os.path.join(epl.REPO, 'sample_nirspec_properties.txt'))
    ap.add_argument('--prefer', default='grating', choices=['grating', 'prism'])
    ap.add_argument('--z_hi', type=float, default=7.3)
    ap.add_argument('--z_min', type=float, default=5.0)
    ap.add_argument('--muv_max', type=float, default=-18.0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    # ---- the SBC queries and the pool they were evaluated against (sbi_pixel_sbc.py lines ~159-164) ----
    sbc = np.load(os.path.expanduser(args.sbc))
    theta_all, x_all, n_gal, n_los = epl.load_pool(os.path.expanduser(args.pool_dir), args.pool_split)
    q_all = sbc['query_idx']
    pool_mask = np.ones(len(theta_all), bool); pool_mask[q_all] = False      # pool = val minus ALL queries
    theta_pool = theta_all[pool_mask].reshape(-1, n_gal, n_los).astype(np.float64)
    n_pool = len(theta_pool)
    if n_pool != int(sbc['n_pool']):
        raise ValueError(f"pool has {n_pool} maps, the SBC used {int(sbc['n_pool'])}")
    nq = min(args.n_queries, len(q_all))
    q = q_all[:nq]
    true_grid = theta_all[q].reshape(nq, n_gal, n_los)
    if not np.array_equal(true_grid, sbc['true_grid'][:nq]):
        raise ValueError('--pool_dir maps do not match the SBC truths: wrong dataset or network')
    x_q = x_all[q]                                            # (nq, 2 n_gal): EWs, upper-limit flags
    ew_q, ul_q = x_q[:, :n_gal].astype(float), x_q[:, n_gal:] == 1

    # ---- catalog (sigma_g, Muv, z) and the galaxy model ----
    cat = epl.load_real_catalog(args.lya_catalog, args.properties_catalog, args.z_min, args.muv_max,
                                args.prefer, args.z_hi)
    if len(cat['muv']) != n_gal:
        raise ValueError(f"catalog has {len(cat['muv'])} galaxies, pool {n_gal}")
    sigma = cat['sigma']
    model = epl.GalaxyModel(cat, args.ew_model, 1, args.seed)   # only its deterministic methods are used
    print(f"[mocks] {nq} queries, pool {n_pool} maps, {ul_q.size - ul_q.sum()} detections in total", flush=True)

    # ---- log-likelihood of every pool map for every query ----
    logL = np.zeros((nq, n_pool))
    for g in range(n_gal):
        T = model.transmission(g, theta_pool[:, g, :], n_los, args.n_dv)   # (n_pool, n_dv), query-independent
        dist = model.ew_distribution(g)
        if ul_q[:, g].any():                                   # upper limit: threshold 3 sigma_g, shared
            logL[ul_q[:, g]] += epl.loglik_quadrature(T, None, True, sigma[g], dist, args.n_y)[None, :]
        for qi in np.where(~ul_q[:, g])[0]:                    # detections: depends on the reported EW
            logL[qi] += epl.loglik_quadrature(T, ew_q[qi, g], False, sigma[g], dist, args.n_y)
        if (g + 1) % 10 == 0:
            print(f"[mocks] {g + 1}/{n_gal} galaxies", flush=True)

    p_exact = np.empty((nq, n_gal, n_los)); ess = np.empty(nq)
    for qi in range(nq):
        _, ess[qi], p_exact[qi] = epl.weights_and_map(logL[qi], theta_pool)
    np.savez(args.output, query_idx=q, true_grid=true_grid.astype(np.uint8), p_exact=p_exact.astype(np.float32),
             p_nn=sbc['post_p'][:nq], ess_exact=ess, ess_nn=sbc['ess'][:nq], x_q=x_q,
             prior_p=theta_pool.mean(axis=0).astype(np.float32), redshifts=cat['z_gal_red'],
             n_gal=n_gal, n_los=n_los, n_dv=args.n_dv, n_y=args.n_y, ew_model=args.ew_model)
    print(f"[mocks] saved {args.output}; exact ESS median {np.median(ess):.0f}, "
          f"network ESS median {np.median(sbc['ess'][:nq]):.0f}", flush=True)


if __name__ == '__main__':
    main()
