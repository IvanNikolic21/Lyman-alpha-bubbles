"""
Toy/calibration demo for the pixel-field NRE: pick N known mock (theta, x)
pairs, treat each one's OWN x as if it were an observation, run the SAME
pool-based inference `sbi_pixel_field.py infer` uses, and save the result
alongside that example's TRUE theta for a side-by-side comparison.

Built in response to real feedback on a presentation of this pipeline:
"Have you got a toy example working that can help us build intuition? I'm
not sure we should look at the real data until we know it works on mock
data." -- this answers exactly that, with ground truth available for direct
comparison, which the real-data case never has.

Deliberately does NOT import sbi_pixel_field.py/real_data_run.py -- neither
the real catalog nor py21cmfast is needed here, only the already-trained
ratio_estimator.pt and the already-generated mock pool. Runs in whatever
env has torch/sbi (no py21cmfast requirement), lighter than `infer` itself.

For each of N_EXAMPLES chosen pool indices: the CANDIDATE POOL used for
inference EXCLUDES that example's own index (otherwise it could just
trivially match itself) -- inference is run against the other ~9,999 sims,
exactly as it would be for a genuinely new observation.

Usage
-----
python mock_recovery_demo.py \\
    --ratio sbi_runs/pixel_mock/ratio_estimator.pt \\
    --pool_dir bubble_mocks_run1_x \\
    --pool_split val \\
    --n_examples 3 \\
    --output_dir sbi_runs/pixel_mock
"""
import argparse
import glob
import os

import numpy as np


def load_sims_standalone(sims_dir, prefix):
    """Same schema as sbi_pixel_field.py's load_sims, reimplemented here to
    avoid importing that module (which pulls in real_data_run.py ->
    py21cmfast at import time, not needed for this script at all)."""
    paths = sorted(glob.glob(os.path.join(sims_dir, f"{prefix}_batch_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_batch_*.npz files found in {sims_dir}")
    thetas, xs = [], []
    n_gal = n_los = None
    for p in paths:
        d = np.load(p)
        thetas.append(d["theta"])
        xs.append(d["x"])
        n_gal, n_los = int(d["n_gal"]), int(d["n_los"])
    return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0), n_gal, n_los


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratio", type=str, required=True, help="ratio_estimator.pt from train_nre.")
    p.add_argument("--pool_dir", type=str, required=True)
    p.add_argument("--pool_split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device!r} requested but torch.cuda.is_available() is False.")

    checkpoint = torch.load(args.ratio, weights_only=False, map_location=args.device)
    ratio_estimator = checkpoint["ratio_estimator"]
    n_gal, n_los = checkpoint["n_gal"], checkpoint["n_los"]

    theta_pool, x_pool, pool_n_gal, pool_n_los = load_sims_standalone(args.pool_dir, args.pool_split)
    if pool_n_gal != n_gal or pool_n_los != n_los:
        raise ValueError(f"--pool_dir sims ({pool_n_gal} gal x {pool_n_los} los) don't match "
                         f"the ratio estimator ({n_gal} gal x {n_los} los).")
    n_pool = len(theta_pool)
    print(f"[mock_recovery] pool_size={n_pool}, n_gal={n_gal}, n_los={n_los}", flush=True)

    rng = np.random.default_rng(args.seed)
    truth_indices = rng.choice(n_pool, size=args.n_examples, replace=False)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for i, t_idx in enumerate(truth_indices):
        theta_true = theta_pool[t_idx].reshape(n_gal, n_los)
        x_obs = x_pool[t_idx]
        candidate_mask = np.ones(n_pool, dtype=bool)
        candidate_mask[t_idx] = False
        theta_cand = theta_pool[candidate_mask]

        x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32, device=args.device)
        theta_cand_t = torch.as_tensor(theta_cand, dtype=torch.float32, device=args.device)
        x_obs_tiled = x_obs_t.unsqueeze(0).expand(len(theta_cand), -1)

        with torch.no_grad():
            log_ratio = ratio_estimator(theta=theta_cand_t, x=x_obs_tiled).squeeze(-1)
        log_ratio = log_ratio.detach().cpu().numpy()

        log_w = log_ratio - log_ratio.max()
        w = np.exp(log_w)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)

        theta_cand_grid = theta_cand.reshape(len(theta_cand), n_gal, n_los)
        marginal_map = np.tensordot(w, theta_cand_grid, axes=(0, 0))

        x_ion_true = 1.0 - theta_true.mean()
        x_ion_est = 1.0 - marginal_map.mean()
        mae = np.abs(marginal_map - theta_true).mean()
        print(f"[mock_recovery] example {i} (pool idx {t_idx}): ESS={ess:.1f}/{n_pool - 1} "
              f"({100 * ess / (n_pool - 1):.2f}%), x_ion true={x_ion_true:.3f} vs. "
              f"posterior-implied={x_ion_est:.3f}, mean|marginal-true|={mae:.3f}", flush=True)

        results[f"example_{i}__theta_true"] = theta_true
        results[f"example_{i}__marginal_map"] = marginal_map
        results[f"example_{i}__ess"] = ess
        results[f"example_{i}__pool_idx"] = t_idx

    out_path = os.path.join(args.output_dir, "mock_recovery_demo.npz")
    np.savez(out_path, n_gal=n_gal, n_los=n_los, n_examples=args.n_examples,
            pool_size=n_pool, **results)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
