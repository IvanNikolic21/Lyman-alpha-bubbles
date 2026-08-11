"""
The bubble-model SBI prior (`_log_prob_theta`/`_BubblePrior`/`make_sbi_prior`),
factored OUT of `sbi_real_data.py` into its own module for one reason:
pickling.

`sbi_real_data.py`'s `train` subcommand is invoked as `python
sbi_real_data.py train ...`, which makes Python register that run as
`sys.modules['__main__']`, NOT `sys.modules['sbi_real_data']` -- even though
it's the same file. If the prior class lived there, `torch.save`d posterior
checkpoints would pickle it with module path `__main__`, and ANY other
script that later unpickles the checkpoint by properly `import`ing
`sbi_real_data` (`sbi_calibrate.py`, `sbi_real_data.py infer` in its own
separate process, etc.) has a DIFFERENT `__main__`, so loading fails with
`AttributeError: Can't get attribute '_BubblePrior' on <module '__main__' ...>`.

Forcibly overriding `__module__` on a class defined inside the entry-point
script does NOT fix this -- it makes `torch.save` itself fail instead
(`PicklingError: ... not the same object as ...`), because pickle then tries
to verify the class by re-`import`ing the entry-point script under its own
filename while it's still running as `__main__`, producing a SECOND,
non-identical class object.

The only clean fix: define the class in a module that is NEVER itself the
`__main__` entry point of any of these workflows. This file is only ever
`import`ed (by `sbi_real_data.py` and, transitively, `sbi_calibrate.py`), so
`sys.modules['sbi_prior']` is populated the same consistent way in every
process, and pickling/unpickling `_BubblePrior` across `train` -> `infer`/
`sbi_calibrate.py` process boundaries works correctly.
"""
import numpy as np
from scipy.stats import exponnorm

import real_data_run as rdr

try:
    import torch as _torch   # soft/optional, matches sbi_real_data.py's convention
except ImportError:
    _torch = None

_N_BUB_TO_NDIM = {1: rdr.NDIM, 2: rdr.NDIM_2BUB, 3: rdr.NDIM_3BUB}
_N_BUB_TO_PRIOR_TRANSFORM = {1: rdr._prior_transform, 2: rdr._prior_transform_2bub,
                             3: rdr._prior_transform_3bub}


# NOTE: exact `sbi` API (constructor args, whether `.log_prob()` is required
# by `build_posterior`'s default sampler) is TBD pending confirming the
# installed version on the cluster -- see `sbi_real_data.py`'s module
# docstring for the full caveat. This is a best-effort implementation of a
# proper torch.distributions-compatible prior matching `_prior_transform`'s
# ACTUAL density (uniform x/y/z box, EMG-truncated r_bub via
# `R_BUB_DIST_PARAMS`, and -- for M2/M3 -- the z-ordering density implied by
# the sequential z2~Uniform(z_lo, z1) construction, NOT a plain
# independent-uniform box).

def _log_prob_theta(theta_batch, n_bub):
    """log p(theta) under the ACTUAL prior_transform density (not a plain
    BoxUniform) -- torch tensor in, numpy out is fine since this only needs
    to be differentiable if sbi's posterior sampler requires gradients
    through the prior (unlikely for a rejection/MCMC-based default sampler;
    revisit if it errors)."""
    s = rdr._S
    theta_batch = np.atleast_2d(np.asarray(theta_batch))
    x_lo, x_hi = s.prior_lo[0], s.prior_hi[0]
    y_lo, y_hi = s.prior_lo[1], s.prior_hi[1]
    z_lo, z_hi = s.prior_lo[2], s.prior_hi[2]
    r_lo, r_hi = s.prior_lo[3], s.prior_hi[3]
    f_lo = exponnorm.cdf(r_lo, *rdr.R_BUB_DIST_PARAMS)
    f_hi = exponnorm.cdf(r_hi, *rdr.R_BUB_DIST_PARAMS)

    def _log_f_r(r):
        pdf = exponnorm.pdf(r, *rdr.R_BUB_DIST_PARAMS) / (f_hi - f_lo)
        return np.where((r >= r_lo) & (r <= r_hi), np.log(np.maximum(pdf, 1e-300)), -np.inf)

    log_p = np.zeros(theta_batch.shape[0])
    if n_bub == 1:
        x, y, z, r = theta_batch.T
        log_p += -np.log(x_hi - x_lo) - np.log(y_hi - y_lo) - np.log(z_hi - z_lo)
        log_p += _log_f_r(r)
    elif n_bub == 2:
        x1, y1, z1, r1, x2, y2, z2, r2 = theta_batch.T
        log_p += -2 * np.log(x_hi - x_lo) - 2 * np.log(y_hi - y_lo)
        log_p += _log_f_r(r1) + _log_f_r(r2)
        valid = (z1 >= z2) & (z2 >= z_lo) & (z1 <= z_hi)
        log_p += np.where(valid, -np.log(z_hi - z_lo) - np.log(np.maximum(z1 - z_lo, 1e-300)), -np.inf)
    elif n_bub == 3:
        x1, y1, z1, r1, x2, y2, z2, r2, x3, y3, z3, r3 = theta_batch.T
        log_p += -3 * np.log(x_hi - x_lo) - 3 * np.log(y_hi - y_lo)
        log_p += _log_f_r(r1) + _log_f_r(r2) + _log_f_r(r3)
        valid = (z1 >= z2) & (z2 >= z3) & (z3 >= z_lo) & (z1 <= z_hi)
        log_p += np.where(
            valid,
            -np.log(z_hi - z_lo) - np.log(np.maximum(z1 - z_lo, 1e-300))
            - np.log(np.maximum(z2 - z_lo, 1e-300)),
            -np.inf,
        )
    else:
        raise ValueError(f"n_bub must be 1, 2, or 3, got {n_bub}")
    return log_p


if _torch is not None:
    class _BubblePrior(_torch.distributions.Distribution):
        """A torch.distributions.Distribution-compatible prior wrapping
        `_prior_transform`/`_log_prob_theta`, for `sbi.inference.NPE(prior=...)`.
        `device`: must match the `device=` passed to `NPE(...)` -- sbi calls
        `.sample()`/`.log_prob()` on this prior during training/posterior
        building, and a device mismatch between the prior's tensors and the
        density estimator's (e.g. prior on CPU, network on CUDA) will error
        or silently force a slow/incorrect fallback.

        Defined at true module scope (guarded by `_torch is not None` so
        importing this module still works with no torch installed) in a
        module that's never a `__main__` entry point -- see this file's
        module docstring for why that's required for cross-process
        pickling (`train` -> `infer`/`sbi_calibrate.py`) to work at all."""
        arg_constraints = {}
        has_rsample = False

        def __init__(self, n_bub, device='cpu'):
            self.n_bub = n_bub
            self.device = device
            self.ndim = _N_BUB_TO_NDIM[n_bub]
            self.prior_transform = _N_BUB_TO_PRIOR_TRANSFORM[n_bub]
            super().__init__(event_shape=_torch.Size([self.ndim]), validate_args=False)

        def sample(self, sample_shape=_torch.Size()):
            n = int(np.prod(sample_shape)) if len(sample_shape) else 1
            u = np.random.uniform(size=(n, self.ndim))
            theta = np.array([self.prior_transform(ui) for ui in u])
            out = _torch.as_tensor(theta, dtype=_torch.float32, device=self.device)
            return out.reshape(*sample_shape, self.ndim) if len(sample_shape) else out[0]

        def log_prob(self, value):
            value_np = value.detach().cpu().numpy()
            lp = _log_prob_theta(value_np, self.n_bub)
            return _torch.as_tensor(lp, dtype=_torch.float32, device=self.device)

        @property
        def support(self):
            # Approximate: a per-dimension box, NOT the true support for
            # M2/M3 (which excludes z2 > z1 / z3 > z2 -- `log_prob` already
            # returns -inf there, but this box constraint alone wouldn't
            # reject a support-restriction/rejection sampler that only
            # checks the box). Fine for M1; for M2/M3 this may let a few
            # invalid (wrong z-order) proposals through the box check that
            # `log_prob` then correctly zeroes out downstream -- verify this
            # is actually handled correctly by whatever sbi does with
            # `support` once installed. Reads `rdr._S` fresh on each access
            # (not cached at construction) so this reflects whatever catalog
            # THIS process loaded, not whatever the training process had.
            s = rdr._S
            lo = _torch.as_tensor(np.tile(s.prior_lo, self.n_bub), dtype=_torch.float32,
                                  device=self.device)
            hi = _torch.as_tensor(np.tile(s.prior_hi, self.n_bub), dtype=_torch.float32,
                                  device=self.device)
            return _torch.distributions.constraints.interval(lo, hi)


def make_sbi_prior(n_bub, device='cpu'):
    return _BubblePrior(n_bub, device=device)
