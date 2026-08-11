"""
Custom theta embedding network for `sbi_pixel_field.py`'s NRE classifier,
factored into its own module for the SAME reason `sbi_prior.py` exists: this
object gets embedded inside the saved `ratio_estimator.pt` checkpoint
(`torch.save` pickles the whole classifier, including its embedding nets),
and `sbi_pixel_field.py` is invoked directly as `python sbi_pixel_field.py
train_nre ...`, which binds anything defined there to `sys.modules['__main__']`
in that process -- unresolvable by any OTHER process (`infer`, run separately)
that properly `import`s `sbi_pixel_field`. See `sbi_prior.py`'s module
docstring for the full explanation; same fix here: define it somewhere that
is never itself a `__main__` entry point.

Motivation for the embedding net itself: theta is a flattened (n_gal *
n_los,) binary vector, but it has real 2D grid structure (one row per
galaxy, one column per LOS bin) -- sbi's default classifier treats it as
n_gal*n_los independent flat inputs with no notion of that structure. A
small CNN that reshapes theta back to (n_gal, n_los) before convolving can
exploit spatial locality (nearby LOS bins within a galaxy, and -- via the
shared-lightcone design -- some structure across nearby galaxies) instead of
learning it from scratch via a fully-connected layer.
"""
try:
    import torch as _torch
    import torch.nn as _nn
except ImportError:
    _torch = None
    _nn = None


if _torch is not None:
    class ThetaGridEmbedding(_nn.Module):
        """Reshapes a flattened (batch, n_gal*n_los) theta back to (batch, 1,
        n_gal, n_los) and runs a small 2D CNN over it. `n_gal`/`n_los` are
        explicit constructor args (not closures) for the same picklability
        reason `sbi_prior.py::_BubblePrior` takes `n_bub`/`device` explicitly."""

        def __init__(self, n_gal, n_los, out_dim=64):
            super().__init__()
            self.n_gal = n_gal
            self.n_los = n_los
            self.out_dim = out_dim
            # Pool to a fixed small spatial size regardless of n_gal/n_los so
            # this works across different catalog sizes/--n_los choices
            # without needing to hardcode a flattened conv-output size.
            pool_gal = min(6, n_gal)
            pool_los = min(10, n_los)
            self.net = _nn.Sequential(
                _nn.Conv2d(1, 16, kernel_size=3, padding=1), _nn.ReLU(),
                _nn.Conv2d(16, 32, kernel_size=3, padding=1), _nn.ReLU(),
                _nn.AdaptiveAvgPool2d((pool_gal, pool_los)),
                _nn.Flatten(),
                _nn.Linear(32 * pool_gal * pool_los, out_dim), _nn.ReLU(),
            )

        def forward(self, theta):
            x = theta.reshape(-1, 1, self.n_gal, self.n_los)
            return self.net(x)
