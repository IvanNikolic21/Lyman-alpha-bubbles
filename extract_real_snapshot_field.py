"""
Extract the real neutral-fraction field from one of the ALREADY-CACHED
single-box snapshots (lyabubbles/lightcone_field.py's _RAW_SNAPSHOTS) as a
plain .npy array, so it can be copied to a local machine and compared
against the synthetic bubble mock (mock_bubble_lightcone.py) -- no new
simulation, just reading an existing HDF5 file.

Defaults to the z=6.5 snapshot (x_H=0.27111432, the first/most-ionized
entry in the table) since that's the single cleanest real data point to
compare a mock generated at a matching target x_ion against.

UNVERIFIED: the exact internal HDF5 dataset key for the neutral-fraction
array (guessed candidates below, common py21cmfast v3/v4 names) -- if none
of them match, this script prints the file's actual internal structure
instead of guessing further, so you can tell me the real key name.

Usage:
    python extract_real_snapshot_field.py [z]
    (z defaults to 6.5 -- must exactly match one of _RAW_SNAPSHOTS' entries,
    e.g. 6.5, 6.6158, 6.7681, ...)

Then copy the resulting .npy to the local repo clone (e.g.
scp/rsync it into Lyman-alpha-bubbles/) so it's readable there.
"""
import sys

import h5py
import numpy as np

Z = float(sys.argv[1]) if len(sys.argv) > 1 else 6.5

_PATH_TEMPLATE = (
    "/lustre/astro/ivannik/21cmFAST_cache/cf853e8a92de487bf9c865a26e65e76d/"
    "1956/ffa852ccaa39d8f82951cc98ff798ab4/{z}/"
    "ff490db45ce98b111ca6e375b0d8c8f0/IonizedBox.h5"
)
path = _PATH_TEMPLATE.format(z=f"{Z:.4f}")
print(f"[open] {path}")

CANDIDATE_KEYS = [
    "IonizedBox/OutputFields/neutral_fraction",   # confirmed 2026-08-26 against a real cached snapshot
    "xH_box", "neutral_fraction", "x_HI", "xHI",
    "OutputFields/xH_box", "OutputFields/neutral_fraction",
]


def find_dataset(h5obj, name_fragment, path=""):
    """Recursively search for any dataset whose name contains name_fragment
    (case-insensitive) -- fallback if none of CANDIDATE_KEYS match exactly."""
    hits = []
    for key in h5obj.keys():
        item = h5obj[key]
        full = f"{path}/{key}" if path else key
        if isinstance(item, h5py.Dataset):
            if name_fragment.lower() in key.lower():
                hits.append(full)
        elif isinstance(item, h5py.Group):
            hits.extend(find_dataset(item, name_fragment, full))
    return hits


with h5py.File(path, "r") as f:
    arr = None
    used_key = None
    for k in CANDIDATE_KEYS:
        if k in f:
            arr = f[k][()]
            used_key = k
            break

    if arr is None:
        print("[!] None of the guessed candidate keys matched. Searching for anything "
              "with 'x' or 'neutral' or 'ion' in the name instead...")
        for frag in ("neutral", "xH", "ion"):
            hits = find_dataset(f, frag)
            if hits:
                print(f"    possible matches for '{frag}': {hits}")
        print("\n[structure] top-level keys:", list(f.keys()))
        def _print_tree(h5obj, prefix="  "):
            for key in h5obj.keys():
                item = h5obj[key]
                if isinstance(item, h5py.Group):
                    print(f"{prefix}{key}/ (group)")
                    _print_tree(item, prefix + "  ")
                else:
                    print(f"{prefix}{key}  shape={item.shape} dtype={item.dtype}")
        _print_tree(f)
        print("\n[!] No array extracted -- tell me which key above is the neutral "
              "fraction / ionized fraction field and I'll fix CANDIDATE_KEYS.")
        sys.exit(1)

    print(f"[found] key='{used_key}' shape={arr.shape} dtype={arr.dtype} "
          f"mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}")

out_path = f"real_snapshot_z{Z:.4f}_field.npy"
np.save(out_path, arr.astype(np.float32))
print(f"[saved] {out_path}  ({arr.nbytes / 1e6:.1f} MB before float32 cast)")
print("Copy this .npy to the local repo clone (Lyman-alpha-bubbles/) so it's readable there.")