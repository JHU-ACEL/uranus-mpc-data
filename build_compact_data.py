#!/usr/bin/env python
"""
One-time, offline preprocessing for Interactive_Plots.py (run on your own machine).

Converts the original long-format pickles into compact ``.npz`` files:

  * float32 arrays shaped (groups, trajectories, time steps) instead of a long DataFrame
    with a repeated Time/Group column  ->  much smaller downloads and far less memory
  * plain numeric arrays instead of pickled pandas objects  ->  no dependence on the
    pandas version that wrote the pickle (a common failure when running in the browser)
  * the aggregate "Full" group is dropped for the Earth/Uranus files (the notebook
    never shows it); Quaternion Errors are dropped because the notebook never plots them

Usage
-----
    pip install numpy pandas
    python build_compact_data.py                       # reads the pickles from GitHub
    python build_compact_data.py --src ./data          # ...or from a local folder
    python build_compact_data.py --out ./compact

Then upload the contents of ``--out`` somewhere that serves files with CORS enabled
(e.g. a GitHub repo -> raw.githubusercontent.com, or GitHub Pages) and set
``COMPACT_BASE`` in Interactive_Plots.py to that folder's URL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SRC = "https://raw.githubusercontent.com/JHU-ACEL/uranus-mpc-data/main/data"
HORIZONS = [50, 100, 150, 200, 250]
BODIES = ("earth", "uranus")

# output name -> (pickle file name, drop the aggregate "Full" group?)
FILES = {f"{b}_{h}": (f"{b}_{h}.pkl", True) for b in BODIES for h in HORIZONS}
FILES["torque_df"] = ("torque_v_dipole.pkl", False)


def to_arrays(df: pd.DataFrame, drop_full: bool) -> dict[str, np.ndarray]:
    """Long-format DataFrame -> dict of compact arrays (same ordering the notebook expects)."""
    if drop_full:
        df = df.loc[df["Group"] != "Full"]

    groups = list(df["Group"].unique())
    n_steps = int(df["Time"].nunique())

    angle, omega, times = [], [], []
    batch_sizes = set()
    for g in groups:
        gdf = df.loc[df["Group"] == g]
        if len(gdf) % n_steps:
            raise ValueError(f"group {g!r}: {len(gdf)} rows is not a multiple of {n_steps} time steps")
        batch = len(gdf) // n_steps
        batch_sizes.add(batch)
        t = gdf["Time"].to_numpy(dtype=float).reshape(batch, n_steps)
        if not np.allclose(t, t[0]):
            raise ValueError(f"group {g!r}: trajectories do not share one time vector")
        times.append(t[0])
        angle.append(gdf["Angle Errors"].to_numpy(dtype=float).reshape(batch, n_steps))
        omega.append(gdf["Omega Errors"].to_numpy(dtype=float).reshape(batch, n_steps))

    if len(batch_sizes) != 1:
        raise ValueError(f"groups have different numbers of trajectories: {sorted(batch_sizes)}")
    if not all(np.allclose(times[0], t) for t in times):
        raise ValueError("groups do not share one time vector")

    return {
        "groups": np.array(groups),
        "time": times[0].astype(np.float64),
        "angle": np.stack(angle).astype(np.float32),
        "omega": np.stack(omega).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=DEFAULT_SRC, help="folder or URL containing the original .pkl files")
    parser.add_argument("--out", default="compact", help="output folder for the .npz files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = args.src.rstrip("/")
    is_local = "://" not in src

    total_in = total_out = 0
    for name, (pickle_name, drop_full) in FILES.items():
        source = f"{src}/{pickle_name}"
        arrays = to_arrays(pd.read_pickle(source), drop_full)
        target = out / f"{name}.npz"
        np.savez_compressed(target, **arrays)

        # Verify the round trip before trusting the file
        with np.load(target, allow_pickle=False) as z:
            assert list(z["groups"]) == list(arrays["groups"])
            assert np.array_equal(z["angle"], arrays["angle"]) and np.array_equal(z["omega"], arrays["omega"])

        size_out = os.path.getsize(target)
        total_out += size_out
        line = f"{name:12s} groups={len(arrays['groups'])} traj/group={arrays['angle'].shape[1]} " \
               f"steps={arrays['angle'].shape[2]}  ->  {size_out / 1e3:8.1f} kB"
        if is_local:
            size_in = os.path.getsize(source)
            total_in += size_in
            line += f"  (pickle: {size_in / 1e3:8.1f} kB)"
        print(line)

    print(f"\nTotal compact size: {total_out / 1e6:.2f} MB" + (f"  (pickles: {total_in / 1e6:.2f} MB)" if is_local else ""))
    print(f"Upload the contents of '{out}/' and set COMPACT_BASE in Interactive_Plots.py to its URL.")


if __name__ == "__main__":
    main()
