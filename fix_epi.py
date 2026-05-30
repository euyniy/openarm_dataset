"""Verify and repair action overflow for a single episode."""

import warnings
import numpy as np
import pandas as pd
from openarm_dataset.dataset import Dataset

DATASET_PATH = "/home/yy/dataset/test"
DATASET_PATH = "/mnt/syno127/volume1/openarm-dataset/raw_data/pillow-all-cam-merged-till-0529"

FPS = 30
SMOOTHING_CUTOFF = 1.0
JOINT_KEYS = ["arms/right/qpos", "arms/left/qpos"]
CORRUPT_THRESHOLD = 1e6


def describe_vector(X, label="vector"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.nanstd(X, axis=0)
        if caught:
            absmax = float(np.nanmax(np.abs(X)))
            print(
                f"[DIAG overflow] '{label}': dtype={X.dtype} shape={X.shape} "
                f"absmax={absmax:.4e} any_inf={bool(np.any(np.isinf(X)))} "
                f"any_nan={bool(np.any(np.isnan(X)))}"
            )
        else:
            print(f"[OK] '{label}': shape={X.shape} absmax={np.abs(X).max():.4e}")


def repair_parquet(path: str, threshold: float = CORRUPT_THRESHOLD) -> int:
    """Replace corrupt values (|x| > threshold) with the average of their neighbors.

    Returns the number of repaired rows.
    """
    df = pd.read_parquet(path)
    vals = np.stack(df["value"].tolist()).astype(np.float64)
    bad_rows = np.where(np.any(np.abs(vals) > threshold, axis=1))[0]
    if len(bad_rows) == 0:
        return 0

    for idx in bad_rows:
        prev = vals[idx - 1] if idx > 0 else vals[idx + 1]
        next_ = vals[idx + 1] if idx < len(vals) - 1 else vals[idx - 1]
        avg = (prev + next_) / 2.0
        bad_joints = np.abs(vals[idx]) > threshold
        print(f"  repairing row {idx} ({df['timestamp'].iloc[idx]}): "
              f"joints {np.where(bad_joints)[0].tolist()} "
              f"[{vals[idx][bad_joints]}] -> [{avg[bad_joints]}]")
        vals[idx, bad_joints] = avg[bad_joints]

    df["value"] = list(vals.astype(np.float32))
    df.to_parquet(path, index=False)
    return len(bad_rows)


dataset = Dataset(DATASET_PATH, camera_names=[])
dataset.set_smoothing(cutoff=SMOOTHING_CUTOFF)

EP = 1026  # folder 17

# find and repair corrupt parquets before sampling
for attr in dataset._get_embodiment_attributes("action", EP):
    path = str(attr["path"])
    n = repair_parquet(path)
    if n:
        print(f"  repaired {n} row(s) in {path}")

# re-sample after repair
samples = dataset.sample(hz=FPS, episode_index=EP)
print(f"\nsampled {len(samples)} frames after repair")

actions = np.vstack([
    np.concatenate([s.action[k] for k in JOINT_KEYS], axis=0).astype(np.float32)
    for s in samples
])
describe_vector(actions, label=f"action/ep={EP}")

