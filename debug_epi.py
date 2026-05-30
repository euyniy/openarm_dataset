import warnings
import numpy as np
from openarm_dataset.dataset import Dataset


DATASET_PATH = "/home/yy/dataset/test"
DATASET_PATH = "/mnt/syno127/volume1/openarm-dataset/raw_data/pillow-all-cam-merged-till-0529"
FPS = 30
SMOOTHING_CUTOFF = 1.0
JOINT_KEYS = ["arms/right/qpos", "arms/left/qpos"]


def describe_vector(X, label="vector"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        std_vals = np.nanstd(X, axis=0)
        if caught:
            absmax = float(np.nanmax(np.abs(X)))
            print(
                f"[DIAG overflow] '{label}': dtype={X.dtype} shape={X.shape} "
                f"absmax={absmax:.4e} any_inf={bool(np.any(np.isinf(X)))} "
                f"any_nan={bool(np.any(np.isnan(X)))}"
            )
        else:
            print(f"[OK] '{label}': shape={X.shape} absmax={np.abs(X).max():.4e}")


dataset = Dataset(DATASET_PATH, camera_names=[])
dataset.set_smoothing(cutoff=SMOOTHING_CUTOFF)

for e in range(1000, 1200):
    samples = dataset.sample(hz=FPS, episode_index=e)
    print(f"sampled {len(samples)} frames")
    
    # check raw action values per key before stacking
    for k in JOINT_KEYS:
        vals = np.stack([s.action[k] for s in samples]).astype(np.float32)
        print(f"  {k}: shape={vals.shape} absmax={np.abs(vals).max():.4e}")
    
    # stack as lerobot_v21 does
    actions = np.vstack([
        np.concatenate([s.action[k] for k in JOINT_KEYS], axis=0).astype(np.float32)
        for s in samples
    ])
    describe_vector(actions, label="action/ep="+str(e))

