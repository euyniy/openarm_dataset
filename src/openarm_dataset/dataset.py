# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenArm Dataset."""

import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import scipy.signal as signal

from .camera import Camera
from .metadata import Metadata
from .sampler import Sampler, Sample


class Dataset:
    """OpenArm Dataset."""

    def __init__(
        self,
        path: str | os.PathLike,
        meta: Metadata = None,
        camera_names: list[str] = None,
    ):
        """Initialize Dataset.

        Args:
            path: Path of the dataset.
            meta: Metadata of the dataset. Uses the metadata stored in the
                dataset if None.
            camera_names: Names of the camera to use. Uses all cameras in the
                dataset if None.

        """
        self.root_path = Path(path)
        self.meta = Metadata(self.root_path / "metadata.yaml") if meta is None else meta
        self._camera_names = camera_names
        self._smoothing_cutoff = None

    def set_smoothing(self, cutoff: float):
        """Set smoothing."""
        self._smoothing_cutoff = cutoff

    def validate(
        self,
        on_error=None,
        qpos_jump_threshold: float | None = None,
        qpos_absmax: float | None = None,
        min_duration: float | None = None,
    ) -> bool:
        """Validate this dataset.

        Args:
            on_error: Optional callable that is called with an error message
                string for each validation error found. If ``None``, errors
                are not reported.
            qpos_jump_threshold: If set, flag qpos frame-to-frame deltas above
                this value (radians) as abrupt jumps.
            qpos_absmax: If set, flag qpos values whose absolute value
                exceeds this threshold.
            min_duration: If set, flag episodes whose duration is shorter
                than this value (seconds).

        Returns:
            ``True`` if the dataset is valid, ``False`` otherwise.

        """
        valid = True
        checked_paths = set()
        for episode in self.meta.episodes:
            ep_id = episode["id"]

            # File-level checks.
            for type_name in ("obs", "action"):
                for attribute in self.get_embodiment_attributes(type_name, episode):
                    path = attribute["path"]
                    if path in checked_paths or not path.exists():
                        continue
                    checked_paths.add(path)
                    rel = path.relative_to(self.root_path)
                    check_qpos_absmax = (
                        qpos_absmax is not None and attribute["name"] == "qpos"
                    )
                    # Fast path for null detection via parquet metadata.
                    has_null = False
                    file_meta = pq.read_metadata(path)
                    for rg_index in range(file_meta.num_row_groups):
                        row_group = file_meta.row_group(rg_index)
                        for col_index in range(row_group.num_columns):
                            col_meta = row_group.column(col_index)
                            col_name = col_meta.path_in_schema.split(".")[0]
                            if col_name == "timestamp":
                                continue
                            stats = col_meta.statistics
                            if (
                                stats is not None
                                and stats.has_null_count
                                and stats.null_count > 0
                            ):
                                has_null = True
                                break
                        if has_null:
                            break

                    if has_null:
                        if on_error is not None:
                            on_error(f"{rel}: includes null values")
                        valid = False

                    # Do not short-circuit after null detection; continue scanning
                    # values to report non-finite/qpos absmax in the same file.
                    table = pq.read_table(path)
                    has_nonfinite = False
                    for col_name in table.schema.names:
                        if col_name == "timestamp":
                            continue
                        col = table.column(col_name)
                        flat = col.combine_chunks().values
                        if not pa.types.is_floating(flat.type):
                            continue

                        is_finite = pc.is_finite(flat)
                        if not has_nonfinite and not pc.all(is_finite).as_py():
                            has_nonfinite = True
                            if on_error is not None:
                                on_error(f"{rel}: includes NaN or Inf values")
                            valid = False

                        if check_qpos_absmax and col_name == "qpos":
                            col_absmax = pc.max(
                                pc.if_else(is_finite, pc.abs(flat), None)
                            ).as_py()
                            if col_absmax is not None and col_absmax > qpos_absmax:
                                if on_error is not None:
                                    on_error(
                                        f"{rel} (qpos): "
                                        f"absmax={col_absmax:.4f} > {qpos_absmax}"
                                    )
                                valid = False

            # Episode-level checks.
            if min_duration is not None or qpos_jump_threshold is not None:
                # Fast path: if only duration is requested, read just one
                # timestamp column instead of loading full obs/action DataFrames.
                if min_duration is not None and qpos_jump_threshold is None:
                    duration = None
                    for attr in self.get_embodiment_attributes("obs", episode):
                        path = attr["path"]
                        if not path.exists():
                            continue
                        try:
                            ts = pq.read_table(path, columns=["timestamp"]).column(
                                "timestamp"
                            )
                            if len(ts) >= 2:
                                ts_ns = pc.cast(ts, pa.int64())
                                duration = (ts_ns[-1].as_py() - ts_ns[0].as_py()) / 1e9
                        except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
                            if on_error is not None:
                                on_error(
                                    f"episode {ep_id} obs: failed to load duration ({exc})"
                                )
                        break

                    if duration is not None and duration < min_duration:
                        if on_error is not None:
                            on_error(
                                f"episode {ep_id}: "
                                f"duration={duration:.2f}s < {min_duration}s"
                            )
                        valid = False
                    continue

                obs_for_duration = None
                obs_qpos = {}
                if qpos_jump_threshold is not None:
                    try:
                        obs_qpos = self._load_qpos_values("obs", episode, use_unixtime=True)
                    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
                        if on_error is not None:
                            on_error(f"episode {ep_id} obs: failed to load ({exc})")
                        continue
                    obs_for_duration = obs_qpos
                elif min_duration is not None:
                    try:
                        obs_for_duration = self.load_obs(episode, use_unixtime=True)
                    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
                        if on_error is not None:
                            on_error(f"episode {ep_id} obs: failed to load ({exc})")
                        continue

                if min_duration is not None:
                    duration = None
                    for df in obs_for_duration.values():
                        if len(df) >= 2:
                            duration = float(df.index[-1] - df.index[0])
                            break
                    if duration is not None and duration < min_duration:
                        if on_error is not None:
                            on_error(
                                f"episode {ep_id}: "
                                f"duration={duration:.2f}s < {min_duration}s"
                            )
                        valid = False

                if qpos_jump_threshold is not None:
                    try:
                        action_qpos = self._load_qpos_values(
                            "action",
                            episode,
                            use_unixtime=True,
                        )
                    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
                        if on_error is not None:
                            on_error(f"episode {ep_id} action: failed to load ({exc})")
                        action_qpos = {}

                    for source, data_dict in (
                        ("obs", obs_qpos),
                        ("action", action_qpos),
                    ):
                        for key, df in data_dict.items():
                            arr = df.values
                            if arr.shape[0] < 2:
                                continue
                            # Vectorized frame-to-frame jump count across all joints.
                            n_bad = int(
                                np.count_nonzero(
                                    np.abs(arr[1:] - arr[:-1]) > qpos_jump_threshold
                                )
                            )
                            if n_bad > 0:
                                if on_error is not None:
                                    on_error(
                                        f"episode {ep_id} {source}/{key}: "
                                        f"{n_bad} abrupt jump(s) > {qpos_jump_threshold} rad"
                                    )
                                valid = False

        return valid

    def _load_qpos_values(
        self,
        type_: str,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load only qpos attributes for the given stream type."""
        values = {}
        for attribute in self.get_embodiment_attributes(type_, episode):
            if attribute["name"] != "qpos":
                continue
            values[attribute["key"]] = self._load_embodiment_value(
                attribute,
                use_unixtime=use_unixtime,
                cutoff=cutoff or self._smoothing_cutoff,
            )
        return values

    @property
    def num_episodes(self) -> int:
        """Return number of episodes."""
        return self.meta.num_episodes

    @property
    def camera_names(self) -> list[str]:
        """Return camera names."""
        if self._camera_names is not None:
            return self._camera_names
        return list(self.meta.equipment.perceptions.cameras)

    @property
    def camera_format(self) -> str:
        """Return the camera format ("dir" or "tar") shared by all cameras.

        Every camera in the dataset is expected to use the same format.

        Raises:
            ValueError: If cameras use a mix of "dir" and "tar".

        """
        formats = set()
        for episode in self.meta.episodes:
            for name in self.camera_names:
                formats.add(self.load_camera(name, episode).format)
        if len(formats) > 1:
            raise ValueError(f"Inconsistent camera formats: {sorted(formats)}")
        return formats.pop()

    def _episode_id(self, index: int) -> str:
        return self.meta.episodes[index]["id"]

    def episode_path(self, episode: dict = None) -> Path:
        """Return the path of the episode."""
        if episode is None:
            return self.root_path
        return self.root_path / "episodes" / episode["id"]

    def load_obs(
        self,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load obs data.

        Args:
            episode: Episode to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.

        Returns:
            Dictionary mapping names to DataFrames.

        Example:
            {
                "arms/right/qpos": DataFrame,
                "arms/left/qpos": DataFrame,
            }

        """
        return self._load_embodiment_values(
            "obs",
            episode,
            use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_action(
        self,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load action data.

        Args:
            episode: Episode to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.

        Returns:
            Dictionary mapping names to DataFrames.

        Example:
            {
                "arms/right/qpos": DataFrame,
                "arms/left/qpos": DataFrame,
            }

        """
        return self._load_embodiment_values(
            "action",
            episode,
            use_unixtime=use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_cameras(self, episode: dict) -> dict[str, Camera]:
        """Load all camera data.

        Args:
            episode: Episode to load.

        Returns:
            Dictionary mapping names to Camera.

        Example:
            {
                "ceiling": Camera,
                "head": Camera,
                "wrist_right": Camera,
                "wrist_left": Camera,
            }

        """
        return {name: self.load_camera(name, episode) for name in self.camera_names}

    def load_camera(self, name: str, episode: dict) -> Camera:
        """Load camera data.

        Args:
            name: Camera name to load.
            episode: Episode to load.

        Returns:
            Camera.

        """
        if name not in self.camera_names:
            raise KeyError(f"Camera {name} not found. Available: {self.camera_names}")
        base_path = self.episode_path(episode)
        # Unversioned dataset. This is for backward compatibility.
        if self.meta.version is None:
            path = base_path / f"{name}_image"
            if not path.exists() and name.endswith("_wrist"):
                path = base_path / f"{name.removesuffix('_wrist')}_image"
        else:
            path = base_path / "cameras" / name
        return Camera(name, path)

    def sample(
        self,
        hz: float,
        episode: dict,
    ) -> list[Sample]:
        """Sample the all modalities data to the specified hz.

        Args:
            episode: Episode to sample.
            hz: Sampling hz.

        Returns:
            List of Sample.

        Example:
            >>> samples = samples(10, 0)
            >>> samples[0].timestamp
            1773446407.1999931
            >>> samples[0].obs
            {
                "arms/right/qpos": np.ndarray,
                'arms/left/qpos': np.ndarray,
            }
            >>> samples[0].action
            {
                "arms/right/qpos": np.ndarray,
                'arms/left/qpos': np.ndarray,
            }
            >>> samples[0].cameras
            {
                "ceiling": Frame,
                "head": Frame,
                "wrist_right": Frame,
                "wrist_left": Frame,
            }

        """
        sampler = Sampler()
        return list(sampler.sample(self, episode, hz))

    def get_embodiment_attributes(self, type_: str, episode: dict):
        """Return the list of embodiment attributes for the given type and episode."""
        attributes = []
        for name, embodiment in self.meta.equipment.embodiments.items():
            # Unversioned dataset.
            # This is for backward compatibility.
            if self.meta.version is None:
                base_path = self.episode_path(episode) / type_
            else:
                base_path = self.episode_path(episode) / type_ / name
            if embodiment.components:
                for component in embodiment.components:
                    state_path = base_path / component / "state.parquet"
                    if state_path.exists():
                        for attr_name in ("qpos", "qvel", "qtorque"):
                            attributes.append(
                                {
                                    "key": f"{name}/{component}/{attr_name}",
                                    "embodiment": embodiment,
                                    "component": component,
                                    "name": attr_name,
                                    "path": state_path,
                                }
                            )
                        continue
                    for attribute in embodiment.attributes:
                        key = f"{name}/{component}/{attribute}"
                        # Unversioned dataset.
                        # This is for backward compatibility.
                        if self.meta.version is None:
                            path = (
                                base_path / f"{component}_arm" / f"{attribute}.parquet"
                            )
                        else:
                            path = base_path / component / f"{attribute}.parquet"
                        attributes.append(
                            {
                                "key": key,
                                "embodiment": embodiment,
                                "component": component,
                                "name": attribute,
                                "path": path,
                            }
                        )
            else:
                for attribute in embodiment.attributes:
                    key = f"{name}/{attribute}"
                    attributes.append(
                        {
                            "key": key,
                            "embodiment": embodiment,
                            "component": None,
                            "name": attribute,
                            "path": base_path / f"{attribute}.parquet",
                        }
                    )
        return attributes

    def _load_embodiment_values(
        self,
        type_: str,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        values = {}
        for attribute in self.get_embodiment_attributes(type_, episode):
            values[attribute["key"]] = self._load_embodiment_value(
                attribute,
                use_unixtime=use_unixtime,
                cutoff=cutoff,
            )
        return values

    def _load_embodiment_value(
        self,
        attribute: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(attribute["path"])
        if attribute["path"].name == "state.parquet":
            # 0.3.0 uses state.parquet with qpos/qvel/qtorque columns.
            column_name = attribute["name"]
            drop_columns = [c for c in ("qpos", "qvel", "qtorque") if c in df.columns]
        elif "positions" in df:
            # No version and 0.1.0 use "positions"
            column_name = "positions"
            drop_columns = ["positions"]
        else:
            column_name = "value"
            drop_columns = ["value"]
        df[list(attribute["embodiment"].joints)] = pd.DataFrame(
            df[column_name].tolist(),
            index=df.index,
        )
        df = df.drop(columns=drop_columns)
        if use_unixtime:
            df["timestamp"] = df["timestamp"].astype("int64") / 1e9
        df = df.set_index("timestamp")
        if cutoff is not None:
            df = self._apply_smoothing(df, cutoff=cutoff)
        return df

    def _apply_smoothing(
        self,
        df: pd.DataFrame,
        cutoff: float = 1.0,
        fps: float = 250.0,
    ) -> pd.DataFrame:
        if df.empty or cutoff is None:
            return df
        if len(df) <= 15:
            return df

        nyquist = fps * 0.5
        Wn = cutoff / nyquist
        Wn = min(0.99, max(0.01, Wn))
        b, a = signal.butter(4, Wn, btype="low")

        filtered_values = signal.filtfilt(b, a, df.values, axis=0)
        return pd.DataFrame(filtered_values, index=df.index, columns=df.columns)

    def _write(self, output: str | os.PathLike, camera_format: str = "dir"):
        """Write this dataset as the latest OpenArm dataset format."""
        output = Path(output)
        self.meta.write(output)
        self._write_data(output, camera_format=camera_format)

    def write(self, output: str | os.PathLike, format: str | None = None, **options):
        """Write this dataset in the specified format."""
        if format is None or format == "openarm":
            return self._write(output, **options)
        elif format == "lerobot_v2.1":
            from .lerobot_v21 import to_lerobotv21

            return to_lerobotv21(self, output, **options)
        elif format == "lerobot_v3.0":
            from .lerobot_v30 import to_lerobotv30

            return to_lerobotv30(self, output, **options)
        elif format == "gr00t":
            from .lerobot_v21 import to_gr00t

            return to_gr00t(self, output, **options)
        elif format == "rrd":
            try:
                from .rrd import to_rrd
            except ModuleNotFoundError as err:
                if err.name == "rerun":
                    raise ModuleNotFoundError(
                        "RRD export requires the optional dependency 'rerun-sdk'. Install with `pip install openarm_dataset[rerun]`."
                    ) from err
                raise

            return to_rrd(self, output, **options)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_data(self, output: Path, camera_format: str = "dir"):
        for episode in self.meta.episodes:
            self._write_episode(output, episode, camera_format=camera_format)

    def _write_episode(self, output: Path, episode: dict, camera_format: str = "dir"):
        self._write_embodiment_data(output, episode)
        self._write_camera_data(output, episode, camera_format=camera_format)

    def _write_embodiment_data(self, output: Path, episode: dict):
        written_state_paths = set()
        for type_ in ["action", "obs"]:
            for attribute in self.get_embodiment_attributes(type_, episode):
                embodiment = attribute["embodiment"]
                component = attribute["component"]
                name = attribute["name"]
                base_path = (
                    output / "episodes" / episode["id"] / type_ / embodiment.name
                )
                # 0.3.0 state.parquet (qpos/qvel/qtorque) is shared across
                # attributes for the same component; copy it once.
                if attribute["path"].name == "state.parquet":
                    if component:
                        new_path = base_path / component / "state.parquet"
                    else:
                        new_path = base_path / "state.parquet"
                    if new_path in written_state_paths:
                        continue
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(attribute["path"], new_path)
                    written_state_paths.add(new_path)
                    continue
                if component:
                    new_path = base_path / component / f"{name}.parquet"
                else:
                    new_path = base_path / f"{name}.parquet"
                new_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.read_parquet(attribute["path"])
                # No version and 0.1.0 use "positions"
                if "positions" in df:
                    df["value"] = df["positions"]
                    df = df.drop(columns=["positions"])
                    df.to_parquet(new_path)
                else:
                    shutil.copy2(attribute["path"], new_path)

    def _write_camera_data(
        self, output: os.PathLike, episode: dict, camera_format: str = "dir"
    ):
        base_path = output / "episodes" / episode["id"]
        for name, camera in self.load_cameras(episode).items():
            if self.meta.version is None:
                if name == "left_wrist":
                    name = "wrist_left"
                elif name == "right_wrist":
                    name = "wrist_right"
            cameras_dir = base_path / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            camera.write(cameras_dir / name, camera_format)
