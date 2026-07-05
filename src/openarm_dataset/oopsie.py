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

"""Conversion script for OpenArm Dataset to the Oopsie annotation format.

Writes one ``oopsiedata_format_v1`` HDF5 file per episode plus one MP4 per
camera, so rollouts collected with OpenArm Dataset can be reviewed and
annotated with the oopsie-tools annotation UI
(https://github.com/oopsie-data/oopsie-tools).

If a rollout eval harness wrote a sidecar ``eval_metadata.yaml`` next to the
dataset's ``metadata.yaml`` (with per-episode ``success``/``note``/``source``/
``timestamp`` fields), it is merged in as a pre-filled
``episode_annotations/<annotator_name>`` group so a human annotator starts
from the rollout's own notes instead of a blank slate. The dataset's own
``metadata.yaml`` remains the sole source of truth for ``task_index`` /
``language_instruction``; only ``success``/``note``/``source``/``timestamp``
are taken from ``eval_metadata.yaml``, matched by episode ``id``.
"""

from __future__ import annotations

import datetime
import json
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from .dataset import Dataset
from .ffmpeg import encode_mp4
from .metadata import OpenArm

SCHEMA_VERSION = "oopsiedata_format_v1"
ANNOTATION_SCHEMA_VERSION = "oopsie_failure_taxonomy_v1"
DEFAULT_ANNOTATOR_NAME = "rollout_eval"
EVAL_METADATA_FILENAME = "eval_metadata.yaml"


def _joint_name(component: str | None, joint: str) -> str:
    return f"{component}_{joint}.pos" if component else f"{joint}.pos"


def _find_arms_embodiment(dataset: Dataset) -> tuple[str, OpenArm]:
    """Return the (name, embodiment) of the dataset's bimanual OpenArm embodiment."""
    for name, embodiment in dataset.meta.equipment.embodiments.items():
        if isinstance(embodiment, OpenArm):
            return name, embodiment
    raise ValueError(
        "No OpenArm embodiment found in dataset; the oopsie export requires "
        "bimanual arms data."
    )


def _arms_layout(name: str, embodiment: OpenArm) -> tuple[list[str], list[str]]:
    """Return (obs/action keys per component, joint names), in concatenation order."""
    keys = [f"{name}/{component}/qpos" for component in embodiment.components]
    joint_names = [
        _joint_name(component, joint)
        for component in embodiment.components
        for joint in embodiment.joints
    ]
    return keys, joint_names


def _extra_attribute_key(name: str, component: str | None, attribute: str) -> str:
    return f"{name}_{component}_{attribute}" if component else f"{name}_{attribute}"


def _collect_sample_arrays(
    dataset: Dataset,
    episode: dict,
    samples: list,
    arms_name: str,
    arms_embodiment: OpenArm,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build (T, D) float64 observations/actions dicts for one episode's samples.

    ``joint_position`` concatenates every arm component's qpos (joints +
    trailing gripper); ``gripper_position`` slices out just the trailing
    gripper column per component. Any other embodiment present (e.g. a
    lifter) is passed through under its own extra key, but is not wired
    into the robot_profile's required ``robot_state_keys``/``action_space``.
    """
    arms_keys, _ = _arms_layout(arms_name, arms_embodiment)
    num_joints = len(arms_embodiment.joints)
    gripper_cols = [i * num_joints + num_joints - 1 for i in range(len(arms_keys))]

    def _build(type_: str) -> dict[str, np.ndarray]:
        values: dict[str, np.ndarray] = {}
        joint_position = np.stack(
            [
                np.concatenate([sample[type_][key] for key in arms_keys])
                for sample in samples
            ]
        ).astype(np.float64)
        values["joint_position"] = joint_position
        values["gripper_position"] = joint_position[:, gripper_cols]

        for attribute in dataset.get_embodiment_attributes(type_, episode):
            if isinstance(attribute["embodiment"], OpenArm):
                continue
            key = _extra_attribute_key(
                attribute["embodiment"].name, attribute["component"], attribute["name"]
            )
            values[key] = np.stack(
                [
                    np.atleast_1d(sample[type_][attribute["key"]])
                    for sample in samples
                ]
            ).astype(np.float64)
        return values

    return _build("obs"), _build("action")


def _build_robot_profile(
    dataset: Dataset,
    arms_name: str,
    arms_embodiment: OpenArm,
    fps: int,
    policy_name: str,
    gripper_name: str,
) -> dict[str, Any]:
    """Build a dict matching oopsie-tools' RobotProfile field names/shape."""
    _, joint_names = _arms_layout(arms_name, arms_embodiment)
    return {
        "policy_name": policy_name,
        "robot_name": f"{arms_embodiment.id}_{arms_embodiment.version}",
        "gripper_name": gripper_name,
        "is_biarm": len(arms_embodiment.components) >= 2,
        "uses_mobile_base": False,
        "control_freq": fps,
        "camera_names": list(dataset.camera_names),
        "robot_state_keys": ["joint_position", "gripper_position"],
        "robot_state_joint_names": joint_names,
        "action_space": ["joint_position", "gripper_position"],
        "action_joint_names": joint_names,
        "orientation_representation": None,
        "robot_state_orientation_representation": None,
        "controller": None,
        "gains": None,
        "intrinsic_calibration_matrix": None,
        "extrinsic_calibration_matrix": None,
    }


def _load_eval_metadata(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    episodes = {}
    for entry in data.get("episodes") or []:
        episodes[str(entry.get("id"))] = entry
    return {"episodes": episodes, "checkpoint": data.get("checkpoint")}


def _discover_eval_metadata_path(dataset: Dataset) -> Path | None:
    candidate = dataset.root_path / EVAL_METADATA_FILENAME
    return candidate if candidate.is_file() else None


def _parse_iso_timestamp(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _write_annotation(
    f: h5py.File,
    episode_eval: dict[str, Any] | None,
    annotator_name: str,
) -> None:
    """Pre-fill episode_annotations/<annotator_name> from a rollout eval_metadata entry.

    Only written when eval_metadata has a usable ``success`` value for this
    episode; otherwise the episode is left un-annotated, ready for a human
    to fill in from scratch via the oopsie-tools annotation UI.
    """
    if episode_eval is None or episode_eval.get("success") is None:
        return
    timestamp = _parse_iso_timestamp(
        str(episode_eval.get("timestamp", ""))
    ) or datetime.datetime.now().timestamp()

    group = f.create_group("episode_annotations").create_group(annotator_name)
    group.attrs["schema"] = ANNOTATION_SCHEMA_VERSION
    group.attrs["source"] = str(episode_eval.get("source", "rollout"))
    group.attrs["timestamp"] = float(timestamp)
    group.attrs["success"] = 1.0 if episode_eval["success"] else 0.0
    group.attrs["failure_description"] = str(episode_eval.get("note", ""))
    group.attrs["taxonomy_schema"] = "none"
    group.attrs["taxonomy"] = json.dumps({}, ensure_ascii=False)
    group.attrs["additional_notes"] = ""


def _write_episode_h5(
    dataset: Dataset,
    episode: dict,
    output_dir: Path,
    fps: int,
    lab_id: str,
    operator_name: str,
    annotator_name: str,
    robot_profile: dict[str, Any],
    arms_name: str,
    arms_embodiment: OpenArm,
    episode_eval: dict[str, Any] | None,
    checkpoint_info: dict[str, Any] | None,
) -> Path:
    episode_id = str(episode["id"])
    samples = dataset.sample(hz=fps, episode=episode)
    if not samples:
        raise ValueError(
            f"Episode {episode_id!r} has no synchronized samples at {fps} Hz."
        )

    observations, actions = _collect_sample_arrays(
        dataset, episode, samples, arms_name, arms_embodiment
    )

    camera_frames = {
        name: [sample.cameras[name] for sample in samples]
        for name in dataset.camera_names
    }
    video_rel_paths: dict[str, str] = {}
    for camera_name, frames in camera_frames.items():
        # Sibling of the .h5 (not a "videos/" subdirectory): the oopsie-tools
        # browse-only annotator falls back to globbing "<h5 stem>_*.mp4" next
        # to the .h5 file when it can't find an `image_observations` group.
        video_path = output_dir / f"{episode_id}_{camera_name}.mp4"
        encode_mp4(frames, fps, video_path)
        video_rel_paths[camera_name] = video_path.name

    language_instruction = dataset.meta.tasks[int(episode["task_index"])]["prompt"]

    timestamp = None
    if episode_eval is not None:
        timestamp = _parse_iso_timestamp(str(episode_eval.get("timestamp", "")))
    if timestamp is None:
        first_frames = next(iter(camera_frames.values()), None)
        timestamp = (
            first_frames[0].timestamp
            if first_frames
            else datetime.datetime.now().timestamp()
        )

    h5_path = output_dir / f"{episode_id}.h5"
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION
        f.attrs["episode_id"] = episode_id
        f.attrs["language_instruction"] = language_instruction
        f.attrs["lab_id"] = lab_id
        f.attrs["operator_name"] = operator_name
        f.attrs["timestamp"] = float(timestamp)
        f.attrs.create(
            "robot_profile",
            json.dumps(robot_profile, ensure_ascii=False),
            dtype=str_dtype,
        )
        f.attrs["source_dataset_path"] = str(dataset.root_path)
        f.attrs["source_episode_id"] = episode_id
        if checkpoint_info:
            if checkpoint_info.get("path"):
                f.attrs["source_checkpoint_path"] = str(checkpoint_info["path"])
            captured_at = (checkpoint_info.get("auto_info") or {}).get("captured_at")
            if captured_at:
                f.attrs["source_checkpoint_captured_at"] = str(captured_at)

        obs_group = f.create_group("observations")
        video_group = obs_group.create_group("video_paths")
        for camera_name, rel_path in video_rel_paths.items():
            video_group.create_dataset(camera_name, data=rel_path, dtype=str_dtype)

        robot_states = obs_group.create_group("robot_states")
        for key, values in observations.items():
            robot_states.create_dataset(key, data=values, dtype=np.float64)

        action_group = f.create_group("actions")
        for key, values in actions.items():
            action_group.create_dataset(key, data=values, dtype=np.float64)

        _write_annotation(f, episode_eval, annotator_name)

    return h5_path


def to_oopsie(
    dataset: Dataset,
    output_dir: str | Path,
    fps: int = 30,
    lab_id: str | None = None,
    operator_name: str | None = None,
    annotator_name: str = DEFAULT_ANNOTATOR_NAME,
    policy_name: str | None = None,
    gripper_name: str | None = None,
    eval_metadata_path: str | Path | None = None,
) -> None:
    """Convert the given dataset to the oopsie annotation format.

    Args:
        dataset: Dataset to convert.
        output_dir: Directory to write ``<episode_id>.h5`` and
            ``<episode_id>_<camera>.mp4`` files into.
        fps: Common sampling rate for observations, actions, and cameras.
        lab_id: ``lab_id`` HDF5 attribute (default: ``dataset.meta.location``).
        operator_name: ``operator_name`` HDF5 attribute (default:
            ``dataset.meta.operator``).
        annotator_name: Annotator name under which rollout notes from
            ``eval_metadata.yaml`` are pre-filled into
            ``episode_annotations/<annotator_name>``. Pass the same name to
            oopsie-tools' annotator server (``--annotator-name``) to review
            and refine these pre-filled notes in place.
        policy_name: ``robot_profile.policy_name`` (default:
            ``dataset.meta.operation_type``).
        gripper_name: ``robot_profile.gripper_name`` (default: derived from
            the arms embodiment id).
        eval_metadata_path: Path to a rollout ``eval_metadata.yaml`` to merge
            in. Defaults to a sibling ``eval_metadata.yaml`` next to the
            dataset's own ``metadata.yaml``, if one exists.

    Raises:
        ValueError: If ``fps`` is not positive, the dataset has no episodes,
            no OpenArm embodiment is found, or ``lab_id``/``operator_name``
            cannot be resolved.
        FileNotFoundError: If an explicit ``eval_metadata_path`` does not exist.

    """
    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")
    if not dataset.meta.episodes:
        raise ValueError("No episodes to write.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arms_name, arms_embodiment = _find_arms_embodiment(dataset)

    lab_id = lab_id or dataset.meta.location
    operator_name = operator_name or dataset.meta.operator
    if not lab_id:
        raise ValueError(
            "lab_id is required (dataset.meta.location is empty); pass lab_id explicitly."
        )
    if not operator_name:
        raise ValueError(
            "operator_name is required (dataset.meta.operator is empty); "
            "pass operator_name explicitly."
        )

    if eval_metadata_path is not None:
        eval_metadata_path = Path(eval_metadata_path)
        if not eval_metadata_path.is_file():
            raise FileNotFoundError(f"eval_metadata_path not found: {eval_metadata_path}")
    else:
        eval_metadata_path = _discover_eval_metadata_path(dataset)

    eval_metadata = _load_eval_metadata(eval_metadata_path) if eval_metadata_path else None
    checkpoint_info = eval_metadata.get("checkpoint") if eval_metadata else None
    eval_episodes: dict[str, Any] = eval_metadata.get("episodes", {}) if eval_metadata else {}

    robot_profile = _build_robot_profile(
        dataset,
        arms_name,
        arms_embodiment,
        fps,
        policy_name or (dataset.meta.operation_type or "openarm_dataset"),
        gripper_name or f"{arms_embodiment.id}_gripper",
    )

    seen_eval_ids = set()
    for episode in dataset.meta.episodes:
        episode_id = str(episode["id"])
        episode_eval = eval_episodes.get(episode_id)
        if episode_eval is not None:
            seen_eval_ids.add(episode_id)
            eval_task_index = episode_eval.get("task_index")
            if eval_task_index is not None and str(eval_task_index) != str(
                episode["task_index"]
            ):
                warnings.warn(
                    f"eval_metadata task_index for episode {episode_id!r} "
                    f"({eval_task_index}) differs from the dataset's own "
                    f"task_index ({episode['task_index']}); using the dataset's "
                    "task_index for language_instruction and ignoring "
                    "eval_metadata's task_index.",
                    stacklevel=2,
                )
        _write_episode_h5(
            dataset=dataset,
            episode=episode,
            output_dir=output_dir,
            fps=fps,
            lab_id=lab_id,
            operator_name=operator_name,
            annotator_name=annotator_name,
            robot_profile=robot_profile,
            arms_name=arms_name,
            arms_embodiment=arms_embodiment,
            episode_eval=episode_eval,
            checkpoint_info=checkpoint_info,
        )

    missing_ids = set(eval_episodes) - seen_eval_ids
    if missing_ids:
        warnings.warn(
            "eval_metadata.yaml has entries for episode ids not present in "
            f"the dataset: {sorted(missing_ids)}",
            stacklevel=2,
        )
