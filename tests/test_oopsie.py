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

from pathlib import Path
import json

import h5py
import pytest
import yaml

from openarm_dataset import Dataset
from openarm_dataset.oopsie import SCHEMA_VERSION, ANNOTATION_SCHEMA_VERSION

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30

# Mirrors oopsie_tools.utils.robot_profile.robot_profile.REQUIRED_KEYS, without
# importing the (optional, separate) oopsie-tools package at test time.
ROBOT_PROFILE_REQUIRED_KEYS = frozenset(
    {
        "policy_name",
        "robot_name",
        "gripper_name",
        "control_freq",
        "is_biarm",
        "uses_mobile_base",
        "camera_names",
        "robot_state_keys",
        "action_space",
    }
)


def _episode_ids(dataset: Dataset) -> list[str]:
    return [str(episode["id"]) for episode in dataset.meta.episodes]


def _str_attr(obj, key: str) -> str:
    """Read an HDF5 attr as ``str``, tolerating bytes (h5py version differences)."""
    value = obj.attrs[key]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def test_oopsie_write_basic(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.write(
        tmp_path,
        format="oopsie",
        fps=FPS,
        lab_id="Test Lab",
        operator_name="Test Operator",
    )

    for episode in dataset.meta.episodes:
        episode_id = str(episode["id"])
        h5_path = tmp_path / f"{episode_id}.h5"
        assert h5_path.exists()

        for camera_name in dataset.camera_names:
            video_path = tmp_path / f"{episode_id}_{camera_name}.mp4"
            assert video_path.exists(), f"missing video: {video_path}"

        with h5py.File(h5_path, "r") as f:
            assert _str_attr(f, "schema") == SCHEMA_VERSION
            assert _str_attr(f, "episode_id") == episode_id
            assert _str_attr(f, "lab_id") == "Test Lab"
            assert _str_attr(f, "operator_name") == "Test Operator"
            expected_prompt = dataset.meta.tasks[int(episode["task_index"])]["prompt"]
            assert _str_attr(f, "language_instruction") == expected_prompt

            profile = json.loads(_str_attr(f, "robot_profile"))
            assert ROBOT_PROFILE_REQUIRED_KEYS <= profile.keys()
            assert profile["is_biarm"] is True
            assert profile["camera_names"] == list(dataset.camera_names)
            assert profile["robot_state_keys"] == ["joint_position", "gripper_position"]
            assert profile["action_space"] == ["joint_position", "gripper_position"]
            assert len(profile["robot_state_joint_names"]) == 16

            for cam in dataset.camera_names:
                assert f["observations/video_paths"][cam][()].decode() == (
                    f"{episode_id}_{cam}.mp4"
                )

            joint_position = f["observations/robot_states/joint_position"][()]
            gripper_position = f["observations/robot_states/gripper_position"][()]
            assert joint_position.shape[1] == 16
            assert gripper_position.shape == (joint_position.shape[0], 2)
            assert f["actions/joint_position"].shape == joint_position.shape
            assert f["actions/gripper_position"].shape == gripper_position.shape

            # The "lifter" embodiment (present in this fixture alongside "arms")
            # is passed through as an extra, non-required key.
            assert f["observations/robot_states/lifter_elevation"].shape[1] == 1
            assert f["actions/lifter_elevation"].shape[1] == 1

            assert "episode_annotations" not in f


def test_oopsie_eval_metadata_merge(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    episode_ids = _episode_ids(dataset)
    assert "0" in episode_ids and "3" in episode_ids

    eval_metadata_path = tmp_path / "eval_metadata.yaml"
    eval_metadata_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": {
                    "path": "/data/pi_checkpoints/example/1000/",
                    "auto_info": {"captured_at": "2026-01-01T00:00:00"},
                },
                "episodes": [
                    {
                        "id": "0",
                        "task_index": int(
                            next(
                                e
                                for e in dataset.meta.episodes
                                if str(e["id"]) == "0"
                            )["task_index"]
                        ),
                        "success": False,
                        "note": "test failure note",
                        "source": "rollout",
                        "timestamp": "2026-01-02T03:04:05",
                    },
                    {
                        "id": "99",  # not present in the dataset
                        "task_index": 0,
                        "success": True,
                        "note": "orphaned entry",
                        "source": "rollout",
                        "timestamp": "2026-01-02T03:04:05",
                    },
                ],
            }
        )
    )

    output_dir = tmp_path / "out"
    with pytest.warns(UserWarning, match="episode ids not present"):
        dataset.write(
            output_dir,
            format="oopsie",
            fps=FPS,
            lab_id="Test Lab",
            operator_name="Test Operator",
            eval_metadata_path=eval_metadata_path,
        )

    with h5py.File(output_dir / "0.h5", "r") as f:
        group = f["episode_annotations/rollout_eval"]
        assert _str_attr(group, "schema") == ANNOTATION_SCHEMA_VERSION
        assert _str_attr(group, "source") == "rollout"
        assert group.attrs["success"] == 0.0
        assert _str_attr(group, "failure_description") == "test failure note"
        assert _str_attr(f, "source_checkpoint_path") == "/data/pi_checkpoints/example/1000/"
        assert _str_attr(f, "source_checkpoint_captured_at") == "2026-01-01T00:00:00"

    with h5py.File(output_dir / "3.h5", "r") as f:
        # No eval_metadata entry for episode "3": left un-annotated, but the
        # dataset-wide checkpoint provenance is still recorded.
        assert "episode_annotations" not in f
        assert _str_attr(f, "source_checkpoint_path") == "/data/pi_checkpoints/example/1000/"


def test_oopsie_eval_metadata_task_index_mismatch_warns(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    episode = next(e for e in dataset.meta.episodes if str(e["id"]) == "0")

    eval_metadata_path = tmp_path / "eval_metadata.yaml"
    eval_metadata_path.write_text(
        yaml.safe_dump(
            {
                "episodes": [
                    {
                        "id": "0",
                        "task_index": int(episode["task_index"]) + 1,
                        "success": True,
                        "note": "",
                        "source": "rollout",
                        "timestamp": "2026-01-02T03:04:05",
                    },
                ],
            }
        )
    )

    output_dir = tmp_path / "out"
    with pytest.warns(UserWarning, match="task_index"):
        dataset.write(
            output_dir,
            format="oopsie",
            fps=FPS,
            lab_id="Test Lab",
            operator_name="Test Operator",
            eval_metadata_path=eval_metadata_path,
        )

    # Despite the mismatched eval_metadata task_index, language_instruction
    # still comes from the dataset's own metadata.yaml.
    with h5py.File(output_dir / "0.h5", "r") as f:
        expected_prompt = dataset.meta.tasks[int(episode["task_index"])]["prompt"]
        assert _str_attr(f, "language_instruction") == expected_prompt


def test_oopsie_invalid_fps(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    with pytest.raises(ValueError, match="fps"):
        dataset.write(tmp_path, format="oopsie", fps=0)


def test_oopsie_missing_eval_metadata_path(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    with pytest.raises(FileNotFoundError):
        dataset.write(
            tmp_path,
            format="oopsie",
            fps=FPS,
            eval_metadata_path=tmp_path / "does_not_exist.yaml",
        )
