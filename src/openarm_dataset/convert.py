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

"""Convert OpenArm dataset format."""

import argparse
import openarm_dataset
import pathlib


def main():
    """Convert OpenArm dataset format."""
    parser = argparse.ArgumentParser(description="Convert OpenArm dataset format")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to be converted",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Path of converted OpenArm dataset",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--format",
        help="Format of the output dataset (default: openarm)",
        default="openarm",
        choices=["openarm", "lerobot_v2.1", "lerobot_v3.0", "gr00t", "oopsie"],
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for the output dataset (default: 30) if the output format is lerobot_v2.1, lerobot_v3.0, gr00t or oopsie",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--smoothing-cutoff",
        help="Cutoff frequency for smoothing (default: 1.0) if the output format is lerobot_v2.1, lerobot_v3.0 or gr00t",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--train-split",
        help="Split ratio for training dataset (default: 0.8) if the output format is lerobot_v2.1, lerobot_v3.0 or gr00t",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--success-only",
        help="Include only successful episodes in the output dataset (default: False) if the output format is lerobot_v2.1, lerobot_v3.0 or gr00t",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--camera-format",
        help="How to store camera frames when the output format is openarm: "
        "'dir' (one JPEG file per frame, default) or 'tar' (one .tar archive "
        "per camera, recommended for Hugging Face Hub file-count limits)",
        default="dir",
        choices=["dir", "tar"],
    )
    parser.add_argument(
        "--lab-id",
        help="lab_id attribute for the output episodes (default: the source "
        "dataset's location) if the output format is oopsie",
        default=None,
    )
    parser.add_argument(
        "--operator-name",
        help="operator_name attribute for the output episodes (default: the "
        "source dataset's operator) if the output format is oopsie",
        default=None,
    )
    parser.add_argument(
        "--annotator-name",
        help="Annotator name under which rollout notes from --eval-metadata "
        "are pre-filled (default: rollout_eval) if the output format is oopsie",
        default="rollout_eval",
    )
    parser.add_argument(
        "--policy-name",
        help="robot_profile.policy_name for the output episodes (default: "
        "derived from the source dataset) if the output format is oopsie",
        default=None,
    )
    parser.add_argument(
        "--gripper-name",
        help="robot_profile.gripper_name for the output episodes (default: "
        "derived from the source dataset) if the output format is oopsie",
        default=None,
    )
    parser.add_argument(
        "--eval-metadata",
        help="Path to a rollout eval_metadata.yaml to merge into "
        "episode_annotations (default: auto-discover a sibling "
        "eval_metadata.yaml next to the input dataset's metadata.yaml) if "
        "the output format is oopsie",
        type=pathlib.Path,
        default=None,
    )

    args = parser.parse_args()
    write_kwargs = {"format": args.format}
    if args.format in ("lerobot_v2.1", "lerobot_v3.0", "gr00t"):
        write_kwargs["fps"] = args.fps
        write_kwargs["smoothing_cutoff"] = args.smoothing_cutoff
        write_kwargs["train_split"] = args.train_split
        write_kwargs["success_only"] = args.success_only
    elif args.format == "oopsie":
        write_kwargs["fps"] = args.fps
        write_kwargs["lab_id"] = args.lab_id
        write_kwargs["operator_name"] = args.operator_name
        write_kwargs["annotator_name"] = args.annotator_name
        write_kwargs["policy_name"] = args.policy_name
        write_kwargs["gripper_name"] = args.gripper_name
        write_kwargs["eval_metadata_path"] = args.eval_metadata
    else:
        write_kwargs["camera_format"] = args.camera_format

    old_dataset = openarm_dataset.Dataset(args.input)
    old_dataset.write(args.output, **write_kwargs)


if __name__ == "__main__":
    main()
