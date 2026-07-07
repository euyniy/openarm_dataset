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

"""Validate OpenArm dataset."""

import argparse
import pathlib
import sys

import openarm_dataset


def main():
    """Validate OpenArm dataset."""
    parser = argparse.ArgumentParser(description="Validate OpenArm dataset")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to validate",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--qpos-jump-threshold",
        "--jump-threshold",
        dest="qpos_jump_threshold",
        type=float,
        default=1.0,
        metavar="RAD",
        help="Flag qpos frame-to-frame deltas above this value (rad); disabled if not set",
    )
    parser.add_argument(
        "--qpos-absmax",
        "--absmax-warn",
        dest="qpos_absmax",
        type=float,
        default=6.28,
        metavar="RAD",
        help="Flag qpos values whose absolute value exceeds this threshold; disabled if not set",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Flag episodes shorter than this duration (seconds); disabled if not set",
    )
    args = parser.parse_args()
    dataset = openarm_dataset.Dataset(args.input)
    valid = dataset.validate(
        on_error=lambda error: print(error, file=sys.stderr),
        qpos_jump_threshold=args.qpos_jump_threshold,
        qpos_absmax=args.qpos_absmax,
        min_duration=args.min_duration,
    )
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
