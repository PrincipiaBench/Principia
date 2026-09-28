"""Directional-consistency filtering for the Principia eval scripts.

Run after the scenario scripts. For each sample's ``results.json``, checks
that each object moves in the expected direction along a single axis, using
only that axis of the frame-to-frame motion. Every scenario uses the vertical
(downward) check except pendulum, which uses the horizontal check (object 1
swings right, object 2 swings left).

Each step scores +1 if it moves in the expected direction and -1 if not;
steps shorter than ``EPS`` pixels along the axis are ignored. The score is
the mean over the steps from frame 0 up to the scenario's end frame.

Usage::

    python -m principia.direction_consistency /path/to/<model_dir>
    python -m principia.direction_consistency /path/to/<model_dir> --scenarios pendulum spring

Results are written next to each ``results.json`` as
``direction_consistency_vertical_only.json`` (or
``direction_consistency_horizontal_only.json`` for pendulum), which
``principia.invariance_score`` reads.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .common import list_samples, setup_logging

logger = logging.getLogger(__name__)

EPS = 2.0  # ignore frame-to-frame motions smaller than this (pixels)

VERTICAL_NAME = "direction_consistency_vertical_only.json"
HORIZONTAL_NAME = "direction_consistency_horizontal_only.json"


@dataclass
class DirectionCheck:
    axis: int  # 0 for x, 1 for y
    signs: tuple[float, float]  # expected sign of motion for object 1 and 2
    positions_keys: tuple[str, str]
    # Results keys of the last frame scored for each object. None scores the
    # whole video and skips steps where the object is lost.
    end_frame_keys: Optional[tuple[str, str]]
    output_name: str


GLOBAL_CENTROIDS = ("centroid_positions_gen_1_global", "centroid_positions_gen_2_global")
DOWN = (1.0, 1.0)


def vertical(end_frame_keys, positions_keys=GLOBAL_CENTROIDS) -> DirectionCheck:
    return DirectionCheck(1, DOWN, positions_keys, end_frame_keys, VERTICAL_NAME)


CHECKS = {
    "coeff_friction": vertical(("drop_frame_gen_1", "drop_frame_gen_2")),
    "moment_of_inertia": vertical(("drop_frame_gen_1", "drop_frame_gen_2")),
    "restitution": vertical(("drop_frame_gen_1", "drop_frame_gen_2")),
    "spring": vertical(("drop_frame_1", "drop_frame_2")),
    # Scored until each ball leaves its ledge.
    "projectile": vertical(("left_extreme_drop_frame_gen_1", "right_extreme_drop_frame_gen_2")),
    "conservation_of_momentum": vertical(None, ("centroid_positions_gen_1", "centroid_positions_gen_2")),
    "pendulum": DirectionCheck(0, (1.0, -1.0), GLOBAL_CENTROIDS, ("tp_frame_1", "tp_frame_2"), HORIZONTAL_NAME),
}


def axis_consistency(
    positions: Optional[list],
    end_frame: Optional[int],
    axis: int,
    expected_sign: float,
    whole_video: bool,
) -> tuple[Optional[float], int]:
    """Return (score in [-1, 1] or None, number of steps scored) for one object.

    With ``whole_video`` the full trajectory is used and steps touching a
    lost frame are skipped. Otherwise frames 0..end_frame are used, and the
    score is None if the end frame is unknown or the object is lost within it.
    """
    if positions is None:
        return None, 0
    if not whole_video:
        if end_frame is None:
            return None, 0
        positions = positions[: int(end_frame) + 1]
        if any(p is None for p in positions):
            return None, 0

    scores = []
    for p1, p2 in zip(positions[:-1], positions[1:]):
        if p1 is None or p2 is None:
            continue
        component = p2[axis] - p1[axis]
        if abs(component) < EPS:
            continue
        scores.append(1.0 if np.sign(component) == np.sign(expected_sign) else -1.0)

    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)


def compute_direction_consistency(check: DirectionCheck, data: dict) -> dict:
    out = {}
    for i in (0, 1):
        end_frame = data.get(check.end_frame_keys[i]) if check.end_frame_keys else None
        score, n_steps = axis_consistency(
            data.get(check.positions_keys[i]),
            end_frame,
            check.axis,
            check.signs[i],
            whole_video=check.end_frame_keys is None,
        )
        out[f"direction_consistency_gen_{i + 1}"] = score
        out[f"end_frame_gen_{i + 1}"] = end_frame
        out[f"num_steps_used_gen_{i + 1}"] = n_steps
        out[f"expected_sign_gen_{i + 1}"] = check.signs[i]

    valid = [s for s in (out["direction_consistency_gen_1"], out["direction_consistency_gen_2"]) if s is not None]
    out["direction_consistency_mean"] = float(np.mean(valid)) if valid else None
    out["expected_axis"] = "xy"[check.axis]
    out["eps_pixels"] = EPS
    return out


def run_direction_consistency(
    model_dir: Path, scenarios: Optional[list[str]] = None, output_name: str = "results.json"
) -> None:
    """Write the direction-consistency file for every sample with a results file."""
    for scenario in scenarios or CHECKS:
        check = CHECKS[scenario]
        scenario_dir = model_dir / scenario
        if not scenario_dir.is_dir():
            logger.info("[%s] direction consistency skipped: %s not found", scenario, scenario_dir)
            continue

        n_done = n_failed = 0
        for sample_dir in list_samples(scenario_dir):
            results_path = sample_dir / output_name
            if not results_path.is_file():
                continue
            try:
                with open(results_path) as f:
                    data = json.load(f)
                out = compute_direction_consistency(check, data)
                with open(sample_dir / check.output_name, "w") as f:
                    json.dump(out, f, indent=4)
                n_done += 1
                logger.debug(
                    "[%s] %s: gen_1=%s gen_2=%s", scenario, sample_dir.name,
                    out["direction_consistency_gen_1"], out["direction_consistency_gen_2"],
                )
            except Exception:
                logger.exception("[%s] %s: direction consistency failed", scenario, sample_dir.name)
                n_failed += 1
        logger.info("[%s] direction consistency: %d written, %d failed", scenario, n_done, n_failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", type=Path, help="Folder holding the model's generated scenarios.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(CHECKS),
        default=list(CHECKS),
        metavar="SCENARIO",
        help=f"Scenarios to process (default: all). Choices: {', '.join(CHECKS)}.",
    )
    parser.add_argument("--output-name", default="results.json", help="Results filename in each sample folder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every sample's scores.")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.model_dir.is_dir():
        raise SystemExit(f"Directory not found: {args.model_dir}")
    run_direction_consistency(args.model_dir.resolve(), args.scenarios, args.output_name)


if __name__ == "__main__":
    main()
