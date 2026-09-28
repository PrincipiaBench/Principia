"""Moment-of-inertia evaluation.

Records, for the hollow and solid objects rolling down the incline, the first
frame at which the bottom of each mask reaches the ground line. The
measurement is the same as for friction; the reference also records which
side the hollow object starts on.

Usage::

    python -m principia.scenarios.moment_of_inertia /path/to/<model_dir>
"""

from ..common import Scenario, Tracks, VideoInfo, run_scenario_cli
from .coeff_friction import compute_metrics as ground_contact_metrics


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    results = ground_contact_metrics(metadata, video, tracks)
    results["hollow_is_left"] = metadata["hollow_is_left"]
    return results


SCENARIO = Scenario("moment_of_inertia", compute_metrics)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
