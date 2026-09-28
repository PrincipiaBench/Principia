"""Pendulum evaluation.

Measures each pendulum's string length on the first frame and its period,
taken as twice the time until the bob first turns back on the far side of
its pivot. Object 1 is released on the left and swings right; object 2 is
released on the right and swings left.

Usage::

    python -m principia.scenarios.pendulum /path/to/<model_dir>
"""

import math
from typing import Optional

from ..common import (
    Point,
    Scenario,
    Tracks,
    VideoInfo,
    annotation_scale,
    points_to_annotation,
    reference_point,
    run_scenario_cli,
)


def find_turning_frame(
    centroids: list[Optional[Point]], pivot_x: float, direction: int
) -> Optional[int]:
    """Return the frame at which the bob turns back past the pivot.

    ``direction`` is +1 for a bob released on the left (swinging right) and
    -1 for one released on the right. Returns None if the bob never turns, or
    if it is lost before it does.
    """
    for i in range(1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            return None
        dx = centroids[i][0] - centroids[i - 1][0]
        if direction * dx < 0 and direction * (centroids[i][0] - pivot_x) > 0:
            return i - 1
    return None


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    tops = [reference_point(metadata, f"string_top_{k}", video) for k in (1, 2)]
    if any(c[0] is None for c in tracks.centroids):
        raise ValueError("Object not found on the first frame; cannot measure string length")

    lengths = [math.dist(c[0], top) for c, top in zip(tracks.centroids, tops)]
    tp_frames = [
        find_turning_frame(c, top[0], direction)
        for c, top, direction in zip(tracks.centroids, tops, (1, -1))
    ]
    periods = [2 * f / video.fps if f is not None else None for f in tp_frames]

    sx, sy = annotation_scale(video)
    centroids_global = [points_to_annotation(c, video) for c in tracks.centroids]
    tops_global = [(t[0] * sx, t[1] * sy) for t in tops]
    lengths_global = [math.dist(c[0], top) for c, top in zip(centroids_global, tops_global)]

    return {
        "length1_model": lengths[0],
        "length2_model": lengths[1],
        "time_period_gen_1": periods[0],
        "time_period_gen_2": periods[1],
        "tp_frame_1": tp_frames[0],
        "tp_frame_2": tp_frames[1],

        "centroid_positions_gen_1_model": tracks.centroids[0],
        "centroid_positions_gen_2_model": tracks.centroids[1],
        "string_top_1_model": tops[0],
        "string_top_2_model": tops[1],

        "length1_global": lengths_global[0],
        "length2_global": lengths_global[1],
        "centroid_positions_gen_1_global": centroids_global[0],
        "centroid_positions_gen_2_global": centroids_global[1],
        "string_top_1_global": tops_global[0],
        "string_top_2_global": tops_global[1],

        "video_resolution": list(video.resolution),
    }


SCENARIO = Scenario("pendulum", compute_metrics)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
