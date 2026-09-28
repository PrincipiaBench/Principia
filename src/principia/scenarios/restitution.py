"""Coefficient-of-restitution evaluation.

Each ball is dropped onto the ground. Measures the drop height, time of
flight, first impact and rebound height, and from these the coefficient of
restitution sqrt(rebound / drop) for each ball, plus the ratios between the
two balls' drop heights and squared flight times.

Usage::

    python -m principia.scenarios.restitution /path/to/<model_dir>
"""

from typing import Optional

import numpy as np

from ..common import (
    PORTRAIT,
    Point,
    Scenario,
    Tracks,
    VideoInfo,
    annotation_scale,
    points_to_annotation,
    reference_point,
    run_scenario_cli,
)

# A ball has hit the ground once it is below the ground line and moves down by
# less than this many pixels between frames.
IMPACT_DY_THRESHOLD = 1


def find_impact_frame(centroids: list[Optional[Point]], ground_y: float) -> Optional[int]:
    """Return the frame of first impact with the ground.

    Returns None if the ball never lands, or if it is lost before it does.
    """
    for i in range(1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            return None
        y, prev_y = centroids[i][1], centroids[i - 1][1]
        if y > ground_y and y - prev_y < IMPACT_DY_THRESHOLD:
            return i - 1
    return None


def find_rebound(
    centroids: list[Optional[Point]], impact_frame: int
) -> tuple[Optional[float], Optional[int]]:
    """Return (rebound height in pixels, frame of rebound peak) after an impact.

    The peak is the last frame before the ball starts falling again, or the
    final frame if it never does. Both values are None if the ball is lost
    before the peak.
    """
    impact_y = centroids[impact_frame][1]
    for i in range(impact_frame + 1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            return None, None
        if centroids[i][1] > centroids[i - 1][1]:
            return impact_y - centroids[i - 1][1], i - 1
    return impact_y - centroids[-1][1], len(centroids) - 1


def ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Return min(a, b) / max(a, b) for two positive values, else None."""
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return min(a, b) / max(a, b)


def measure_ball(centroids: list[Optional[Point]], ground_y: float, fps: float) -> dict:
    """Return the drop, rebound and restitution measurements for one ball."""
    m = dict.fromkeys(
        ("impact_frame", "time_of_flight", "start_y", "impact_y", "height",
         "rebound", "rebound_peak_frame", "coeff_rest")
    )
    impact = find_impact_frame(centroids, ground_y)
    if impact is None:
        return m

    m["impact_frame"] = impact
    m["time_of_flight"] = impact / fps
    m["start_y"] = centroids[0][1]
    m["impact_y"] = centroids[impact][1]
    m["height"] = m["impact_y"] - m["start_y"]
    m["rebound"], m["rebound_peak_frame"] = find_rebound(centroids, impact)
    if m["rebound"] is not None and m["height"] > 0 and m["rebound"] >= 0:
        m["coeff_rest"] = float(np.sqrt(m["rebound"] / m["height"]))
    return m


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    ground_y = reference_point(metadata, "ground", video)[1]
    b1, b2 = (measure_ball(c, ground_y, video.fps) for c in tracks.centroids)

    t_ratio = ratio(b1["time_of_flight"], b2["time_of_flight"])
    _, sy = annotation_scale(video)
    return {
        "coeff_rest_gen_1": b1["coeff_rest"],
        "coeff_rest_gen_2": b2["coeff_rest"],
        "height_ratio": ratio(b1["height"], b2["height"]),
        "time_of_flight_ratio_squared": t_ratio**2 if t_ratio is not None else None,
        "height_gen_1": b1["height"],
        "height_gen_2": b2["height"],
        "start_gen_y1": b1["start_y"],
        "impact_gen_y1": b1["impact_y"],
        "start_gen_y2": b2["start_y"],
        "impact_gen_y2": b2["impact_y"],
        "rebound_gen_1": b1["rebound"],
        "rebound_gen_2": b2["rebound"],
        "time_of_flight_gen_1": b1["time_of_flight"],
        "drop_frame_gen_1": b1["impact_frame"],
        "time_of_flight_gen_2": b2["time_of_flight"],
        "drop_frame_gen_2": b2["impact_frame"],

        "centroid_positions_gen_1_model": tracks.centroids[0],
        "centroid_positions_gen_2_model": tracks.centroids[1],
        "ground_gt_threshold_model": ground_y,
        "centroid_positions_gen_1_global": points_to_annotation(tracks.centroids[0], video),
        "centroid_positions_gen_2_global": points_to_annotation(tracks.centroids[1], video),
        "ground_gt_threshold_global": ground_y * sy,

        "rebound_peak_frame_gen_1": b1["rebound_peak_frame"],
        "rebound_peak_frame_gen_2": b2["rebound_peak_frame"],
    }


SCENARIO = Scenario("restitution", compute_metrics, PORTRAIT)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
