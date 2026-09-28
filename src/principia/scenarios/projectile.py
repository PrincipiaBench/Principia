"""Projectile evaluation.

Each ball rolls off a ledge (object 1 off the left edge, object 2 off the
right) and lands on the ground. Measures the horizontal distance from the edge
to the landing point and scores whether the ordering of the two distances
matches the release heights in the reference.

Usage::

    python -m principia.scenarios.projectile /path/to/<model_dir>
"""

from typing import Callable, Optional

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

# A ball has landed once it is below the ground line and moves down by less
# than this many pixels between frames.
LANDING_DY_THRESHOLD = 1
# Distances within this many pixels count as equal when release heights match.
PIXEL_TOLERANCE = 20


def find_landing(
    centroids: list[Optional[Point]], ground_y: float, past_edge: Callable[[float], bool]
) -> tuple[Optional[int], Optional[int]]:
    """Return (landing frame, first frame past the ledge edge) for one ball.

    Either value is None if it is not observed. The landing frame is None if
    the ball is lost before it lands.
    """
    edge_frame = None
    for i in range(1, len(centroids)):
        if centroids[i] is None or centroids[i - 1] is None:
            return None, edge_frame
        x, y = centroids[i]
        if edge_frame is None and past_edge(x):
            edge_frame = i
        if y > ground_y and y - centroids[i - 1][1] < LANDING_DY_THRESHOLD:
            return i - 1, edge_frame
    return None, edge_frame


def score_distances(pos_1: float, pos_2: float, dist_1: float, dist_2: float) -> int:
    """Return 1 if the ball released higher travels further, else 0."""
    if pos_1 > pos_2:
        return int(dist_1 > dist_2)
    if pos_1 < pos_2:
        return int(dist_1 < dist_2)
    return int(abs(dist_1 - dist_2) <= PIXEL_TOLERANCE)


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    ground_y = reference_point(metadata, "ground", video)[1]
    left_x = reference_point(metadata, "left_extreme", video)[0]
    right_x = reference_point(metadata, "right_extreme", video)[0]
    pos_1, pos_2 = metadata["pos1"], metadata["pos2"]
    centroids_1, centroids_2 = tracks.centroids

    drop_1, left_edge_frame = find_landing(centroids_1, ground_y, lambda x: x < left_x)
    drop_2, right_edge_frame = find_landing(centroids_2, ground_y, lambda x: x > right_x)

    impact_x_1 = centroids_1[drop_1][0] if drop_1 is not None else None
    impact_x_2 = centroids_2[drop_2][0] if drop_2 is not None else None
    # A non-positive distance means the ball landed on the wrong side of its edge.
    dist_1 = left_x - impact_x_1 if impact_x_1 is not None else None
    dist_2 = impact_x_2 - right_x if impact_x_2 is not None else None
    dist_1 = dist_1 if dist_1 is not None and dist_1 > 0 else None
    dist_2 = dist_2 if dist_2 is not None and dist_2 > 0 else None

    if dist_1 is not None and dist_2 is not None:
        gen_score = score_distances(pos_1, pos_2, dist_1, dist_2)
    else:
        # A ball never landed beyond its edge, so the ordering cannot hold.
        gen_score = 0

    sx, sy = annotation_scale(video)
    return {
        "pos_ball1": pos_1,
        "pos_ball2": pos_2,
        "gen_score": gen_score,
        "dist_gen_1": dist_1,
        "dist_gen_2": dist_2,
        "impact_x_gen_1": impact_x_1,
        "impact_x_gen_2": impact_x_2,
        "drop_frame_gen_1": drop_1,
        "drop_frame_gen_2": drop_2,
        "left_extreme_drop_frame_gen_1": left_edge_frame,
        "right_extreme_drop_frame_gen_2": right_edge_frame,

        "ground_gt_threshold_model": ground_y,
        "left_extreme_x_model": left_x,
        "right_extreme_x_model": right_x,
        "centroid_positions_gen_1_model": centroids_1,
        "centroid_positions_gen_2_model": centroids_2,

        "ground_gt_threshold_global": ground_y * sy,
        "left_extreme_x_global": left_x * sx,
        "right_extreme_x_global": right_x * sx,
        "centroid_positions_gen_1_global": points_to_annotation(centroids_1, video),
        "centroid_positions_gen_2_global": points_to_annotation(centroids_2, video),
    }


SCENARIO = Scenario("projectile", compute_metrics)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
