"""Coefficient-of-friction evaluation.

Records, for each of the two objects sliding down the incline, the first frame
at which the bottom of its mask reaches the ground line.

Usage::

    python -m principia.scenarios.coeff_friction /path/to/<model_dir>
"""

from typing import Optional

from ..common import (
    Scenario,
    Tracks,
    VideoInfo,
    annotation_scale,
    points_to_annotation,
    reference_point,
    run_scenario_cli,
    ys_to_annotation,
)


def find_ground_contact_frame(bottoms: list[Optional[int]], ground_y: float) -> Optional[int]:
    """Return the first frame (from 1) whose mask bottom reaches the ground.

    Returns None if the object never reaches the ground, or if its mask is
    lost before it does.
    """
    for i in range(1, len(bottoms)):
        if bottoms[i] is None or bottoms[i - 1] is None:
            return None
        if bottoms[i] >= ground_y:
            return i
    return None


def compute_metrics(metadata: dict, video: VideoInfo, tracks: Tracks) -> dict:
    ground_y = reference_point(metadata, "ground", video)[1]
    drop_frames = [find_ground_contact_frame(b, ground_y) for b in tracks.bottoms]

    _, sy = annotation_scale(video)
    bottoms_global = [ys_to_annotation(b, video) for b in tracks.bottoms]
    centroids_global = [points_to_annotation(c, video) for c in tracks.centroids]

    return {
        "drop_frame_gen_1": drop_frames[0],
        "drop_frame_gen_2": drop_frames[1],

        "ground_gt_threshold_global": ground_y * sy,
        "bottom_positions_generated_1_global": bottoms_global[0],
        "bottom_positions_generated_2_global": bottoms_global[1],
        "centroid_positions_gen_1_global": centroids_global[0],
        "centroid_positions_gen_2_global": centroids_global[1],

        "ground_gt_threshold_model": ground_y,
        "bottom_positions_generated_1_model": tracks.bottoms[0],
        "bottom_positions_generated_2_model": tracks.bottoms[1],
        "centroid_positions_gen_1_model": tracks.centroids[0],
        "centroid_positions_gen_2_model": tracks.centroids[1],

        "video_resolution": list(video.resolution),
    }


SCENARIO = Scenario("coeff_friction", compute_metrics)


if __name__ == "__main__":
    run_scenario_cli(SCENARIO, __doc__.split("\n")[0])
