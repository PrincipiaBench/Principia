"""Run every Principia scenario evaluation on one model's outputs.

Loads SAM3 once and runs the scenario evaluations one after another. A
scenario whose folder is missing from the model directory is skipped, and a
scenario that fails does not stop the ones after it.

Usage::

    python -m principia /path/to/<model_dir>
    python -m principia /path/to/<model_dir> --scenarios pendulum spring

Each sample's results are written to
``<model_dir>/<scenario>/sample_<n>/results.json``, each scenario's failed
samples to ``<model_dir>/<scenario>/failures.json``, and an overall summary to
``<model_dir>/eval_summary.json``. Finally the directional-consistency files
are written and the invariance scores are computed from whichever results
files exist and printed.
"""

import argparse
import json
import logging

from . import direction_consistency
from . import invariance_score as invariance
from .common import (
    add_common_args,
    build_predictor,
    config_from_args,
    evaluate_scenario,
    find_scenario,
    setup_logging,
)
from .scenarios import (
    coeff_friction,
    conservation_of_momentum,
    moment_of_inertia,
    pendulum,
    projectile,
    restitution,
    spring,
)

logger = logging.getLogger(__name__)

# Scenario folder name -> scenario definition, in the order they are run.
SCENARIOS = {
    module.SCENARIO.name: module.SCENARIO
    for module in (
        coeff_friction,
        conservation_of_momentum,
        moment_of_inertia,
        pendulum,
        projectile,
        restitution,
        spring,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
        metavar="SCENARIO",
        help=f"Scenarios to run (default: all). Choices: {', '.join(SCENARIOS)}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)

    if not config.model_dir.is_dir():
        raise SystemExit(f"Directory not found: {config.model_dir}")
    logger.info("Model: %s", config.model_dir)
    logger.info("Reference: %s", config.reference_root)

    # Check every scenario up front so that SAM3 is only loaded if there is work to do.
    scenarios = []
    summary: dict[str, dict] = {}
    for scenario in args.scenarios:
        try:
            find_scenario(scenario, config)
            scenarios.append(scenario)
        except FileNotFoundError as e:
            logger.warning("[%s] skipped: %s", scenario, e)
            summary[scenario] = {"status": "skipped", "reason": str(e)}
    if not scenarios:
        raise SystemExit("No scenarios to evaluate.")

    predictor = build_predictor(args.device)
    for scenario in scenarios:
        try:
            failures = evaluate_scenario(predictor, SCENARIOS[scenario], config)
            summary[scenario] = {"status": "done", "failed_samples": failures}
        except Exception as e:
            logger.exception("[%s] aborted", scenario)
            summary[scenario] = {"status": "error", "reason": str(e)}

    summary_path = config.model_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Summary (%s):", summary_path)
    for scenario in args.scenarios:
        entry = summary[scenario]
        detail = entry.get("reason") or f"{len(entry['failed_samples'])} failed samples"
        logger.info("  %-26s %-8s %s", scenario, entry["status"], detail)

    direction_consistency.run_direction_consistency(config.model_dir, args.scenarios, config.output_name)
    invariance.print_scores(invariance.compute_invariance_scores([config.model_dir], config.output_name))


if __name__ == "__main__":
    main()
