from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

#Running comand - python main.py --config config.yaml --only-fusion

ROOT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT_DIR / "pipeline"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()

    # Make imports like "from src.wgs84 import ..." work
    # even when the executed script is inside the pipeline folder.
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT_DIR) if not old_pythonpath else str(ROOT_DIR) + os.pathsep + old_pythonpath

    print("\n>>> " + " ".join(str(x) for x in cmd))
    subprocess.check_call([str(x) for x in cmd], cwd=ROOT_DIR, env=env)


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}\nMissing file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current Option 2 dynamic view-center navigation pipeline."
    )

    parser.add_argument("--config", default="config.yaml")

    parser.add_argument(
        "--only-fusion",
        action="store_true",
        help="Run only the final fusion optimizer using existing intermediate outputs.",
    )

    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip fast query/reference reranking and use existing outputs/fast_reranked_matches.csv.",
    )

    parser.add_argument(
        "--skip-anchor-vo",
        action="store_true",
        help="Skip anchor_vo_v2 and use existing outputs/anchor_vo_v2_route.csv and outputs/anchor_vo_v2_vo_debug.csv.",
    )

    parser.add_argument(
        "--skip-satellite-prior",
        action="store_true",
        help="Skip satellite/map prior and use existing outputs/option2_satellite_prior.csv.",
    )

    args = parser.parse_args()
    py = sys.executable

    require_file(ROOT_DIR / args.config, "Config file was not found.")

    fusion_script = PIPELINE_DIR / "option2_fusion_optimizer.py"
    satellite_prior_script = PIPELINE_DIR / "option2_satellite_prior.py"
    anchor_vo_script = PIPELINE_DIR / "anchor_vo_v2.py"
    rerank_script = PIPELINE_DIR / "fast_rerank_query_reference_matches.py"

    require_file(fusion_script, "The fusion optimizer script was not found in the pipeline folder.")

    if args.only_fusion:
        run([py, fusion_script, "--config", args.config])
        return

    if not args.skip_rerank:
        require_file(rerank_script, "The fast rerank script was not found in the pipeline folder.")
        run(
            [
                py,
                rerank_script,
                "--max-candidates-per-query",
                "50",
                "--keep-top-k",
                "5",
                "--detector",
                "orb",
                "--min-inliers",
                "10",
                "--max-contact-sheets",
                "120",
                "--resize-width",
                "720",
                "--rotations",
                "0",
                "--max-query-frames",
                "0",
            ]
        )

    if not args.skip_anchor_vo:
        require_file(anchor_vo_script, "The anchor VO script was not found in the pipeline folder.")
        run([py, anchor_vo_script, "--config", args.config])

    if not args.skip_satellite_prior:
        require_file(satellite_prior_script, "The satellite/map prior script was not found in the pipeline folder.")
        run([py, satellite_prior_script, "--config", args.config])

    run([py, fusion_script, "--config", args.config])


if __name__ == "__main__":
    main()
