from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.geo_align import build_geo_alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path(cfg["paths"]["outputs_dir"])
    result = build_geo_alignment(
        sfm_dir=out / "sfm_txt",
        reference_metadata_csv=out / "reference_metadata.csv",
        out_json=out / "geo_alignment.json",
    )

    print("Geo alignment finished")
    print(f"Frames used: {result['num_used_frames']}")
    print(f"Median error: {result['median_alignment_error_m']:.2f} m")
    print(f"Mean error: {result['mean_alignment_error_m']:.2f} m")


if __name__ == "__main__":
    main()
