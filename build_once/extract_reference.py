from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.frame_io import extract_reference_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path(cfg["paths"]["outputs_dir"])
    frames_dir = out / "reference_frames"
    meta_csv = out / "reference_metadata.csv"

    df = extract_reference_frames(
        cfg["paths"]["reference_videos_dir"],
        frames_dir,
        frame_step_sec=float(cfg["reference"]["frame_step_sec"]),
        telemetry_time_offset_sec=float(cfg["reference"].get("telemetry_time_offset_sec", 0.0)),
        blur_threshold=float(cfg["reference"].get("blur_threshold", 0.0)),
    )
    df.to_csv(meta_csv, index=False)
    print(f"Saved {len(df)} reference frames")
    print(meta_csv)


if __name__ == "__main__":
    main()
