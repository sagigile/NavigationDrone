from __future__ import annotations

from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

from src.telemetry import read_telemetry_for_video, nearest_telemetry


def laplacian_blur_score(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames_from_video(
    video_path: str | Path,
    out_dir: str | Path,
    frame_step_sec: float,
    blur_threshold: float = 0.0,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Extract frames from a video without telemetry."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_frames = max(1, int(round(frame_step_sec * fps)))

    rows = []
    frame_idx = 0
    saved_idx = 0

    pbar = tqdm(total=total if total > 0 else None, desc=f"Extracting {video_path.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step_frames == 0:
            t = frame_idx / fps
            blur = laplacian_blur_score(frame)
            if blur >= blur_threshold:
                name = f"{prefix}_{saved_idx:06d}.jpg"
                out_path = out_dir / name
                cv2.imwrite(str(out_path), frame)
                rows.append(
                    {
                        "image_name": name,
                        "source_video": video_path.name,
                        "time_sec": t,
                        "blur_score": blur,
                    }
                )
                saved_idx += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return pd.DataFrame(rows)


def extract_reference_frames(
    reference_videos_dir: str | Path,
    out_dir: str | Path,
    frame_step_sec: float,
    telemetry_time_offset_sec: float,
    blur_threshold: float,
) -> pd.DataFrame:
    """Extract geotagged frames from all videos in a directory."""
    reference_videos_dir = Path(reference_videos_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        [p for p in reference_videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    )
    if not videos:
        raise FileNotFoundError(f"No reference videos found in {reference_videos_dir}")

    all_rows = []
    for video in videos:
        telemetry = read_telemetry_for_video(video, telemetry_time_offset_sec)
        frames = extract_frames_from_video(
            video,
            out_dir,
            frame_step_sec=frame_step_sec,
            blur_threshold=blur_threshold,
            prefix=video.stem,
        )

        for _, row in frames.iterrows():
            tel = nearest_telemetry(telemetry, float(row["time_sec"]))
            all_rows.append(
                {
                    **row.to_dict(),
                    "lat": tel["lat"],
                    "lon": tel["lon"],
                    "alt": tel.get("alt"),
                    "rel_alt": tel.get("rel_alt"),
                    "telemetry_time_sec": tel["time_sec"],
                    "telemetry_delta_sec": float(row["time_sec"] - tel["time_sec"]),
                }
            )

    return pd.DataFrame(all_rows)


def extract_query_frames(video_path: str | Path, out_dir: str | Path, frame_step_sec: float) -> pd.DataFrame:
    return extract_frames_from_video(video_path, out_dir, frame_step_sec=frame_step_sec, blur_threshold=0.0)
