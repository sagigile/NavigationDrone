from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


@dataclass
class TelemetryPoint:
    time_sec: float
    lat: float
    lon: float
    alt: float | None = None
    rel_alt: float | None = None


_SRT_TIME_RE = re.compile(
    r"(?P<h1>\d\d):(?P<m1>\d\d):(?P<s1>\d\d),(?P<ms1>\d\d\d)\s+-->\s+"
    r"(?P<h2>\d\d):(?P<m2>\d\d):(?P<s2>\d\d),(?P<ms2>\d\d\d)"
)
_LAT_RE = re.compile(r"\[latitude:\s*([-+]?\d+(?:\.\d+)?)\]")
_LON_RE = re.compile(r"\[longitude:\s*([-+]?\d+(?:\.\d+)?)\]")
_REL_ALT_RE = re.compile(r"rel_alt:\s*([-+]?\d+(?:\.\d+)?)")
_ABS_ALT_RE = re.compile(r"abs_alt:\s*([-+]?\d+(?:\.\d+)?)")


def _to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def read_srt_telemetry(path: str | Path, time_offset_sec: float = 0.0) -> pd.DataFrame:
    """Read DJI-like SRT telemetry and return columns: time_sec, lat, lon, alt, rel_alt."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    rows: list[dict] = []

    for block in blocks:
        time_m = _SRT_TIME_RE.search(block)
        lat_m = _LAT_RE.search(block)
        lon_m = _LON_RE.search(block)
        if not (time_m and lat_m and lon_m):
            continue

        start_sec = _to_sec(time_m["h1"], time_m["m1"], time_m["s1"], time_m["ms1"])
        rel_alt_m = _REL_ALT_RE.search(block)
        abs_alt_m = _ABS_ALT_RE.search(block)

        rows.append(
            {
                "time_sec": start_sec + time_offset_sec,
                "lat": float(lat_m.group(1)),
                "lon": float(lon_m.group(1)),
                "alt": float(abs_alt_m.group(1)) if abs_alt_m else None,
                "rel_alt": float(rel_alt_m.group(1)) if rel_alt_m else None,
            }
        )

    if not rows:
        raise ValueError(f"No telemetry rows were parsed from {path}")

    return pd.DataFrame(rows).sort_values("time_sec").reset_index(drop=True)


def read_csv_telemetry(path: str | Path, time_offset_sec: float = 0.0) -> pd.DataFrame:
    """Read CSV telemetry. Required columns: time_sec, lat, lon. Optional: alt, rel_alt."""
    df = pd.read_csv(path)
    required = {"time_sec", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing telemetry CSV columns: {missing}")

    df = df.copy()
    df["time_sec"] = df["time_sec"].astype(float) + time_offset_sec
    if "alt" not in df.columns:
        df["alt"] = None
    if "rel_alt" not in df.columns:
        df["rel_alt"] = None
    return df[["time_sec", "lat", "lon", "alt", "rel_alt"]].sort_values("time_sec").reset_index(drop=True)


def read_telemetry_for_video(video_path: Path, time_offset_sec: float = 0.0) -> pd.DataFrame:
    """Find telemetry file with the same stem as the video: .srt or .csv."""
    srt = video_path.with_suffix(".srt")
    csv = video_path.with_suffix(".csv")
    if srt.exists():
        return read_srt_telemetry(srt, time_offset_sec)
    if csv.exists():
        return read_csv_telemetry(csv, time_offset_sec)
    raise FileNotFoundError(f"No telemetry file found for {video_path}. Expected {srt.name} or {csv.name}")


def nearest_telemetry(df: pd.DataFrame, time_sec: float) -> dict:
    """Return telemetry row nearest to a requested video timestamp."""
    idx = (df["time_sec"] - time_sec).abs().idxmin()
    return df.loc[idx].to_dict()
