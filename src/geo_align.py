from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.wgs84 import geodetic_to_enu, enu_to_geodetic
from src.colmap_model import read_images_txt, find_images_txt


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * R @ src + t."""
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src and dst must be N x 3 arrays")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 corresponding points for similarity alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s[-1, -1] = -1

    r = u @ s @ vt
    var_src = np.mean(np.sum(src_c**2, axis=1))
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * r @ mu_src
    return scale, r, t


def apply_similarity(point_xyz: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return scale * rot @ point_xyz + trans


def build_geo_alignment(
    sfm_dir: str | Path,
    reference_metadata_csv: str | Path,
    out_json: str | Path,
) -> dict:
    """Align COLMAP/SfM world coordinates to local ENU coordinates using reference telemetry."""
    images_txt = find_images_txt(sfm_dir)
    images = read_images_txt(images_txt)
    meta = pd.read_csv(reference_metadata_csv)

    # Use median reference origin.
    origin_lat = float(meta["lat"].median())
    origin_lon = float(meta["lon"].median())
    if "alt" in meta.columns and meta["alt"].notna().any():
        origin_alt = float(meta["alt"].dropna().median())
    else:
        origin_alt = 0.0

    src_xyz = []
    dst_enu = []
    used = []

    for _, row in meta.iterrows():
        name = row["image_name"]
        if name not in images:
            continue
        alt = row["alt"] if "alt" in row and pd.notna(row["alt"]) else origin_alt
        enu = geodetic_to_enu(float(row["lat"]), float(row["lon"]), float(alt), origin_lat, origin_lon, origin_alt)
        src_xyz.append(images[name]["camera_center"])
        dst_enu.append(enu)
        used.append(name)

    src_xyz = np.asarray(src_xyz, dtype=float)
    dst_enu = np.asarray(dst_enu, dtype=float)

    scale, rot, trans = umeyama_similarity(src_xyz, dst_enu)
    aligned = np.array([apply_similarity(p, scale, rot, trans) for p in src_xyz])
    errors = np.linalg.norm(aligned - dst_enu, axis=1)

    result = {
        "origin": {"lat": origin_lat, "lon": origin_lon, "alt": origin_alt},
        "scale": scale,
        "rotation": rot.tolist(),
        "translation": trans.tolist(),
        "num_used_frames": len(used),
        "median_alignment_error_m": float(np.median(errors)),
        "mean_alignment_error_m": float(np.mean(errors)),
        "max_alignment_error_m": float(np.max(errors)),
        "used_images": used,
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_geo_alignment(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sfm_xyz_to_geodetic(point_xyz: np.ndarray, alignment: dict) -> tuple[float, float, float]:
    scale = float(alignment["scale"])
    rot = np.asarray(alignment["rotation"], dtype=float)
    trans = np.asarray(alignment["translation"], dtype=float)
    origin = alignment["origin"]

    enu = apply_similarity(point_xyz, scale, rot, trans)
    return enu_to_geodetic(enu, origin["lat"], origin["lon"], origin["alt"])
