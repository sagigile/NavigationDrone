from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Force Python to use the local hloc clone inside this project.
HLOC_ROOT = Path(r"C:\Users\taliy\PythonProject\TestNav\Hierarchical-Localization")
if HLOC_ROOT.exists():
    sys.path.insert(0, str(HLOC_ROOT))

from config import load_config
from frame_io import extract_query_frames
from colmap_model import camera_center_from_qt, qvec_to_rotmat
from geo_align import load_geo_alignment, sfm_xyz_to_geodetic, apply_similarity
from wgs84 import geodetic_to_enu, enu_to_geodetic
from kalman import ConstantVelocityKalman3D
from kml_export import write_route_kml


# -----------------------------------------------------------------------------
# Camera intrinsics
# -----------------------------------------------------------------------------

def load_median_camera_from_cameras_txt(cameras_txt: Path) -> dict:
    """Read COLMAP cameras.txt and return a robust median camera for query frames."""
    if not cameras_txt.exists():
        return {
            "model": "SIMPLE_PINHOLE",
            "width": 1920,
            "height": 1080,
            "f": None,
            "cx": None,
            "cy": None,
            "k": 0.0,
        }

    rows = []
    for line in cameras_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        model = parts[1]
        width = int(float(parts[2]))
        height = int(float(parts[3]))
        params = [float(x) for x in parts[4:]]

        if model == "SIMPLE_PINHOLE" and len(params) >= 3:
            rows.append((model, width, height, params[0], params[1], params[2], 0.0))
        elif model == "SIMPLE_RADIAL" and len(params) >= 4:
            rows.append((model, width, height, params[0], params[1], params[2], params[3]))
        elif model == "PINHOLE" and len(params) >= 4:
            f = 0.5 * (params[0] + params[1])
            rows.append(("SIMPLE_PINHOLE", width, height, f, params[2], params[3], 0.0))

    if not rows:
        raise RuntimeError(f"Could not parse camera intrinsics from {cameras_txt}")

    widths = [r[1] for r in rows]
    heights = [r[2] for r in rows]
    fs = [r[3] for r in rows]
    cxs = [r[4] for r in rows]
    cys = [r[5] for r in rows]
    ks = [r[6] for r in rows]

    # SIMPLE_RADIAL is preferred because your COLMAP model was reconstructed with it.
    return {
        "model": "SIMPLE_RADIAL",
        "width": int(round(float(np.median(widths)))),
        "height": int(round(float(np.median(heights)))),
        "f": float(np.median(fs)),
        "cx": float(np.median(cxs)),
        "cy": float(np.median(cys)),
        "k": float(np.median(ks)),
    }


def get_query_camera(cfg: dict, out: Path, query_frames_dir: Path) -> dict:
    """Get query camera intrinsics from config or from COLMAP cameras.txt."""
    camera_cfg = cfg.get("camera", {})

    cameras_txt = Path(camera_cfg.get("cameras_txt", out / "sfm_txt" / "cameras.txt"))
    cam = load_median_camera_from_cameras_txt(cameras_txt)

    image_paths = sorted(
        list(query_frames_dir.glob("*.jpg"))
        + list(query_frames_dir.glob("*.jpeg"))
        + list(query_frames_dir.glob("*.png"))
    )
    if image_paths:
        first = cv2.imread(str(image_paths[0]))
        if first is not None:
            h, w = first.shape[:2]
            cam["width"] = int(w)
            cam["height"] = int(h)
            if cam["cx"] is None:
                cam["cx"] = w / 2.0
            if cam["cy"] is None:
                cam["cy"] = h / 2.0
            if cam["f"] is None:
                cam["f"] = 1.2 * max(w, h)

    if "f_px" in camera_cfg:
        cam["f"] = float(camera_cfg["f_px"])
    if "cx" in camera_cfg:
        cam["cx"] = float(camera_cfg["cx"])
    if "cy" in camera_cfg:
        cam["cy"] = float(camera_cfg["cy"])
    if "radial_k" in camera_cfg:
        cam["k"] = float(camera_cfg["radial_k"])
    if "model" in camera_cfg:
        cam["model"] = str(camera_cfg["model"])

    return cam


def write_query_list(query_frames_dir: Path, out_txt: Path, camera: dict) -> Path:
    """Create the query image list expected by hloc.localize_sfm."""
    image_paths = sorted(
        list(query_frames_dir.glob("*.jpg"))
        + list(query_frames_dir.glob("*.jpeg"))
        + list(query_frames_dir.glob("*.png"))
    )

    if not image_paths:
        raise RuntimeError(f"No query frames found in {query_frames_dir}")

    w = int(camera["width"])
    h = int(camera["height"])
    f = float(camera["f"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    k = float(camera.get("k", 0.0))
    model = str(camera.get("model", "SIMPLE_RADIAL"))

    with out_txt.open("w", encoding="utf-8") as file:
        for image_path in image_paths:
            if model == "SIMPLE_RADIAL":
                file.write(f"{image_path.name} SIMPLE_RADIAL {w} {h} {f:.6f} {cx:.6f} {cy:.6f} {k:.8f}\n")
            else:
                file.write(f"{image_path.name} SIMPLE_PINHOLE {w} {h} {f:.6f} {cx:.6f} {cy:.6f}\n")

    return out_txt


# -----------------------------------------------------------------------------
# View center computation
# -----------------------------------------------------------------------------

def sfm_point_to_enu(point_xyz: np.ndarray, alignment: dict) -> np.ndarray:
    scale = float(alignment["scale"])
    rot = np.asarray(alignment["rotation"], dtype=float)
    trans = np.asarray(alignment["translation"], dtype=float)
    return apply_similarity(point_xyz, scale, rot, trans)


def sfm_direction_to_enu(direction_xyz: np.ndarray, alignment: dict) -> np.ndarray:
    rot = np.asarray(alignment["rotation"], dtype=float)
    d = rot @ direction_xyz
    n = np.linalg.norm(d)
    if n == 0:
        return d
    return d / n


def get_ground_alt_m(cfg: dict) -> float:
    vc = cfg.get("view_center", {})
    if "ground_alt_m" in vc:
        return float(vc["ground_alt_m"])

    # From the SRT example you sent: abs_alt 759.842 - rel_alt 49.800 = 710.042m.
    # This is only a default. If your flight area has a better known ground altitude, set it in config.yaml.
    return 710.042


def compute_view_center_geodetic(qvec: np.ndarray, tvec: np.ndarray, alignment: dict, cfg: dict) -> tuple[float, float, float, dict]:
    """
    Convert hloc camera pose into the ground point seen at the image center.

    COLMAP stores world-to-camera as x_cam = R*x_world + t.
    The camera center in world coordinates is C = -R.T @ t.
    The optical axis in COLMAP camera coordinates is usually +Z.
    """
    origin = alignment["origin"]
    ground_alt = get_ground_alt_m(cfg)

    center_sfm = camera_center_from_qt(qvec, tvec)
    center_enu = sfm_point_to_enu(center_sfm, alignment)

    r = qvec_to_rotmat(qvec)
    axis_mode = cfg.get("view_center", {}).get("optical_axis", "auto")
    try_opposite = bool(cfg.get("view_center", {}).get("try_opposite_axis", True))

    axis_candidates = []
    if axis_mode == "negative_z":
        axis_candidates = [np.array([0.0, 0.0, -1.0])]
    elif axis_mode == "positive_z":
        axis_candidates = [np.array([0.0, 0.0, 1.0])]
    else:
        axis_candidates = [np.array([0.0, 0.0, 1.0])]
        if try_opposite:
            axis_candidates.append(np.array([0.0, 0.0, -1.0]))

    ground_enu = geodetic_to_enu(
        origin["lat"],
        origin["lon"],
        ground_alt,
        origin["lat"],
        origin["lon"],
        origin["alt"],
    )
    ground_z = float(ground_enu[2])

    chosen = None
    for axis_cam in axis_candidates:
        direction_sfm = r.T @ axis_cam
        direction_enu = sfm_direction_to_enu(direction_sfm, alignment)

        if abs(direction_enu[2]) < 1e-6:
            continue

        lam = (ground_z - center_enu[2]) / direction_enu[2]
        if lam <= 0:
            continue

        p_enu = center_enu + lam * direction_enu
        chosen = (p_enu, direction_enu, lam, axis_cam)
        break

    if chosen is None:
        raise RuntimeError("Could not intersect optical ray with ground plane")

    p_enu, direction_enu, lam, axis_cam = chosen
    lat, lon, alt = enu_to_geodetic(p_enu, origin["lat"], origin["lon"], origin["alt"])

    debug = {
        "camera_enu_e": float(center_enu[0]),
        "camera_enu_n": float(center_enu[1]),
        "camera_enu_u": float(center_enu[2]),
        "view_ray_enu_e": float(direction_enu[0]),
        "view_ray_enu_n": float(direction_enu[1]),
        "view_ray_enu_u": float(direction_enu[2]),
        "ray_ground_lambda_m": float(lam),
        "ground_alt_m": float(ground_alt),
        "axis_cam_x": float(axis_cam[0]),
        "axis_cam_y": float(axis_cam[1]),
        "axis_cam_z": float(axis_cam[2]),
    }
    return lat, lon, alt, debug


# -----------------------------------------------------------------------------
# Geographic and trajectory filters
# -----------------------------------------------------------------------------

def _get_bbox(cfg: dict) -> dict:
    geo = cfg.get("geo_filter", {})
    return {
        "min_lat": float(geo.get("min_lat", 32.0800)),
        "max_lat": float(geo.get("max_lat", 32.1400)),
        "min_lon": float(geo.get("min_lon", 35.1400)),
        "max_lon": float(geo.get("max_lon", 35.2500)),
    }


def is_inside_allowed_area(lat: float, lon: float, cfg: dict) -> bool:
    if not cfg.get("geo_filter", {}).get("enabled", True):
        return True

    if not np.isfinite(lat) or not np.isfinite(lon):
        return False

    bbox = _get_bbox(cfg)
    return (
        bbox["min_lat"] <= lat <= bbox["max_lat"]
        and bbox["min_lon"] <= lon <= bbox["max_lon"]
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def select_best_temporal_chain(df: pd.DataFrame, cfg: dict) -> set[int]:
    filt_cfg = cfg.get("trajectory_filter", {})
    max_speed_mps = float(filt_cfg.get("max_speed_mps", 35.0))
    max_gap_sec = float(filt_cfg.get("max_gap_sec", 8.0))
    max_jump_m = float(filt_cfg.get("max_jump_m", 120.0))

    candidates = []
    for idx, row in df.iterrows():
        if not bool(row.get("candidate_valid", False)):
            continue
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        candidates.append(idx)

    if not candidates:
        return set()

    dp: dict[int, int] = {}
    prev: dict[int, int | None] = {}

    best_idx = candidates[0]
    best_score = 1

    for i in candidates:
        dp[i] = 1
        prev[i] = None

        ti = float(df.loc[i, "time_sec"])
        lati = float(df.loc[i, "lat"])
        loni = float(df.loc[i, "lon"])

        for j in candidates:
            if j >= i:
                break

            tj = float(df.loc[j, "time_sec"])
            dt = ti - tj
            if dt <= 0 or dt > max_gap_sec:
                continue

            latj = float(df.loc[j, "lat"])
            lonj = float(df.loc[j, "lon"])
            dist = haversine_m(latj, lonj, lati, loni)

            allowed_dist = max(max_jump_m, max_speed_mps * dt)
            if dist <= allowed_dist:
                score = dp[j] + 1
                if score > dp[i]:
                    dp[i] = score
                    prev[i] = j

        if dp[i] > best_score:
            best_score = dp[i]
            best_idx = i

    chain = set()
    cur: int | None = best_idx
    while cur is not None:
        chain.add(cur)
        cur = prev[cur]

    min_chain_points = int(filt_cfg.get("min_chain_points", 3))
    if len(chain) < min_chain_points:
        return set()

    return chain


def densify_track(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    dense_cfg = cfg.get("route_display", {})
    if not dense_cfg.get("densify", True):
        return df.copy()

    step_sec = float(dense_cfg.get("dense_step_sec", 0.5))
    max_gap_sec = float(dense_cfg.get("max_interpolation_gap_sec", 6.0))

    valid = df[(df["localized"] == True) & df["lat"].notna() & df["lon"].notna()].copy()
    valid = valid.sort_values("time_sec")
    if len(valid) < 2:
        return valid

    dense_rows = []
    for k in range(len(valid) - 1):
        a = valid.iloc[k].to_dict()
        b = valid.iloc[k + 1].to_dict()

        dense_rows.append({**a, "interpolated": False})

        t0 = float(a["time_sec"])
        t1 = float(b["time_sec"])
        dt = t1 - t0

        if dt <= 0 or dt > max_gap_sec:
            continue

        n = int(dt // step_sec)
        for m in range(1, n):
            t = t0 + m * step_sec
            alpha = (t - t0) / dt

            row = dict(a)
            row["time_sec"] = t
            row["lat"] = (1.0 - alpha) * float(a["lat"]) + alpha * float(b["lat"])
            row["lon"] = (1.0 - alpha) * float(a["lon"]) + alpha * float(b["lon"])

            if not pd.isna(a.get("alt")) and not pd.isna(b.get("alt")):
                row["alt"] = (1.0 - alpha) * float(a["alt"]) + alpha * float(b["alt"])

            row["localized"] = True
            row["interpolated"] = True
            row["reject_reason"] = ""
            dense_rows.append(row)

    dense_rows.append({**valid.iloc[-1].to_dict(), "interpolated": False})
    return pd.DataFrame(dense_rows)


# -----------------------------------------------------------------------------
# hloc query localization
# -----------------------------------------------------------------------------

def _run_hloc_query_localization(cfg: dict, out: Path, query_frames_dir: Path) -> Path:
    try:
        from hloc import extract_features, match_features, pairs_from_retrieval, localize_sfm
    except ImportError as e:
        raise ImportError("hloc is required for query localization") from e

    feature_conf = extract_features.confs[cfg["hloc"]["local_feature"]]
    retrieval_conf = extract_features.confs[cfg["hloc"]["retrieval_feature"]]

    if cfg["hloc"]["matcher_preferred"] in match_features.confs:
        matcher_conf = match_features.confs[cfg["hloc"]["matcher_preferred"]]
    elif cfg["hloc"]["matcher_preferred"].replace("+", "-") in match_features.confs:
        matcher_conf = match_features.confs[cfg["hloc"]["matcher_preferred"].replace("+", "-")]
    else:
        matcher_conf = match_features.confs[cfg["hloc"]["matcher_fallback"]]

    sfm_dir = out / "sfm"
    features_dir = out / "features"

    feature_output = feature_conf["output"]
    retrieval_output = retrieval_conf["output"]
    matcher_output = matcher_conf["output"]

    ref_retrieval = features_dir / f"{retrieval_output}.h5"
    ref_features = features_dir / f"{feature_output}.h5"

    query_features = features_dir / f"query-{feature_output}.h5"
    query_retrieval = features_dir / f"query-{retrieval_output}.h5"
    loc_pairs = out / "pairs-query-loc.txt"
    loc_matches = features_dir / f"query-{matcher_output}.h5"
    results = out / "query_hloc_results.txt"

    extract_features.main(feature_conf, query_frames_dir, feature_path=query_features)
    extract_features.main(retrieval_conf, query_frames_dir, feature_path=query_retrieval)

    try:
        pairs_from_retrieval.main(
            query_retrieval,
            loc_pairs,
            num_matched=int(cfg["hloc"].get("retrieval_top_k", 20)),
            db_descriptors=ref_retrieval,
            query_prefix="",
            db_prefix="",
        )
    except TypeError:
        pairs_from_retrieval.main(
            query_retrieval,
            loc_pairs,
            num_matched=int(cfg["hloc"].get("retrieval_top_k", 20)),
            db_descriptors=ref_retrieval,
        )

    try:
        match_features.main(
            matcher_conf,
            loc_pairs,
            features=query_features,
            matches=loc_matches,
            features_ref=ref_features,
        )
    except TypeError:
        raise RuntimeError(
            "Your hloc version does not support separate query/reference feature files in match_features.main. "
            "Update hloc, or merge query and reference feature h5 files before matching."
        )

    camera = get_query_camera(cfg, out, query_frames_dir)
    query_list = write_query_list(query_frames_dir, out / "query_list.txt", camera)
    print(f"Using query camera: {camera}")

    try:
        localize_sfm.main(
            sfm_dir,
            query_list,
            loc_pairs,
            query_features,
            loc_matches,
            results,
            covisibility_clustering=False,
        )
    except TypeError:
        localize_sfm.main(
            sfm_dir,
            query_list,
            loc_pairs,
            query_features,
            loc_matches,
            results,
        )

    return results


def _parse_hloc_results(results_txt: Path) -> dict[str, dict]:
    poses = {}
    if not results_txt.exists():
        raise FileNotFoundError(results_txt)

    for line in results_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        name = parts[0]
        try:
            qvec = np.array([float(x) for x in parts[1:5]], dtype=float)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=float)
        except ValueError:
            continue

        poses[name] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_center": camera_center_from_qt(qvec, tvec),
        }
    return poses


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out = Path(cfg["paths"]["outputs_dir"])
    query_frames_dir = out / "query_frames"
    query_meta_csv = out / "query_metadata.csv"

    query_df = extract_query_frames(
        cfg["paths"]["query_video_path"],
        query_frames_dir,
        frame_step_sec=float(cfg["query"]["frame_step_sec"]),
    )
    query_df.to_csv(query_meta_csv, index=False)

    results_txt = _run_hloc_query_localization(cfg, out, query_frames_dir)
    poses = _parse_hloc_results(results_txt)
    alignment = load_geo_alignment(out / "geo_alignment.json")

    use_view_center = bool(cfg.get("view_center", {}).get("enabled", True))

    rows = []
    for _, row in query_df.iterrows():
        name = row["image_name"]

        base = row.to_dict()
        base["localized"] = False
        base["candidate_valid"] = False
        base["lat"] = None
        base["lon"] = None
        base["alt"] = None
        base["camera_lat"] = None
        base["camera_lon"] = None
        base["camera_alt"] = None
        base["reject_reason"] = ""

        if name not in poses:
            base["reject_reason"] = "no_hloc_pose"
            rows.append(base)
            continue

        pose = poses[name]
        camera_lat, camera_lon, camera_alt = sfm_xyz_to_geodetic(pose["camera_center"], alignment)
        base["camera_lat"] = camera_lat
        base["camera_lon"] = camera_lon
        base["camera_alt"] = camera_alt

        if use_view_center:
            try:
                lat, lon, alt, debug = compute_view_center_geodetic(pose["qvec"], pose["tvec"], alignment, cfg)
                base.update(debug)
                base["position_type"] = "view_center_ground_intersection"
            except Exception as exc:
                base["reject_reason"] = f"view_center_failed:{type(exc).__name__}"
                rows.append(base)
                continue
        else:
            lat, lon, alt = camera_lat, camera_lon, camera_alt
            base["position_type"] = "camera_center"

        base["lat"] = lat
        base["lon"] = lon
        base["alt"] = alt

        if not is_inside_allowed_area(lat, lon, cfg):
            base["reject_reason"] = "outside_allowed_area"
            rows.append(base)
            continue

        base["candidate_valid"] = True
        base["reject_reason"] = "candidate_before_temporal_filter"
        rows.append(base)

    raw_candidates = pd.DataFrame(rows)
    chain = select_best_temporal_chain(raw_candidates, cfg)

    filtered_rows = []
    for idx, r in raw_candidates.iterrows():
        rd = r.to_dict()
        if idx in chain:
            rd["localized"] = True
            rd["reject_reason"] = ""
        else:
            rd["localized"] = False
            if rd.get("candidate_valid", False) and rd.get("reject_reason") == "candidate_before_temporal_filter":
                rd["reject_reason"] = "not_in_best_temporal_chain"
        filtered_rows.append(rd)

    raw_csv = out / "query_localization_raw.csv"
    pd.DataFrame(filtered_rows).to_csv(raw_csv, index=False)

    if cfg.get("kalman", {}).get("enabled", True):
        raw = pd.DataFrame(filtered_rows)
        origin = alignment["origin"]
        kf = ConstantVelocityKalman3D(
            process_noise=float(cfg["kalman"].get("process_noise", 2.0)),
            measurement_noise=float(cfg["kalman"].get("measurement_noise", 8.0)),
        )

        smooth_rows = []
        for _, r in raw.iterrows():
            if not bool(r.get("localized", False)) or pd.isna(r["lat"]) or pd.isna(r["lon"]):
                smooth_rows.append(r.to_dict())
                continue

            enu = geodetic_to_enu(
                float(r["lat"]),
                float(r["lon"]),
                float(r.get("alt", origin["alt"])),
                origin["lat"],
                origin["lon"],
                origin["alt"],
            )
            smoothed_enu = kf.update(float(r["time_sec"]), enu)
            slat, slon, salt = enu_to_geodetic(smoothed_enu, origin["lat"], origin["lon"], origin["alt"])

            rd = r.to_dict()
            rd["raw_lat"] = rd["lat"]
            rd["raw_lon"] = rd["lon"]
            rd["raw_alt"] = rd["alt"]
            rd["lat"] = slat
            rd["lon"] = slon
            rd["alt"] = salt
            smooth_rows.append(rd)

        smoothed_csv = out / "query_localization_smoothed.csv"
        smoothed_df = pd.DataFrame(smooth_rows)
        smoothed_df.to_csv(smoothed_csv, index=False)
    else:
        smoothed_csv = raw_csv
        smoothed_df = pd.DataFrame(filtered_rows)

    dense_df = densify_track(smoothed_df, cfg)
    dense_csv = out / "query_route_dense.csv"
    dense_df.to_csv(dense_csv, index=False)

    kml = out / "query_route.kml"
    write_route_kml(dense_csv, kml)

    num_candidates = int(raw_candidates["candidate_valid"].sum()) if "candidate_valid" in raw_candidates else 0
    num_final = int(pd.DataFrame(filtered_rows)["localized"].sum())
    print(f"Raw candidate localizations inside area: {num_candidates}")
    print(f"Final temporally consistent localizations: {num_final}")
    print(f"Raw localization: {raw_csv}")
    print(f"Smoothed localization: {smoothed_csv}")
    print(f"Dense route CSV: {dense_csv}")
    print(f"KML route: {kml}")


if __name__ == "__main__":
    main()
