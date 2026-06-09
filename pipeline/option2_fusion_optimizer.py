from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from src.wgs84 import geodetic_to_enu, enu_to_geodetic


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_origin(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["origin"]


def latlon_to_enu2(lat: float, lon: float, alt: float, origin: dict) -> np.ndarray:
    enu = geodetic_to_enu(lat, lon, alt, origin["lat"], origin["lon"], origin["alt"])
    return np.array([enu[0], enu[1]], dtype=float)


def enu2_to_latlon(xy: np.ndarray, origin: dict, alt: float | None = None) -> tuple[float, float, float]:
    if alt is None:
        alt = float(origin["alt"])
    return enu_to_geodetic(
        np.array([xy[0], xy[1], alt], dtype=float),
        origin["lat"],
        origin["lon"],
        origin["alt"],
    )


def prepare_query(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_name"] = df["image_name"].astype(str).map(lambda x: Path(x).name)
    df = df.sort_values("time_sec").reset_index(drop=True)
    df["query_index"] = np.arange(len(df), dtype=int)
    return df


def load_initial_route(route_csv: Path, query_df: pd.DataFrame, origin: dict) -> np.ndarray:
    route = pd.read_csv(route_csv)
    if route.empty:
        raise RuntimeError(f"Initial route is empty: {route_csv}")

    route["image_name"] = route["image_name"].astype(str).map(lambda x: Path(x).name)
    q_lookup = query_df.set_index("image_name", drop=False)

    n = len(query_df)
    out = np.zeros((n, 2), dtype=float)
    known = []

    lat_col = "lat" if "lat" in route.columns else "map_lat"
    lon_col = "lon" if "lon" in route.columns else "map_lon"

    for _, row in route.iterrows():
        name = row["image_name"]
        if name not in q_lookup.index:
            continue
        if pd.isna(row.get(lat_col)) or pd.isna(row.get(lon_col)):
            continue

        idx = int(q_lookup.loc[name]["query_index"])
        alt = float(row["alt"]) if "alt" in row and pd.notna(row["alt"]) else float(origin["alt"])
        xy = latlon_to_enu2(float(row[lat_col]), float(row[lon_col]), alt, origin)
        known.append((idx, xy))

    if not known:
        raise RuntimeError("No initial route points match query_metadata.csv")

    known = sorted(known, key=lambda z: z[0])

    first_i, first_xy = known[0]
    out[: first_i + 1] = first_xy

    for (i0, xy0), (i1, xy1) in zip(known[:-1], known[1:]):
        if i1 <= i0:
            continue
        for i in range(i0, i1 + 1):
            a = (i - i0) / (i1 - i0)
            out[i] = (1 - a) * xy0 + a * xy1

    last_i, last_xy = known[-1]
    out[last_i:] = last_xy

    return out


def load_reference_anchors(matches_csv: Path, query_df: pd.DataFrame, origin: dict, cfg: dict) -> list[dict]:
    opt = cfg["option2"]["fusion"]
    strong_min = int(opt.get("reference_strong_min_inliers", 25))
    weak_min = int(opt.get("reference_weak_min_inliers", 8))
    strong_w = float(opt.get("reference_strong_weight", 3.0))
    weak_w = float(opt.get("reference_weak_weight", 0.8))

    if not matches_csv.exists():
        print(f"Reference matches CSV not found: {matches_csv}. Continuing without reference anchors.")
        return []

    df = pd.read_csv(matches_csv)
    if df.empty:
        return []

    df["query_image"] = df["query_image"].astype(str).map(lambda x: Path(x).name)
    q_lookup = query_df.set_index("image_name", drop=False)

    if "reranked_rank" in df.columns:
        df = df[df["reranked_rank"] == 1].copy()

    out = []
    for _, row in df.iterrows():
        name = row["query_image"]
        if name not in q_lookup.index:
            continue
        if pd.isna(row.get("reference_lat")) or pd.isna(row.get("reference_lon")):
            continue

        inliers = int(row.get("inliers", 0))
        if inliers >= strong_min:
            w = strong_w
            level = "ref_strong"
        elif inliers >= weak_min:
            w = weak_w
            level = "ref_weak"
        else:
            continue

        idx = int(q_lookup.loc[name]["query_index"])
        alt = float(row["reference_alt"]) if "reference_alt" in row and pd.notna(row["reference_alt"]) else float(origin["alt"])
        xy = latlon_to_enu2(float(row["reference_lat"]), float(row["reference_lon"]), alt, origin)

        quality = min(2.0, max(0.5, inliers / max(1, strong_min)))
        out.append(
            {
                "idx": idx,
                "xy": xy,
                "weight": w * quality,
                "source": level,
                "inliers": inliers,
                "confidence": min(1.0, inliers / max(1.0, 2.0 * strong_min)),
                "image_name": name,
                "ref": row.get("reference_image", ""),
            }
        )

    return out


def load_map_priors(map_csv: Path, query_df: pd.DataFrame, origin: dict, cfg: dict) -> list[dict]:
    opt = cfg["option2"]["fusion"]

    min_inliers = int(opt.get("map_prior_min_inliers", 12))
    base_w = float(opt.get("map_prior_weight", 4.0))

    if not map_csv.exists():
        print(f"Map prior CSV not found: {map_csv}. Continuing without map priors.")
        return []

    df = pd.read_csv(map_csv)
    if df.empty or "localized" not in df.columns:
        return []

    df["image_name"] = df["image_name"].astype(str).map(lambda x: Path(x).name)
    q_lookup = query_df.set_index("image_name", drop=False)

    out = []

    for _, row in df.iterrows():
        if not bool(row.get("localized", False)):
            continue

        if pd.isna(row.get("map_lat")) or pd.isna(row.get("map_lon")):
            continue

        inliers = int(row.get("inliers", 0))
        if inliers < min_inliers:
            continue

        name = row["image_name"]
        if name not in q_lookup.index:
            continue

        idx = int(q_lookup.loc[name]["query_index"])
        xy = latlon_to_enu2(float(row["map_lat"]), float(row["map_lon"]), float(origin["alt"]), origin)

        good_matches = float(row.get("good_matches", 0.0))

        med_err_raw = row.get("median_reproj_error_px", np.nan)
        med_err = float(med_err_raw) if pd.notna(med_err_raw) else 999.0

        inlier_conf = 0.35 + 0.65 * min(
            1.0,
            max(0.0, (inliers - min_inliers) / max(1.0, 2.0 * min_inliers)),
        )

        err_conf = 1.0 / (1.0 + max(0.0, med_err) / 8.0)
        match_conf = min(1.0, good_matches / 80.0)

        confidence = 0.55 * inlier_conf + 0.35 * err_conf + 0.10 * match_conf
        confidence = float(np.clip(confidence, 0.05, 1.0))

        dynamic_weight = base_w * (0.25 + 1.75 * confidence)

        out.append(
            {
                "idx": idx,
                "xy": xy,
                "weight": dynamic_weight,
                "source": "map_prior",
                "inliers": inliers,
                "confidence": confidence,
                "image_name": name,
                "ref": "",
                "median_reproj_error_px": med_err,
                "good_matches": good_matches,
            }
        )

    return out


def build_map_confidence(n: int, map_meas: list[dict]) -> np.ndarray:
    map_conf = np.zeros(n, dtype=float)

    for m in map_meas:
        if m.get("source") != "map_prior":
            continue

        idx = int(m["idx"])
        conf = float(m.get("confidence", 0.0))

        if 0 <= idx < n:
            map_conf[idx] = max(map_conf[idx], conf)

    return map_conf


def weaken_reference_when_map_confident(
    ref_meas: list[dict],
    map_conf: np.ndarray,
    cfg: dict,
) -> list[dict]:
    opt = cfg["option2"]["fusion"]

    if not bool(opt.get("dynamic_view_center_weights", True)):
        return ref_meas

    reduction = float(opt.get("reference_reduction_when_map_confident", 0.75))
    min_factor = float(opt.get("min_reference_weight_factor", 0.20))

    updated = []

    for m in ref_meas:
        mm = dict(m)
        idx = int(mm["idx"])

        conf = float(map_conf[idx]) if 0 <= idx < len(map_conf) else 0.0

        factor = 1.0 - reduction * conf
        factor = max(min_factor, factor)

        mm["weight"] = float(mm["weight"]) * factor
        mm["map_conf_at_frame"] = conf

        updated.append(mm)

    return updated


def load_vo_steps(vo_csv: Path, n: int) -> np.ndarray:
    if not vo_csv.exists():
        print(f"VO debug CSV not found: {vo_csv}. Continuing with zero VO steps.")
        return np.zeros((n, 2), dtype=float)

    df = pd.read_csv(vo_csv)
    steps = np.zeros((n, 2), dtype=float)

    if not {"dx", "dy"}.issubset(df.columns):
        return steps

    m = min(n, len(df))
    if "motion_ok" in df.columns:
        ok = df["motion_ok"].fillna(False).astype(bool).to_numpy()[:m]
    else:
        ok = np.ones(m, dtype=bool)

    dx = df["dx"].fillna(0.0).astype(float).to_numpy()[:m]
    dy = df["dy"].fillna(0.0).astype(float).to_numpy()[:m]

    steps[:m, 0] = np.where(ok, dx, 0.0)
    steps[:m, 1] = np.where(ok, dy, 0.0)
    return steps


def estimate_vo_transform(vo_steps: np.ndarray, init_xy: np.ndarray) -> np.ndarray:
    if len(init_xy) < 4:
        return np.eye(2)

    v = vo_steps[1:]
    r = init_xy[1:] - init_xy[:-1]

    valid = np.linalg.norm(v, axis=1) > 1e-6
    if valid.sum() < 5:
        return np.eye(2)

    B, *_ = np.linalg.lstsq(v[valid], r[valid], rcond=None)
    A = B.T

    if not np.all(np.isfinite(A)):
        return np.eye(2)

    det = np.linalg.det(A)
    scale = np.sqrt(abs(det)) if np.isfinite(det) else 0.0

    if scale < 0.001 or scale > 30:
        return np.eye(2)

    return A


def bounds_from_config(cfg: dict, origin: dict) -> tuple[float, float, float, float] | None:
    b = cfg["option2"]["fusion"].get("map_bounds")
    if b is None:
        geo = cfg.get("geo_filter", {})
        if geo.get("enabled", False):
            b = {
                "min_lat": geo["min_lat"],
                "max_lat": geo["max_lat"],
                "min_lon": geo["min_lon"],
                "max_lon": geo["max_lon"],
            }

    if b is None:
        return None

    corners = [
        latlon_to_enu2(float(b["min_lat"]), float(b["min_lon"]), float(origin["alt"]), origin),
        latlon_to_enu2(float(b["min_lat"]), float(b["max_lon"]), float(origin["alt"]), origin),
        latlon_to_enu2(float(b["max_lat"]), float(b["min_lon"]), float(origin["alt"]), origin),
        latlon_to_enu2(float(b["max_lat"]), float(b["max_lon"]), float(origin["alt"]), origin),
    ]

    arr = np.vstack(corners)
    return float(arr[:, 0].min()), float(arr[:, 0].max()), float(arr[:, 1].min()), float(arr[:, 1].max())


def optimize(
    query_df: pd.DataFrame,
    init_xy: np.ndarray,
    measurements: list[dict],
    vo_steps: np.ndarray,
    cfg: dict,
    origin: dict,
) -> np.ndarray:
    opt = cfg["option2"]["fusion"]

    n = len(query_df)
    t = query_df["time_sec"].astype(float).to_numpy()

    map_conf = np.zeros(n, dtype=float)
    for m in measurements:
        if m.get("source") == "map_prior":
            idx = int(m["idx"])
            if 0 <= idx < n:
                map_conf[idx] = max(map_conf[idx], float(m.get("confidence", 0.0)))

    dynamic_weights = bool(opt.get("dynamic_view_center_weights", True))
    speed_relief = float(opt.get("speed_relief_from_map_confidence", 0.95))
    smooth_relief = float(opt.get("smoothness_relief_from_map_confidence", 0.65))

    A = estimate_vo_transform(vo_steps, init_xy)
    vo_m = np.array([A @ step for step in vo_steps], dtype=float)

    vo_w = float(opt.get("vo_motion_weight", 2.5))
    smooth_w = float(opt.get("smoothness_weight", 1.0))
    prior_w = float(opt.get("initial_route_prior_weight", 0.10))
    speed_w = float(opt.get("speed_soft_weight", 0.20))
    max_speed = float(opt.get("max_speed_mps", 250.0))
    bounds_w = float(opt.get("map_bounds_weight", 5.0))
    bounds_margin = float(opt.get("map_bounds_margin_m", 20.0))
    enu_bounds = bounds_from_config(cfg, origin)

    x0 = init_xy.reshape(-1)

    def residual(vec: np.ndarray) -> np.ndarray:
        xy = vec.reshape(n, 2)
        res = []

        for m in measurements:
            i = int(m["idx"])
            if 0 <= i < n:
                res.extend((float(m["weight"]) * (xy[i] - m["xy"])).tolist())

        for i in range(1, n):
            dt = max(1e-3, t[i] - t[i - 1])
            delta = xy[i] - xy[i - 1]

            res.extend((vo_w * (delta - vo_m[i])).tolist())

            speed = float(np.linalg.norm(delta) / dt)

            if dynamic_weights:
                local_map_conf = max(map_conf[i], map_conf[i - 1])
                local_speed_w = speed_w * (1.0 - speed_relief * local_map_conf)
                local_speed_w = max(0.02 * speed_w, local_speed_w)
            else:
                local_speed_w = speed_w

            res.append(local_speed_w * max(0.0, speed - max_speed))

        for i in range(1, n - 1):
            dt1 = max(1e-3, t[i] - t[i - 1])
            dt2 = max(1e-3, t[i + 1] - t[i])

            v1 = (xy[i] - xy[i - 1]) / dt1
            v2 = (xy[i + 1] - xy[i]) / dt2

            if dynamic_weights:
                local_map_conf = map_conf[i]
                local_smooth_w = smooth_w * (1.0 - smooth_relief * local_map_conf)
                local_smooth_w = max(0.10 * smooth_w, local_smooth_w)
            else:
                local_smooth_w = smooth_w

            res.extend((local_smooth_w * (v2 - v1)).tolist())

        if prior_w > 0:
            res.extend((prior_w * (xy - init_xy)).reshape(-1).tolist())

        if enu_bounds is not None and bounds_w > 0:
            min_x, max_x, min_y, max_y = enu_bounds
            for i in range(n):
                x, y = xy[i]
                res.append(bounds_w * max(0.0, (min_x - bounds_margin) - x))
                res.append(bounds_w * max(0.0, x - (max_x + bounds_margin)))
                res.append(bounds_w * max(0.0, (min_y - bounds_margin) - y))
                res.append(bounds_w * max(0.0, y - (max_y + bounds_margin)))

        return np.array(res, dtype=float)

    print("Running option-2 dynamic view-center global optimization...")
    print(f"Dynamic view-center weights: {dynamic_weights}")
    print(f"Map confidence frames: {(map_conf > 0).sum()}/{n}")
    print(f"Mean map confidence over confident frames: {map_conf[map_conf > 0].mean() if (map_conf > 0).any() else 0:.3f}")

    result = least_squares(
        residual,
        x0,
        loss=str(opt.get("robust_loss", "soft_l1")),
        f_scale=float(opt.get("robust_f_scale", 20.0)),
        max_nfev=int(opt.get("max_nfev", 200)),
        verbose=1,
    )

    print(f"Optimization success: {result.success}")
    print(f"Final cost: {result.cost:.3f}")

    return result.x.reshape(n, 2)


def write_kml(df: pd.DataFrame, path: Path) -> None:
    try:
        import simplekml
    except ImportError as exc:
        raise ImportError("Install simplekml first: pip install simplekml") from exc

    kml = simplekml.Kml()
    coords = []

    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        alt = float(row["alt"]) if "alt" in row and pd.notna(row["alt"]) else 0.0

        coords.append((lon, lat, alt))

        p = kml.newpoint(name="", coords=[(lon, lat, alt)])
        p.description = (
            f"<b>image:</b> {row['image_name']}<br/>"
            f"<b>time:</b> {float(row['time_sec']):.2f}<br/>"
            f"<b>measurement_sources:</b> {row['measurement_sources']}<br/>"
            f"<b>map_confidence:</b> {float(row.get('map_confidence', 0.0)):.3f}<br/>"
            f"<b>optimizer_offset_m:</b> {float(row['optimizer_offset_m']):.2f}<br/>"
        )
        p.style.iconstyle.scale = 0.35

    if len(coords) >= 2:
        line = kml.newlinestring(name="Option 2 dynamic view-center route", coords=coords)
        line.style.linestyle.width = 4
        line.style.linestyle.color = simplekml.Color.red

    path.parent.mkdir(parents=True, exist_ok=True)
    kml.save(str(path))


def write_outputs(
    query_df: pd.DataFrame,
    xy: np.ndarray,
    init_xy: np.ndarray,
    measurements: list[dict],
    origin: dict,
    out_csv: Path,
    out_kml: Path,
) -> None:
    by_idx: dict[int, list[str]] = {}
    map_conf = np.zeros(len(query_df), dtype=float)

    for m in measurements:
        idx = int(m["idx"])
        by_idx.setdefault(idx, []).append(str(m["source"]))
        if m.get("source") == "map_prior":
            map_conf[idx] = max(map_conf[idx], float(m.get("confidence", 0.0)))

    rows = []

    for i, row in query_df.iterrows():
        lat, lon, alt = enu2_to_latlon(xy[i], origin)
        init_lat, init_lon, _ = enu2_to_latlon(init_xy[i], origin)

        rows.append(
            {
                "image_name": row["image_name"],
                "time_sec": float(row["time_sec"]),
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "initial_lat": init_lat,
                "initial_lon": init_lon,
                "optimizer_offset_m": float(np.linalg.norm(xy[i] - init_xy[i])),
                "measurement_sources": ",".join(sorted(set(by_idx.get(i, [])))),
                "map_confidence": float(map_conf[i]),
            }
        )

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_kml(out, out_kml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query-meta", default="outputs/query_metadata.csv")
    parser.add_argument("--initial-route", default="outputs/anchor_vo_v2_route.csv")
    parser.add_argument("--matches", default="outputs/fast_reranked_matches.csv")
    parser.add_argument("--satellite-prior", default="outputs/option2_satellite_prior.csv")
    parser.add_argument("--vo-debug", default="outputs/anchor_vo_v2_vo_debug.csv")
    parser.add_argument("--geo-alignment", default="outputs/geo_alignment.json")
    parser.add_argument("--out-csv", default="outputs/option2_fused_route.csv")
    parser.add_argument("--out-kml", default="outputs/option2_fused_route.kml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    origin = load_origin(Path(args.geo_alignment))
    query_df = prepare_query(Path(args.query_meta))

    init_xy = load_initial_route(Path(args.initial_route), query_df, origin)

    ref_meas = load_reference_anchors(Path(args.matches), query_df, origin, cfg)
    map_meas = load_map_priors(Path(args.satellite_prior), query_df, origin, cfg)

    map_conf = build_map_confidence(len(query_df), map_meas)
    ref_meas = weaken_reference_when_map_confident(ref_meas, map_conf, cfg)

    measurements = ref_meas + map_meas

    vo_steps = load_vo_steps(Path(args.vo_debug), len(query_df))

    print(f"Query frames: {len(query_df)}")
    print(f"Reference anchor measurements: {len(ref_meas)}")
    print(f"Satellite/map prior measurements: {len(map_meas)}")
    print(f"Total measurements: {len(measurements)}")

    optimized_xy = optimize(query_df, init_xy, measurements, vo_steps, cfg, origin)

    write_outputs(
        query_df=query_df,
        xy=optimized_xy,
        init_xy=init_xy,
        measurements=measurements,
        origin=origin,
        out_csv=Path(args.out_csv),
        out_kml=Path(args.out_kml),
    )

    print(f"Saved fused route CSV: {args.out_csv}")
    print(f"Saved fused route KML: {args.out_kml}")


if __name__ == "__main__":
    main()
