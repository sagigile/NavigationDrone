from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import yaml

from src.wgs84 import geodetic_to_enu, enu_to_geodetic


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_image(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def resize_keep_aspect(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def estimate_pair_motion(img_a: np.ndarray, img_b: np.ndarray, detector, resize_width: int, invert_motion: bool) -> dict:
    a = resize_keep_aspect(img_a, resize_width)
    b = resize_keep_aspect(img_b, resize_width)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    kpa, desa = detector.detectAndCompute(ga, None)
    kpb, desb = detector.detectAndCompute(gb, None)
    if desa is None or desb is None or len(kpa) < 12 or len(kpb) < 12:
        return {"ok": False, "dx": 0.0, "dy": 0.0, "inliers": 0, "matches": 0}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(desa, desb, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.78 * n.distance:
                good.append(m)
    if len(good) < 12:
        return {"ok": False, "dx": 0.0, "dy": 0.0, "inliers": 0, "matches": len(good)}
    src = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    aff, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if aff is None or mask is None:
        return {"ok": False, "dx": 0.0, "dy": 0.0, "inliers": 0, "matches": len(good)}
    inliers = int(mask.ravel().sum())
    if inliers < 8:
        return {"ok": False, "dx": 0.0, "dy": 0.0, "inliers": inliers, "matches": len(good)}
    sign = -1.0 if invert_motion else 1.0
    return {"ok": True, "dx": sign * float(aff[0, 2]), "dy": sign * float(aff[1, 2]), "inliers": inliers, "matches": len(good)}


def compute_vo(query_df: pd.DataFrame, query_frames_dir: Path, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    c = cfg.get("anchor_vo_v2", {})
    resize_width = int(c.get("vo_resize_width", 640))
    orb_features = int(c.get("vo_orb_features", 3500))
    invert_motion = bool(c.get("invert_motion", True))
    detector = cv2.ORB_create(nfeatures=orb_features, fastThreshold=8)
    vo_xy = np.zeros((len(query_df), 2), dtype=float)
    rows = []
    prev_img, prev_name = None, None
    for i, row in query_df.iterrows():
        name = Path(str(row["image_name"])).name
        path = find_image(query_frames_dir, name)
        img = cv2.imread(str(path)) if path is not None else None
        if i == 0 or prev_img is None or img is None:
            rows.append({"image_name": name, "time_sec": row["time_sec"], "previous_image": prev_name, "motion_ok": False, "dx": 0.0, "dy": 0.0, "motion_inliers": 0, "motion_matches": 0})
            prev_img, prev_name = img, name
            continue
        motion = estimate_pair_motion(prev_img, img, detector, resize_width, invert_motion)
        vo_xy[i] = vo_xy[i - 1] + np.array([motion["dx"], motion["dy"]], dtype=float)
        rows.append({"image_name": name, "time_sec": row["time_sec"], "previous_image": prev_name, "motion_ok": bool(motion["ok"]), "dx": motion["dx"], "dy": motion["dy"], "motion_inliers": motion["inliers"], "motion_matches": motion["matches"]})
        prev_img, prev_name = img, name
        if (i + 1) % 25 == 0:
            print(f"VO computed for {i + 1}/{len(query_df)} query frames")
    return pd.DataFrame(rows), vo_xy


def classify_matches(matches_csv: Path, query_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg.get("anchor_vo_v2", {})
    strong_inliers = int(c.get("strong_anchor_min_inliers", 20))
    strong_matches = int(c.get("strong_anchor_min_good_matches", 20))
    weak_inliers = int(c.get("weak_anchor_min_inliers", 10))
    weak_matches = int(c.get("weak_anchor_min_good_matches", 10))
    matches = pd.read_csv(matches_csv)
    if matches.empty:
        return pd.DataFrame()
    matches["query_image"] = matches["query_image"].astype(str).map(lambda x: Path(x).name)
    matches["reference_image"] = matches["reference_image"].astype(str).map(lambda x: Path(x).name)
    q = query_df.copy()
    q["image_name"] = q["image_name"].astype(str).map(lambda x: Path(x).name)
    q_lookup = q.reset_index().set_index("image_name", drop=False)
    m = matches[matches["reranked_rank"] == 1].copy()
    rows = []
    for _, row in m.iterrows():
        qname = row["query_image"]
        if qname not in q_lookup.index or pd.isna(row.get("reference_lat")) or pd.isna(row.get("reference_lon")):
            continue
        inliers = int(row.get("inliers", 0))
        good = int(row.get("good_matches", 0))
        level = "bad"
        if inliers >= strong_inliers and good >= strong_matches:
            level = "strong"
        elif inliers >= weak_inliers and good >= weak_matches:
            level = "weak"
        qrow = q_lookup.loc[qname]
        rows.append({"query_index": int(qrow["index"]), "query_image": qname, "time_sec": float(qrow["time_sec"]), "reference_image": row["reference_image"], "lat": float(row["reference_lat"]), "lon": float(row["reference_lon"]), "alt": float(row["reference_alt"]) if "reference_alt" in row and pd.notna(row["reference_alt"]) else np.nan, "inliers": inliers, "good_matches": good, "rotation_deg": row.get("rotation_deg", None), "anchor_level": level})
    out = pd.DataFrame(rows)
    return out.sort_values("query_index").reset_index(drop=True) if not out.empty else out


def reject_impossible_strong_anchors(anchors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg.get("anchor_vo_v2", {})
    max_speed = float(c.get("strong_anchor_max_speed_mps", 70.0))
    max_jump = float(c.get("strong_anchor_max_jump_m", 350.0))
    max_gap = float(c.get("strong_anchor_max_gap_sec", 300.0))
    strong = anchors[anchors["anchor_level"] == "strong"].copy().reset_index(drop=True)
    if strong.empty:
        return strong
    states = strong.to_dict("records")
    dp, prev = [0.0] * len(states), [None] * len(states)
    best_i, best_score = 0, -1e18
    for i, s in enumerate(states):
        dp[i] = 100 + float(s["inliers"]) + 0.2 * float(s["good_matches"])
        for j in range(i):
            p = states[j]
            dt = float(s["time_sec"]) - float(p["time_sec"])
            if dt <= 0 or dt > max_gap:
                continue
            dist = haversine_m(float(p["lat"]), float(p["lon"]), float(s["lat"]), float(s["lon"]))
            if dist > max(max_jump, max_speed * dt):
                continue
            score = dp[j] + 100 + float(s["inliers"]) + 0.2 * float(s["good_matches"]) - 0.02 * dist
            if score > dp[i]:
                dp[i], prev[i] = score, j
        if dp[i] > best_score:
            best_score, best_i = dp[i], i
    chain = []
    cur = best_i
    while cur is not None:
        chain.append(states[cur])
        cur = prev[cur]
    chain.reverse()
    return pd.DataFrame(chain).sort_values("query_index").reset_index(drop=True)


def enu_from_anchor(anchor: dict, origin: dict) -> np.ndarray:
    alt = float(anchor["alt"]) if not pd.isna(anchor.get("alt")) else float(origin["alt"])
    return geodetic_to_enu(float(anchor["lat"]), float(anchor["lon"]), alt, float(origin["lat"]), float(origin["lon"]), float(origin["alt"]))


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def fit_vo_to_anchors(vo_a: np.ndarray, vo_b: np.ndarray, enu_a: np.ndarray, enu_b: np.ndarray):
    dv = vo_b - vo_a
    de = enu_b[:2] - enu_a[:2]
    ndv, nde = float(np.linalg.norm(dv)), float(np.linalg.norm(de))
    if ndv < 1e-6 or nde < 1e-6:
        return None
    theta = math.atan2(de[1], de[0]) - math.atan2(dv[1], dv[0])
    return nde / ndv, rot2(theta)


def interpolate_segment(query_df, vo_xy, a, b, weak_inside, origin, cfg):
    c = cfg.get("anchor_vo_v2", {})
    weak_weight = float(c.get("weak_anchor_pull_weight", 0.35))
    radius = max(1.0, float(c.get("weak_anchor_frame_radius", 20)))
    ia, ib = int(a["query_index"]), int(b["query_index"])
    enu_a, enu_b = enu_from_anchor(a, origin), enu_from_anchor(b, origin)
    fit = fit_vo_to_anchors(vo_xy[ia], vo_xy[ib], enu_a, enu_b)
    weak_items = []
    for _, w in weak_inside.iterrows():
        iw = int(w["query_index"])
        if ia < iw < ib:
            weak_items.append((iw, enu_from_anchor(w.to_dict(), origin)))
    rows = []
    for idx in range(ia, ib + 1):
        q = query_df.iloc[idx]
        alpha = (idx - ia) / max(1, ib - ia)
        if fit is not None:
            scale, R = fit
            enu2 = enu_a[:2] + scale * (R @ (vo_xy[idx] - vo_xy[ia]))
            enu = np.array([enu2[0], enu2[1], (1 - alpha) * enu_a[2] + alpha * enu_b[2]], dtype=float)
            method = "vo_scaled_between_strong_anchors"
        else:
            enu = (1 - alpha) * enu_a + alpha * enu_b
            method = "linear_between_strong_anchors"
        corr, total_w = np.zeros(3), 0.0
        for iw, wenu in weak_items:
            d = abs(idx - iw)
            wgt = math.exp(-(d * d) / (2 * radius * radius))
            if wgt >= 0.03:
                corr += wgt * (wenu - enu)
                total_w += wgt
        weak_here = any(iw == idx for iw, _ in weak_items)
        if total_w > 0:
            enu = enu + weak_weight * corr / total_w
            method += "+weak_pull"
        lat, lon, alt = enu_to_geodetic(enu, float(origin["lat"]), float(origin["lon"]), float(origin["alt"]))
        is_strong = idx == ia or idx == ib
        rows.append({"image_name": q["image_name"], "time_sec": float(q["time_sec"]), "lat": lat, "lon": lon, "alt": alt, "localized": True, "is_strong_anchor": bool(is_strong), "is_weak_anchor": bool(weak_here), "confidence": "strong" if is_strong else ("weak" if weak_here else "vo_bridge"), "method": method, "segment_start_anchor": a["query_image"], "segment_end_anchor": b["query_image"], "segment_start_reference": a["reference_image"], "segment_end_reference": b["reference_image"]})
    return rows


def build_route(query_df, vo_xy, anchors_all, strong_selected, origin, cfg):
    if len(strong_selected) < 2:
        print("Need at least two selected strong anchors. Try lowering strong thresholds.")
        return pd.DataFrame()
    weak_all = anchors_all[anchors_all["anchor_level"] == "weak"].copy()
    strong = strong_selected.to_dict("records")
    rows = []
    for a, b in zip(strong[:-1], strong[1:]):
        ia, ib = int(a["query_index"]), int(b["query_index"])
        weak_inside = weak_all[(weak_all["query_index"] > ia) & (weak_all["query_index"] < ib)]
        rows.extend(interpolate_segment(query_df, vo_xy, a, b, weak_inside, origin, cfg))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["image_name"], keep="first").sort_values("time_sec").reset_index(drop=True)


def write_kml(route: pd.DataFrame, out_kml: Path) -> None:
    import simplekml
    kml = simplekml.Kml()
    coords = []
    for _, row in route.iterrows():
        lat, lon = float(row["lat"]), float(row["lon"])
        alt = float(row["alt"]) if pd.notna(row.get("alt")) else 0.0
        coords.append((lon, lat, alt))
        p = kml.newpoint(name="", coords=[(lon, lat, alt)])
        p.description = f"<b>time:</b> {row['time_sec']:.2f}<br/><b>confidence:</b> {row['confidence']}<br/><b>method:</b> {row['method']}<br/><b>start_ref:</b> {row['segment_start_reference']}<br/><b>end_ref:</b> {row['segment_end_reference']}<br/>"
        if row["confidence"] == "strong":
            p.style.iconstyle.scale = 0.8
            p.style.iconstyle.color = simplekml.Color.green
        elif row["confidence"] == "weak":
            p.style.iconstyle.scale = 0.65
            p.style.iconstyle.color = simplekml.Color.yellow
        else:
            p.style.iconstyle.scale = 0.45
            p.style.iconstyle.color = simplekml.Color.orange
    if len(coords) >= 2:
        line = kml.newlinestring(name="Anchor + VO V2 route", coords=coords)
        line.style.linestyle.width = 4
        line.style.linestyle.color = simplekml.Color.red
    out_kml.parent.mkdir(parents=True, exist_ok=True)
    kml.save(str(out_kml))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query-meta", default="outputs/query_metadata.csv")
    parser.add_argument("--query-frames", default="outputs/query_frames")
    parser.add_argument("--matches", default="outputs/fast_reranked_matches.csv")
    parser.add_argument("--geo-alignment", default="outputs/geo_alignment.json")
    parser.add_argument("--out-route-csv", default="outputs/anchor_vo_v2_route.csv")
    parser.add_argument("--out-route-kml", default="outputs/anchor_vo_v2_route.kml")
    parser.add_argument("--out-anchors-csv", default="outputs/anchor_vo_v2_anchors.csv")
    parser.add_argument("--out-strong-csv", default="outputs/anchor_vo_v2_selected_strong_anchors.csv")
    parser.add_argument("--out-vo-debug", default="outputs/anchor_vo_v2_vo_debug.csv")
    args = parser.parse_args()
    cfg = load_yaml(Path(args.config))
    query_df = pd.read_csv(args.query_meta)
    query_df["image_name"] = query_df["image_name"].astype(str).map(lambda x: Path(x).name)
    query_df = query_df.sort_values("time_sec").reset_index(drop=True)
    origin = json.loads(Path(args.geo_alignment).read_text(encoding="utf-8"))["origin"]
    print("Computing visual odometry for all query frames...")
    vo_debug, vo_xy = compute_vo(query_df, Path(args.query_frames), cfg)
    Path(args.out_vo_debug).parent.mkdir(parents=True, exist_ok=True)
    vo_debug.to_csv(args.out_vo_debug, index=False)
    print("Classifying reranked matches into strong / weak / bad anchors...")
    anchors_all = classify_matches(Path(args.matches), query_df, cfg)
    anchors_all.to_csv(args.out_anchors_csv, index=False)
    if anchors_all.empty:
        print("No anchors found. Check fast_reranked_matches.csv or lower thresholds.")
        return
    strong_selected = reject_impossible_strong_anchors(anchors_all, cfg)
    strong_selected.to_csv(args.out_strong_csv, index=False)
    print(f"All classified anchors: {len(anchors_all)}")
    print(f"Strong anchor candidates: {int((anchors_all['anchor_level'] == 'strong').sum())}")
    print(f"Weak anchor candidates: {int((anchors_all['anchor_level'] == 'weak').sum())}")
    print(f"Selected strong anchors: {len(strong_selected)}")
    route = build_route(query_df, vo_xy, anchors_all, strong_selected, origin, cfg)
    if route.empty:
        print("Could not build route. Try lowering strong_anchor_min_inliers or rerun fast reranking with better candidates.")
        return
    route.to_csv(args.out_route_csv, index=False)
    write_kml(route, Path(args.out_route_kml))
    print(f"Saved anchors: {args.out_anchors_csv}")
    print(f"Saved selected strong anchors: {args.out_strong_csv}")
    print(f"Saved VO debug: {args.out_vo_debug}")
    print(f"Saved route CSV: {args.out_route_csv}")
    print(f"Saved route KML: {args.out_route_kml}")
    print(f"Route points: {len(route)} / {len(query_df)} query frames")


if __name__ == "__main__":
    main()
