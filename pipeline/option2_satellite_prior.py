from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_image(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def resize_keep_aspect(img: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img, 1.0
    scale = max_width / img.shape[1]
    resized = cv2.resize(img, (max_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def make_detector(name: str):
    name = name.lower()
    if name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=7000), "sift"
    return cv2.ORB_create(nfeatures=10000, fastThreshold=8), "orb"


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {angle}")


def center_in_rotated_coords(w: int, h: int, angle: int) -> tuple[float, float]:
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    if angle == 0:
        return cx, cy
    if angle == 90:
        return h - 1 - cy, cx
    if angle == 180:
        return w - 1 - cx, h - 1 - cy
    if angle == 270:
        return cy, w - 1 - cx
    raise ValueError(angle)


def match_descriptors(desc_q, desc_m, detector_type: str, ratio: float):
    if desc_q is None or desc_m is None or len(desc_q) < 8 or len(desc_m) < 8:
        return []
    if detector_type == "sift":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=80))
        raw = matcher.knnMatch(desc_q, desc_m, k=2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(desc_q, desc_m, k=2)

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def pixel_to_latlon(x: float, y: float, bounds: dict, width: int, height: int) -> tuple[float, float]:
    # Assumes the map image is north-up and not rotated.
    top_left_lat = float(bounds["top_left_lat"])
    top_left_lon = float(bounds["top_left_lon"])
    bottom_right_lat = float(bounds["bottom_right_lat"])
    bottom_right_lon = float(bounds["bottom_right_lon"])

    lon = top_left_lon + (x / max(1, width - 1)) * (bottom_right_lon - top_left_lon)
    lat = top_left_lat + (y / max(1, height - 1)) * (bottom_right_lat - top_left_lat)
    return lat, lon


def estimate_query_on_map(
    query_img: np.ndarray,
    map_kp,
    map_desc,
    map_shape: tuple[int, int],
    detector,
    detector_type: str,
    cfg: dict,
) -> dict:
    c = cfg["option2"]["satellite_prior"]
    rotations = [int(v) for v in c.get("rotations", [0, 90, 180, 270])]
    ratio = float(c.get("ratio", 0.75))
    ransac_px = float(c.get("ransac_px", 5.0))

    h0, w0 = query_img.shape[:2]
    best = None

    for angle in rotations:
        qrot = rotate_image(query_img, angle)
        qgray = cv2.cvtColor(qrot, cv2.COLOR_BGR2GRAY)
        qkp, qdesc = detector.detectAndCompute(qgray, None)

        good = match_descriptors(qdesc, map_desc, detector_type, ratio)
        if len(good) < 8:
            continue

        src = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([map_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
        if H is None or mask is None:
            continue

        mask_flat = mask.ravel().astype(bool)
        inliers = int(mask_flat.sum())
        if inliers < 4:
            continue

        projected = cv2.perspectiveTransform(src, H)
        errors = np.linalg.norm(projected.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
        med_err = float(np.median(errors[mask_flat])) if inliers > 0 else 999.0

        # Map the center of the query frame to the map.
        cx, cy = center_in_rotated_coords(w0, h0, angle)
        center = np.array([[[cx, cy]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(center, H)[0, 0]
        mx, my = float(mapped[0]), float(mapped[1])

        mh, mw = map_shape
        inside = 0 <= mx < mw and 0 <= my < mh

        score = inliers - 0.4 * med_err + 0.02 * len(good)
        if not inside:
            score -= 1000.0

        cand = {
            "localized": bool(inside),
            "map_x": mx,
            "map_y": my,
            "inliers": inliers,
            "good_matches": len(good),
            "median_reproj_error_px": med_err,
            "rotation_deg": angle,
            "score": score,
        }

        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        return {
            "localized": False,
            "map_x": None,
            "map_y": None,
            "inliers": 0,
            "good_matches": 0,
            "median_reproj_error_px": None,
            "rotation_deg": None,
            "score": -1e9,
        }

    return best


def write_kml(df: pd.DataFrame, out_kml: Path) -> None:
    try:
        import simplekml
    except ImportError as exc:
        raise ImportError("Install simplekml first: pip install simplekml") from exc

    kml = simplekml.Kml()
    coords = []

    valid = df[df["localized"] == True].copy()
    for _, row in valid.iterrows():
        lat = float(row["map_lat"])
        lon = float(row["map_lon"])
        coords.append((lon, lat, 0.0))
        p = kml.newpoint(name="", coords=[(lon, lat, 0.0)])
        p.description = (
            f"<b>image:</b> {row['image_name']}<br/>"
            f"<b>inliers:</b> {row['inliers']}<br/>"
            f"<b>matches:</b> {row['good_matches']}<br/>"
            f"<b>median_error:</b> {row['median_reproj_error_px']}<br/>"
            f"<b>rotation:</b> {row['rotation_deg']}<br/>"
        )
        p.style.iconstyle.scale = 0.55
        p.style.iconstyle.color = simplekml.Color.cyan

    if len(coords) >= 2:
        line = kml.newlinestring(name="Satellite prior route", coords=coords)
        line.style.linestyle.width = 3
        line.style.linestyle.color = simplekml.Color.cyan

    out_kml.parent.mkdir(parents=True, exist_ok=True)
    kml.save(str(out_kml))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query-meta", default="outputs/query_metadata.csv")
    parser.add_argument("--query-frames", default="outputs/query_frames")
    parser.add_argument("--out-csv", default="outputs/option2_satellite_prior.csv")
    parser.add_argument("--out-kml", default="outputs/option2_satellite_prior.kml")
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    c = cfg["option2"]["satellite_prior"]

    map_path = Path(c.get("map_path", cfg["paths"].get("reference_map_path", "reference_map.png")))
    if not map_path.exists():
        raise FileNotFoundError(f"Map image not found: {map_path}")

    if "bounds" not in c:
        raise ValueError("Missing option2.satellite_prior.bounds in config.yaml")

    query_df = pd.read_csv(args.query_meta)
    query_df["image_name"] = query_df["image_name"].astype(str).map(lambda x: Path(x).name)
    query_df = query_df.sort_values("time_sec").reset_index(drop=True)

    detector, detector_type = make_detector(str(c.get("detector", "sift")))
    query_resize_width = int(c.get("query_resize_width", 960))

    map_img = cv2.imread(str(map_path))
    if map_img is None:
        raise RuntimeError(f"Could not read map image: {map_path}")

    map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    print("Computing map features...")
    map_kp, map_desc = detector.detectAndCompute(map_gray, None)
    print(f"Map features: {0 if map_kp is None else len(map_kp)}")

    rows = []
    qdir = Path(args.query_frames)

    step = int(c.get("frame_step", 1))
    max_frames = int(c.get("max_frames", 0))

    selected = query_df.iloc[::max(1, step)].copy()
    if max_frames > 0:
        selected = selected.iloc[:max_frames]

    for count, (_, row) in enumerate(selected.iterrows(), start=1):
        name = Path(str(row["image_name"])).name
        qpath = find_image(qdir, name)
        base = row.to_dict()
        base.update(
            {
                "localized": False,
                "map_x": None,
                "map_y": None,
                "map_lat": None,
                "map_lon": None,
                "inliers": 0,
                "good_matches": 0,
                "median_reproj_error_px": None,
                "rotation_deg": None,
                "score": -1e9,
            }
        )

        if qpath is None:
            rows.append(base)
            continue

        qimg = cv2.imread(str(qpath))
        if qimg is None:
            rows.append(base)
            continue

        qimg, scale = resize_keep_aspect(qimg, query_resize_width)
        est = estimate_query_on_map(qimg, map_kp, map_desc, map_img.shape[:2], detector, detector_type, cfg)

        base.update(est)
        if est["localized"]:
            lat, lon = pixel_to_latlon(est["map_x"], est["map_y"], c["bounds"], map_img.shape[1], map_img.shape[0])
            base["map_lat"] = lat
            base["map_lon"] = lon

        rows.append(base)

        if count % 20 == 0:
            localized = sum(1 for r in rows if r["localized"])
            print(f"Processed {count}/{len(selected)} query frames | localized={localized}")

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    write_kml(out, Path(args.out_kml))

    print(f"Saved satellite prior CSV: {args.out_csv}")
    print(f"Saved satellite prior KML: {args.out_kml}")
    print(f"Localized frames: {int(out['localized'].sum())}/{len(out)}")


if __name__ == "__main__":
    main()
