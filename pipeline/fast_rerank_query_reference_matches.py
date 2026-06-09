
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_pairs(pairs_path: Path, max_per_query: int) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = {}
    with pairs_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            query_name = Path(parts[0]).name
            ref_name = Path(parts[1]).name
            current = pairs.setdefault(query_name, [])
            if len(current) < max_per_query:
                current.append(ref_name)
    return pairs


def make_detector(name: str):
    if name.lower() == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=3000), "sift"
    return cv2.ORB_create(nfeatures=4000, fastThreshold=8), "orb"


def resize_keep_aspect(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    return cv2.resize(img, (max_width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def find_image(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def compute_features(path: Path, detector, resize_width: int):
    img = cv2.imread(str(path))
    if img is None:
        return None, None
    img = resize_keep_aspect(img, resize_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)


def compute_query_features(path: Path, detector, resize_width: int, rotations: list[int]):
    img = cv2.imread(str(path))
    if img is None:
        return {}
    img = resize_keep_aspect(img, resize_width)
    out = {}
    for angle in rotations:
        q = rotate_image(img, angle)
        gray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
        out[angle] = detector.detectAndCompute(gray, None)
    return out


def match_descriptors(desc_q, desc_r, detector_type: str, ratio: float):
    if desc_q is None or desc_r is None or len(desc_q) < 8 or len(desc_r) < 8:
        return []
    if detector_type == "sift":
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=40))
        raw = matcher.knnMatch(desc_q, desc_r, k=2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(desc_q, desc_r, k=2)
    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def score_features(qkp, qdesc, rkp, rdesc, detector_type: str, ratio: float, ransac_px: float) -> dict:
    good = match_descriptors(qdesc, rdesc, detector_type, ratio)
    if len(good) < 8:
        return {"inliers": 0, "good_matches": len(good), "homography_ok": False}
    src = np.float32([qkp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([rkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None or mask is None:
        return {"inliers": 0, "good_matches": len(good), "homography_ok": False}
    return {"inliers": int(mask.ravel().sum()), "good_matches": len(good), "homography_ok": True}


def write_contact_sheet(query_path: Path, ref_path: Path, out_path: Path, label: str) -> bool:
    q = cv2.imread(str(query_path))
    r = cv2.imread(str(ref_path))
    if q is None or r is None:
        return False
    h = 480
    def resize_to_h(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    q2, r2 = resize_to_h(q), resize_to_h(r)
    h2 = min(q2.shape[0], r2.shape[0])
    canvas = cv2.hconcat([q2[:h2], r2[:h2]])
    cv2.putText(canvas, "QUERY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "FAST RERANKED REF", (q2.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label[:150], (20, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), canvas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs-query-loc.txt")
    parser.add_argument("--query-frames", default="outputs/query_frames")
    parser.add_argument("--reference-frames", default="outputs/reference_frames")
    parser.add_argument("--reference-meta", default="outputs/reference_metadata.csv")
    parser.add_argument("--out-csv", default="outputs/fast_reranked_matches.csv")
    parser.add_argument("--out-pairs", default="outputs/pairs-query-fast-reranked.txt")
    parser.add_argument("--contact-dir", default="outputs/fast_reranked_contact_sheets")
    parser.add_argument("--max-candidates-per-query", type=int, default=30)
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument("--detector", default="orb", choices=["orb", "sift"])
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--resize-width", type=int, default=720)
    parser.add_argument("--ransac-px", type=float, default=6.0)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-contact-sheets", type=int, default=80)
    parser.add_argument("--query-step", type=int, default=1)
    parser.add_argument("--max-query-frames", type=int, default=50)
    parser.add_argument("--rotations", default="0")
    args = parser.parse_args()

    start = time.time()
    pairs = read_pairs(Path(args.pairs), args.max_candidates_per_query)
    query_items = list(pairs.items())[::max(1, args.query_step)]
    if args.max_query_frames > 0:
        query_items = query_items[:args.max_query_frames]

    detector, detector_type = make_detector(args.detector)
    rotations = [int(x.strip()) for x in args.rotations.split(",") if x.strip()]
    qdir = Path(args.query_frames)
    rdir = Path(args.reference_frames)

    ref_meta = pd.read_csv(args.reference_meta)
    ref_meta["image_name"] = ref_meta["image_name"].astype(str).map(lambda x: Path(x).name)
    ref_lookup = ref_meta.set_index("image_name", drop=False)

    needed_refs = sorted({r for _, refs in query_items for r in refs})
    print(f"Query frames to process: {len(query_items)}")
    print(f"Unique reference frames needed: {len(needed_refs)}")
    print(f"Detector={detector_type}, rotations={rotations}, resize_width={args.resize_width}")

    ref_cache = {}
    for i, ref_name in enumerate(needed_refs, 1):
        r_path = find_image(rdir, ref_name)
        if r_path is None:
            continue
        kp, desc = compute_features(r_path, detector, args.resize_width)
        if kp is not None:
            ref_cache[ref_name] = (kp, desc, r_path)
        if i % 50 == 0:
            print(f"Precomputed reference features: {i}/{len(needed_refs)}")
    print(f"Reference features ready: {len(ref_cache)}")

    rows = []
    contact_dir = Path(args.contact_dir)
    contact_dir.mkdir(parents=True, exist_ok=True)
    contact_count = 0

    for qi, (query_name, ref_names) in enumerate(query_items, 1):
        q_path = find_image(qdir, query_name)
        if q_path is None:
            continue
        q_features = compute_query_features(q_path, detector, args.resize_width, rotations)
        scored = []
        for original_rank, ref_name in enumerate(ref_names, 1):
            if ref_name not in ref_cache:
                continue
            rkp, rdesc, r_path = ref_cache[ref_name]
            best = {"inliers": 0, "good_matches": 0, "homography_ok": False, "rotation_deg": None}
            for rot, (qkp, qdesc) in q_features.items():
                s = score_features(qkp, qdesc, rkp, rdesc, detector_type, args.ratio, args.ransac_px)
                if s["inliers"] > best["inliers"] or (s["inliers"] == best["inliers"] and s["good_matches"] > best["good_matches"]):
                    best = {**s, "rotation_deg": rot}
            ref_row = ref_lookup.loc[ref_name].to_dict() if ref_name in ref_lookup.index else {}
            scored.append({
                "query_image": query_name,
                "reference_image": ref_name,
                "original_rank": original_rank,
                "inliers": int(best["inliers"]),
                "good_matches": int(best["good_matches"]),
                "rotation_deg": best["rotation_deg"],
                "homography_ok": bool(best["homography_ok"]),
                "reference_lat": ref_row.get("lat", None),
                "reference_lon": ref_row.get("lon", None),
                "reference_alt": ref_row.get("alt", None),
            })
        scored.sort(key=lambda x: (x["inliers"], x["good_matches"], -x["original_rank"]), reverse=True)
        for new_rank, row in enumerate(scored[:args.keep_top_k], 1):
            row["reranked_rank"] = new_rank
            row["accepted_by_inliers"] = row["inliers"] >= args.min_inliers
            rows.append(row)
            if new_rank == 1 and contact_count < args.max_contact_sheets:
                r_path = find_image(rdir, row["reference_image"])
                if r_path is not None:
                    label = f"{row['query_image']} -> {row['reference_image']} | inliers={row['inliers']} matches={row['good_matches']} rot={row['rotation_deg']}"
                    out_img = contact_dir / f"{contact_count:04d}_{Path(row['query_image']).stem}__{Path(row['reference_image']).stem}.jpg"
                    if write_contact_sheet(q_path, r_path, out_img, label):
                        contact_count += 1
        elapsed = time.time() - start
        eta = elapsed / qi * (len(query_items) - qi)
        best = scored[0] if scored else None
        if best:
            print(f"[{qi}/{len(query_items)}] {query_name}: best={best['reference_image']} inliers={best['inliers']} matches={best['good_matches']} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        else:
            print(f"[{qi}/{len(query_items)}] {query_name}: no candidates")

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    out_pairs = Path(args.out_pairs)
    with out_pairs.open("w", encoding="utf-8") as f:
        for _, row in out_df.iterrows():
            if int(row["inliers"]) >= args.min_inliers:
                f.write(f"{row['query_image']} {row['reference_image']}\n")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved accepted pairs: {out_pairs}")
    print(f"Saved contact sheets: {contact_dir}")
    print(f"Total runtime: {(time.time() - start)/60:.1f} minutes")
    print(f"Rows: {len(out_df)}")
    if len(out_df):
        print(f"Accepted by min_inliers={args.min_inliers}: {int((out_df['inliers'] >= args.min_inliers).sum())}")


if __name__ == "__main__":
    main()
