from __future__ import annotations

from pathlib import Path
import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ],
        dtype=float,
    )


def camera_center_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # COLMAP stores world-to-camera: x_cam = R*x_world + t.
    # Camera center in world coordinates is C = -R^T t.
    r = qvec_to_rotmat(qvec)
    return -r.T @ tvec


def read_images_txt(path: str | Path) -> dict[str, dict]:
    """Read COLMAP images.txt. Returns mapping image_name -> qvec/tvec/camera_center."""
    path = Path(path)
    images: dict[str, dict] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # images.txt has two non-comment lines per image.
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=float)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=float)
            camera_id = int(parts[8])
            image_name = parts[9]
            center = camera_center_from_qt(qvec, tvec)
            images[image_name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_center": center,
            }
        i += 2

    return images


def find_images_txt(sfm_dir: str | Path) -> Path:
    sfm_dir = Path(sfm_dir)
    candidates = list(sfm_dir.rglob("images.txt"))
    if not candidates:
        raise FileNotFoundError(f"Could not find images.txt under {sfm_dir}. Convert COLMAP model to TXT first.")
    # Prefer sparse/0/images.txt if present.
    candidates.sort(key=lambda p: (0 if "sparse" in str(p) else 1, len(str(p))))
    return candidates[0]
