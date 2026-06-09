from __future__ import annotations

import argparse
import sys
from pathlib import Path

HLOC_ROOT = Path(r"C:\Users\taliy\PythonProject\TestNav\Hierarchical-Localization")
sys.path.insert(0, str(HLOC_ROOT))

from src.config import load_config


def _get_matcher_conf(match_features, preferred: str, fallback: str):
    if preferred in match_features.confs:
        return match_features.confs[preferred]
    if preferred.replace("+", "-") in match_features.confs:
        return match_features.confs[preferred.replace("+", "-")]
    if fallback in match_features.confs:
        return match_features.confs[fallback]
    raise KeyError(f"No matcher config found. Tried: {preferred}, {preferred.replace('+','-')}, {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    try:
        from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction
    except ImportError as e:
        raise ImportError(
            "hloc is not installed. Install it with: "
            "git clone --recursive https://github.com/cvg/Hierarchical-Localization.git && pip install -e ."
        ) from e

    out = Path(cfg["paths"]["outputs_dir"])
    images = out / "reference_frames"
    sfm_pairs = out / "pairs-sfm.txt"
    sfm_dir = out / "sfm"
    features_dir = out / "features"

    feature_conf_name = cfg["hloc"]["local_feature"]
    retrieval_conf_name = cfg["hloc"]["retrieval_feature"]

    feature_conf = extract_features.confs[feature_conf_name]
    retrieval_conf = extract_features.confs[retrieval_conf_name]
    matcher_conf = _get_matcher_conf(
        match_features,
        cfg["hloc"]["matcher_preferred"],
        cfg["hloc"]["matcher_fallback"],
    )

    features = extract_features.main(feature_conf, images, feature_path=features_dir / f'{feature_conf["output"]}.h5')
    retrieval = extract_features.main(retrieval_conf, images, feature_path=features_dir / f'{retrieval_conf["output"]}.h5')

    pairs_from_retrieval.main(
        retrieval,
        sfm_pairs,
        num_matched=int(cfg["hloc"].get("retrieval_top_k", 20)),
    )

    matches = match_features.main(
        matcher_conf,
        sfm_pairs,
        features=features,
        matches=features_dir / f'{matcher_conf["output"]}.h5',
    )

    model = reconstruction.main(
        sfm_dir,
        images,
        sfm_pairs,
        features,
        matches,
    )

    print("Built SfM model:")
    print(model)
    print(f"Output SfM dir: {sfm_dir}")


if __name__ == "__main__":
    main()
