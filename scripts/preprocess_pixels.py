#!/usr/bin/env python3
"""Build the four-channel grid tensors used by the congestion experiment."""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FEATURE_PATHS = (
    "routability_features_decompressed/macro_region",
    "routability_features_decompressed/cell_density",
    "routability_features_decompressed/RUDY/RUDY",
    "routability_features_decompressed/RUDY/RUDY_pin",
)
LABEL_H_PATH = (
    "routability_features_decompressed/congestion/congestion_global_routing/"
    "overflow_based/congestion_GR_horizontal_overflow"
)
LABEL_V_PATH = (
    "routability_features_decompressed/congestion/congestion_global_routing/"
    "overflow_based/congestion_GR_vertical_overflow"
)


def minmax_normalize(array):
    array = np.asarray(array)
    value_range = array.max() - array.min()
    if value_range == 0:
        return np.zeros_like(array)
    return (array - array.min()) / value_range


def preprocess(data_root, output_dir, overwrite=False):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_dirs = [data_root / path for path in FEATURE_PATHS]
    label_h_dir = data_root / LABEL_H_PATH
    label_v_dir = data_root / LABEL_V_PATH
    required_dirs = [*feature_dirs, label_h_dir, label_v_dir]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing CircuitNet directories:\n" + "\n".join(missing))

    files = sorted(path.name for path in label_h_dir.iterdir() if path.is_file())
    for file_name in tqdm(files, desc="pixel features"):
        output_path = output_dir / f"{file_name}_pixel.pt"
        if output_path.exists() and not overwrite:
            continue

        features = []
        for feature_dir in feature_dirs:
            source = feature_dir / file_name
            if not source.is_file():
                raise FileNotFoundError(source)
            features.append(torch.from_numpy(minmax_normalize(np.load(source))).float())

        pixel_features = torch.stack(features, dim=-1)
        label_h = torch.from_numpy(np.load(label_h_dir / file_name)).float()
        label_v = torch.from_numpy(np.load(label_v_dir / file_name)).float()
        if label_h.shape != pixel_features.shape[:2] or label_v.shape != pixel_features.shape[:2]:
            raise ValueError(f"Shape mismatch for {file_name}")

        torch.save(
            {
                "x": pixel_features,
                "y_h": label_h,
                "y_v": label_v,
                "layout_size": pixel_features.shape[:2],
            },
            output_path,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Decompressed CircuitNet-N28 root")
    parser.add_argument("--output-dir", default="data/processed/pixel")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.data_root, args.output_dir, args.overwrite)

