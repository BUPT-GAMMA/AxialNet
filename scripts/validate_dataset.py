#!/usr/bin/env python3
"""Validate processed file pairs and the canonical paper split."""

import argparse
from pathlib import Path

import torch


def read_ids(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel-dir", required=True)
    parser.add_argument("--net-dir", required=True)
    parser.add_argument("--split-dir", default="splits/circuitnet_n28_cross_design")
    parser.add_argument("--deep", action="store_true", help="Load and validate every tensor pair")
    args = parser.parse_args()

    pixel_dir = Path(args.pixel_dir)
    net_dir = Path(args.net_dir)
    split_dir = Path(args.split_dir)
    expected = {"train": 705, "val": 705, "test_seen": 5668, "test": 3164}
    all_ids = []
    for name, count in expected.items():
        ids = read_ids(split_dir / f"{name}.txt")
        if len(ids) != count:
            raise ValueError(f"{name}: expected {count}, found {len(ids)}")
        all_ids.extend(ids)

    if len(set(all_ids)) != 10242:
        raise ValueError("Split files overlap or do not cover all 10,242 samples")

    missing = []
    for sample_id in all_ids:
        pixel_path = pixel_dir / f"{sample_id}_pixel.pt"
        net_path = net_dir / f"{sample_id}_net.pt"
        if not pixel_path.is_file() or not net_path.is_file():
            missing.append(sample_id)
            continue
        if args.deep:
            pixel = torch.load(pixel_path, map_location="cpu", weights_only=False)
            net = torch.load(net_path, map_location="cpu", weights_only=False)
            h, w = (int(v) for v in pixel["layout_size"])
            if tuple(pixel["x"].shape) != (h, w, 4):
                raise ValueError(f"{sample_id}: invalid pixel shape {tuple(pixel['x'].shape)}")
            if int(net["num_pixels"]) != h * w:
                raise ValueError(f"{sample_id}: pixel/net size mismatch")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} processed pairs; first: {missing[:5]}")
    print("Validated 10,242 samples and canonical split counts.")


if __name__ == "__main__":
    main()

