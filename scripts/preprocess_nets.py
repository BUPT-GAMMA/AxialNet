#!/usr/bin/env python3
"""Build net features and the sparse net-to-grid incidence used in the paper run."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from lefdef import C_DefReader
from tqdm import tqdm


class PositionDatabase:
    def __init__(self, path):
        self.values = np.load(path, allow_pickle=True).item()

    def get(self, name):
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return self.values.get(name)


def _pin_center(pin):
    if hasattr(pin, "c_num_ports") and pin.c_num_ports > 0:
        port = pin.c_ports[0]
        if port.c_num_rects > 0:
            rect = port.c_rects[0]
            return ((rect.c_xl + rect.c_xh) / 4000.0, (rect.c_yl + rect.c_yh) / 4000.0)
    if hasattr(pin, "c_num_rects") and pin.c_num_rects > 0:
        rect = pin.c_rects[0]
        # Keep the historical fallback conversion for compatibility.
        return ((rect.c_xl + rect.c_xh) / 2.0, (rect.c_yl + rect.c_yh) / 2.0)
    return None


def process_design(def_path, position_path, pixel_path, output_path, grid_pitch=2.25):
    positions = PositionDatabase(position_path)
    pixel_data = torch.load(pixel_path, map_location="cpu", weights_only=False)
    height, width = (int(v) for v in pixel_data["layout_size"])
    design = C_DefReader().read(str(def_path))

    instance_centers = {}
    for index in range(design.c_num_components):
        component = design.c_components[index]
        box = positions.get(component.c_id)
        if box is not None:
            instance_centers[component.c_id] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    io_centers = {}
    for index in range(design.c_num_pins):
        pin = design.c_pins[index]
        center = _pin_center(pin)
        if center is not None:
            io_centers[pin.c_name] = center

    net_features = []
    edge_net = []
    edge_grid = []
    edge_count = []

    for index in range(design.c_num_nets):
        net = design.c_nets[index]
        centers = []
        for pin_index in range(net.c_num_pins):
            instance_name = net.c_instances[pin_index]
            pin_name = net.c_pins[pin_index]
            if instance_name in (b"PIN", "PIN"):
                center = io_centers.get(pin_name)
            else:
                center = instance_centers.get(instance_name)
                if center is None and isinstance(instance_name, bytes):
                    center = instance_centers.get(instance_name.decode("utf-8"))
            if center is not None:
                centers.append(center)

        if len(centers) < 2:
            if net.c_num_pins < 2:
                continue
            raise ValueError(f"{def_path}: net {index} has fewer than two located pins")

        xs, ys = zip(*centers)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width_um = max_x - min_x + 1
        height_um = max_y - min_y + 1
        net_index = len(net_features)
        net_features.append([len(centers), width_um, height_um, width_um * height_um])

        counts = Counter()
        for x_coord, y_coord in centers:
            col = max(0, min(width - 1, int(np.floor(x_coord / grid_pitch))))
            row = max(0, min(height - 1, int(np.floor(y_coord / grid_pitch))))
            counts[row * width + col] += 1
        for grid_index, count in counts.items():
            edge_net.append(net_index)
            edge_grid.append(grid_index)
            edge_count.append([count])

    if not net_features:
        raise ValueError(f"No valid nets in {def_path}")

    net_x = torch.tensor(net_features, dtype=torch.float32)
    layout_width_um = width * grid_pitch
    layout_height_um = height * grid_pitch
    net_x[:, 1] /= layout_width_um
    net_x[:, 2] /= layout_height_um
    net_x[:, 3] /= layout_width_um * layout_height_um

    torch.save(
        {
            "net_x": net_x,
            "num_nets": net_x.size(0),
            "num_pixels": height * width,
            "edge_index_net_to_pixel_cell": torch.tensor([edge_net, edge_grid], dtype=torch.long),
            "edge_attr_net_to_pixel_cell": torch.tensor(edge_count, dtype=torch.float32),
        },
        output_path,
    )


def preprocess(def_root, position_dir, pixel_dir, output_dir, grid_pitch=2.25, overwrite=False):
    def_root = Path(def_root)
    position_dir = Path(position_dir)
    pixel_dir = Path(pixel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def_files = sorted(def_root.rglob("*.def"))
    for def_path in tqdm(def_files, desc="net topology"):
        sample_id = def_path.stem
        position_path = position_dir / f"{sample_id}.npy"
        pixel_path = pixel_dir / f"{sample_id}_pixel.pt"
        output_path = output_dir / f"{sample_id}_net.pt"
        if output_path.exists() and not overwrite:
            continue
        if not position_path.is_file() or not pixel_path.is_file():
            continue
        process_design(def_path, position_path, pixel_path, output_path, grid_pitch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--def-root", required=True, help="Root containing placed DEF files")
    parser.add_argument("--position-dir", required=True, help="CircuitNet instance_placement_micron")
    parser.add_argument("--pixel-dir", default="data/processed/pixel")
    parser.add_argument("--output-dir", default="data/processed/net")
    parser.add_argument("--grid-pitch", type=float, default=2.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        args.def_root,
        args.position_dir,
        args.pixel_dir,
        args.output_dir,
        args.grid_pitch,
        args.overwrite,
    )

