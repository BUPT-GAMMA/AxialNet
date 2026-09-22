#!/usr/bin/env python3
"""Run one processed design through a trained congestion checkpoint."""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axialnet.datasets.Dual_dataset import DualFinetuneDataset  # noqa: E402
from axialnet.models import CombinedModel  # noqa: E402
from axialnet.models.backbones.DualCongestionModel import DualCongestionEncoder  # noqa: E402
from axialnet.models.heads.dual_congestion_head import DualCongestionHead  # noqa: E402
from axialnet.solvers.finetune_solver import compute_metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--pixel-root", required=True)
    parser.add_argument("--net-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    dataset = DualFinetuneDataset(
        root_dir=args.net_root,
        file_list=[args.sample_id],
        pixel_root_dir=args.pixel_root,
        net_root_dir=args.net_root,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False))).to(device)
    model = CombinedModel(
        DualCongestionEncoder(hidden_dim=64, num_loops=2, apply_graph_updates=False),
        DualCongestionHead(
            hidden_channels=128,
            feat_channels=128,
            out_channels=1,
            num_layers=3,
            dropout=0.3,
            decode_mode="separate",
        ),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

    target = batch["grid_l1"].congestion_label
    metrics = compute_metrics(
        output["pred_congestion"],
        target,
        batch_sizes=batch["grid_l1"].size,
        ptr=batch["grid_l1"].ptr,
    )
    print(f"sample={args.sample_id} device={device} latency_ms={elapsed_ms:.3f}")
    print({name: float(value) for name, value in metrics.items()})


if __name__ == "__main__":
    main()
