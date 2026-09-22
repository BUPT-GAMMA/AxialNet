# Data pipeline

## Raw inputs

The congestion experiment uses CircuitNet-N28 at its original grid resolution. Four per-grid inputs are stacked in this exact order:

1. macro region
2. cell density
3. RUDY
4. pin RUDY

Each feature map is independently min-max normalized. The targets are the global-routing horizontal and vertical overflow maps; the total congestion target is their sum.

## Pixel tensors

`scripts/preprocess_pixels.py` writes `<sample>_pixel.pt`:

```text
x            float32 [H, W, 4]
y_h          float32 [H, W]
y_v          float32 [H, W]
layout_size  (H, W)
```

## Net tensors

`scripts/preprocess_nets.py` parses placed DEF and the CircuitNet micron-coordinate dictionary. It writes `<sample>_net.pt`:

```text
net_x                         float32 [N_net, 4]
edge_index_net_to_pixel_cell int64   [2, E]
edge_attr_net_to_pixel_cell  float32 [E, 1]
num_nets                     integer
num_pixels                   integer
```

Net features are degree, normalized bounding-box width, normalized bounding-box height, and normalized bounding-box area. An edge connects a net to each grid containing at least one of its pins/cells; the edge attribute is the number of pins/cells from that net in the grid. This sparse cell-incidence representation is used by the paper configuration.

## Split protocol

The canonical lists under `splits/circuitnet_n28_cross_design/` contain:

| File | Designs | Samples | Role |
| --- | --- | ---: | --- |
| `train.txt` | RISCY-a/b, RISCY-FPU-a/b | 705 | train |
| `val.txt` | RISCY-a/b, RISCY-FPU-a/b | 705 | validation |
| `test_seen.txt` | RISCY-a/b, RISCY-FPU-a/b | 5,668 | held-out seen designs |
| `test.txt` | zero-riscy-a/b | 3,164 | paper unseen test |

When a split directory exists, the dataloader reads these lists directly. The `train_count`, `val_count`, `test_count`, and `split_ratio` fields are then ignored, ensuring that every run uses the same split.

## Validation

The fast validator checks all file pairs and split membership without loading every tensor:

```bash
python scripts/validate_dataset.py --pixel-dir PATH --net-dir PATH
```

Add `--deep` to load all 10,242 pairs and verify tensor sizes. This can take substantial time and I/O.
