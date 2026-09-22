# AxialNet Congestion Prediction

Reference implementation of **AxialNet: Orthogonally Decoupled Layout Learning with Manhattan Bias for Routability Prediction** (ICCAD 2026). This repository contains the complete CircuitNet-N28 congestion pipeline, including preprocessing, canonical splits, training, evaluation, and single-design inference.

This public package intentionally focuses on congestion prediction.

## Results

On the 3,164 unseen `zero-riscy-a/b` designs, the paper reports:

| SSIM | NRMS | Inference latency |
| ---: | ---: | ---: |
| 0.837772 | 0.033517 | 31.408 ms/instance |

## Installation

The paper experiment used Python 3.9, PyTorch 2.6.0+cu124, PyG 2.6.1, and one RTX 4090. A fresh environment can be prepared with:

```bash
conda create -n axialnet python=3.9 -y
conda activate axialnet

# Select the PyTorch command appropriate for your CUDA version if it differs.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

PyG extension wheels are optional for this model on recent PyTorch/PyG releases, but installing the matching wheels from <https://data.pyg.org/whl/> is recommended for the closest environment match.

## Data preparation

Download and decompress CircuitNet-N28 from the [official CircuitNet download page](https://circuitnet.github.io/intro/download.html). The expected inputs are:

- `routability_features_decompressed/{macro_region,cell_density,RUDY,...,congestion/...}`
- placed DEF files under `raw_data/LEF&DEF/decompressed-DEF`
- `graph_features/instance_placement_micron`

Generate the two processed modalities:

```bash
python scripts/preprocess_pixels.py \
  --data-root /path/to/CircuitNet-N28 \
  --output-dir data/processed/pixel

python scripts/preprocess_nets.py \
  --def-root '/path/to/CircuitNet-N28/raw_data/LEF&DEF/decompressed-DEF' \
  --position-dir /path/to/CircuitNet-N28/graph_features/instance_placement_micron \
  --pixel-dir data/processed/pixel \
  --output-dir data/processed/net

python scripts/validate_dataset.py \
  --pixel-dir data/processed/pixel \
  --net-dir data/processed/net
```

The checked-in split contains 705 train, 705 validation, 5,668 seen-test, and 3,164 unseen-test samples. The main paper result uses the unseen test file. Details and tensor schemas are in [docs/data.md](docs/data.md).

## Evaluate a checkpoint

A trained checkpoint can be evaluated with:

```bash
python main.py \
  --config configs/eval_congestion.yaml \
  --mode test \
  --checkpoint /path/to/checkpoint.pth
```

Or keep the data elsewhere:

```bash
python main.py \
  --config configs/eval_congestion.yaml \
  --mode test \
  --checkpoint /path/to/checkpoint.pth \
  --pixel-root /path/to/processed_pixel_data \
  --net-root /path/to/processed_net_data \
  --device cuda:0
```

## Train

```bash
python main.py --config configs/train_congestion.yaml --mode train_test
```

`configs/train_congestion.yaml` is the paper-compatible training configuration. `configs/train_congestion_graph_fixed.yaml` activates GAT feedback between hybrid iterations and should be trained from scratch.

## Repository layout

```text
axialnet/                 model, dataset, solver, and configuration utilities
configs/                  paper-compatible and corrected experiment configs
scripts/                  pixel/net preprocessing and validation
splits/                   exact CircuitNet-N28 cross-design split
docs/                     data preparation guide
tests/                    CPU model and split smoke tests
```

## Verification

```bash
python scripts/run_smoke_tests.py
# or, with the optional test dependency: pytest -q
```

The full 3,164-design evaluation requires the external CircuitNet-N28 data and is expected to be run on a GPU. The unit tests use a small synthetic heterogeneous graph and run on CPU.

For a quick real-data check, use `scripts/infer_one.py` with any sample ID from the split files.

To exercise one complete optimizer/validation/checkpoint/test cycle on that fixture design:

```bash
python main.py --config configs/smoke_real_data.yaml --mode train_test \
  --pixel-root /path/to/processed_pixel_data \
  --net-root /path/to/processed_net_data
```

## License

The code is released under Apache License 2.0. CircuitNet data is not redistributed; obtain it from the upstream project and follow its BSD-3-Clause terms. Third-party notices are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Please cite the ICCAD paper. Machine-readable metadata is provided in [CITATION.cff](CITATION.cff).
