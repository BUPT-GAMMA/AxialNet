from pathlib import Path

import torch
from torch_geometric.data import Batch, HeteroData

from axialnet.models import CombinedModel
from axialnet.models.backbones.DualCongestionModel import DualCongestionEncoder
from axialnet.models.heads.dual_congestion_head import DualCongestionHead


ROOT = Path(__file__).resolve().parents[1]


def make_batch(net_offset=0.0):
    data = HeteroData()
    data["grid_l1"].x = torch.linspace(0, 1, 8 * 8 * 4).reshape(8 * 8, 4)
    data["grid_l1"].size = torch.tensor([8, 8])
    data["grid_l1"].y_h = torch.zeros(8 * 8, 1)
    data["grid_l1"].y_v = torch.zeros(8 * 8, 1)
    data["grid_l1"].congestion_label = torch.zeros(8 * 8, 1)
    data["net"].x = torch.arange(16, dtype=torch.float32).reshape(4, 4) + net_offset
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 18, 19, 36, 37, 62, 63]],
        dtype=torch.long,
    )
    data["net", "to", "grid_l1"].edge_index = edge_index
    data["grid_l1", "to", "net"].edge_index = edge_index.flip(0)
    return Batch.from_data_list([data])


def build_model(apply_graph_updates=False, hidden_dim=8):
    return CombinedModel(
        DualCongestionEncoder(
            grid_in_dim=4,
            net_in_dim=4,
            hidden_dim=hidden_dim,
            num_loops=2,
            apply_graph_updates=apply_graph_updates,
        ),
        DualCongestionHead(hidden_channels=16, num_layers=3, dropout=0.0),
    )


def test_cpu_forward_shape():
    torch.manual_seed(42)
    model = build_model().eval()
    with torch.no_grad():
        output = model(make_batch())
    assert output["pred_h"].shape == (64, 1)
    assert output["pred_v"].shape == (64, 1)
    torch.testing.assert_close(output["pred_congestion"], output["pred_h"] + output["pred_v"])


def test_historical_mode_ignores_graph_updates():
    torch.manual_seed(42)
    model = build_model(apply_graph_updates=False).eval()
    with torch.no_grad():
        first = model(make_batch(net_offset=0.0))["pred_congestion"]
        second = model(make_batch(net_offset=100.0))["pred_congestion"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_corrected_mode_uses_graph_updates():
    torch.manual_seed(42)
    model = build_model(apply_graph_updates=True).eval()
    with torch.no_grad():
        first = model(make_batch(net_offset=0.0))["pred_congestion"]
        second = model(make_batch(net_offset=100.0))["pred_congestion"]
    assert not torch.equal(first, second)


def test_state_dict_strict_round_trip():
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
    )
    state_dict = model.state_dict()
    replica = CombinedModel(
        DualCongestionEncoder(hidden_dim=64, num_loops=2, apply_graph_updates=False),
        DualCongestionHead(
            hidden_channels=128,
            feat_channels=128,
            out_channels=1,
            num_layers=3,
            dropout=0.3,
            decode_mode="separate",
        ),
    )
    result = replica.load_state_dict(state_dict, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_canonical_split_counts_and_disjointness():
    split_dir = ROOT / "splits/circuitnet_n28_cross_design"
    expected = {"train": 705, "val": 705, "test_seen": 5668, "test": 3164}
    groups = {}
    for name, count in expected.items():
        values = {line for line in (split_dir / f"{name}.txt").read_text().splitlines() if line}
        assert len(values) == count
        groups[name] = values
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert groups[left].isdisjoint(groups[right])
    assert len(set().union(*groups.values())) == 10242
