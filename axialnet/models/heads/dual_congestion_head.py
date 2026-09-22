import torch
import torch.nn as nn
import torch.nn.functional as F

from axialnet.utils.registry import MODELS


class _NodeMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        current_dim = in_channels
        for layer_idx in range(max(0, num_layers - 1)):
            if layer_idx == 0 and current_dim is None:
                layers.append(nn.LazyLinear(hidden_channels))
            else:
                layers.append(nn.Linear(current_dim, hidden_channels))
            current_dim = hidden_channels
        self.hidden = nn.ModuleList(layers)
        if current_dim is None:
            self.out = nn.LazyLinear(out_channels)
        else:
            self.out = nn.Linear(current_dim, out_channels)
        self.dropout = float(dropout)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


@MODELS.register
class DualCongestionHead(nn.Module):
    """
    解码模式:
    - separate: 分别解码 grid_h_feat/grid_v_feat，返回 pred_h/pred_v/pred_congestion
    - unified:  拼接 grid_h_feat/grid_v_feat 后统一解码，返回 pred_congestion
    """

    def __init__(
        self,
        hidden_channels=128,
        feat_channels=None,
        out_channels=1,
        num_layers=2,
        dropout=0.1,
        decode_mode="separate",
    ):
        super().__init__()
        self.decode_mode = str(decode_mode).lower()
        if self.decode_mode not in {"separate", "unified"}:
            raise ValueError("decode_mode must be 'separate' or 'unified'")

        self.h_decoder = _NodeMLP(
            in_channels=None,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.v_decoder = _NodeMLP(
            in_channels=None,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.unified_decoder = _NodeMLP(
            in_channels=None,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )

    @staticmethod
    def _ensure_2d(pred):
        if pred.dim() == 1:
            return pred.view(-1, 1)
        return pred

    def forward(self, x_dict, batch):
        if "grid_h_feat" not in x_dict or "grid_v_feat" not in x_dict or "grid_cnn_feat" not in x_dict:
            raise KeyError("DualCongestionHead expects `grid_h_feat`, `grid_v_feat`, and `grid_cnn_feat` in backbone outputs")

        grid_h_feat = x_dict["grid_h_feat"]
        grid_v_feat = x_dict["grid_v_feat"]
        grid_cnn_feat = x_dict["grid_cnn_feat"]

        grid_h_feat_cated = torch.cat([grid_h_feat, grid_v_feat, grid_cnn_feat], dim=-1)
        grid_v_feat_cated = torch.cat([grid_v_feat, grid_h_feat, grid_cnn_feat], dim=-1)

        if self.decode_mode == "separate":
            # pred_h = self._ensure_2d(self.h_decoder(grid_h_feat))
            # pred_v = self._ensure_2d(self.v_decoder(grid_v_feat))
            pred_h = self._ensure_2d(self.h_decoder(grid_h_feat_cated))
            pred_v = self._ensure_2d(self.v_decoder(grid_v_feat_cated))
            pred_congestion = pred_h + pred_v
            return {
                "pred_h": pred_h,
                "pred_v": pred_v,
                "pred_congestion": pred_congestion,
            }

        fused_feat = torch.cat([grid_h_feat, grid_v_feat], dim=-1)
        pred = self._ensure_2d(self.unified_decoder(fused_feat))
        return {
            "pred_congestion": pred,
        }
