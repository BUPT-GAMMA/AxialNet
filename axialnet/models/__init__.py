from torch import nn

from axialnet.utils.registry import MODELS

from .backbones import DualCongestionModel  # noqa: F401
from .heads import dual_congestion_head  # noqa: F401


class CombinedModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch):
        return self.head(self.backbone(batch), batch)


def build_model(cfg):
    backbone_cfg = cfg.model.backbone
    head_cfg = cfg.model.head
    backbone = MODELS.get(backbone_cfg.type)(**backbone_cfg.params)
    head = MODELS.get(head_cfg.type)(**head_cfg.params)
    return CombinedModel(backbone, head)
