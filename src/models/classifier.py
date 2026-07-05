"""
Backbone + linear classification head, used for both the fine-tune stage
(backbone initialized from SimCLR pretraining) and the from-scratch baseline
(backbone randomly initialized). Same architecture in both cases — see
backbone.py docstring for why that matters.
"""
import torch
import torch.nn as nn

from src.models.backbone import build_resnet18_backbone


class KittiClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained_imagenet: bool = False):
        """
        pretrained_imagenet:
          - False for the from-scratch baseline (random init end-to-end).
          - Also False when building the fine-tune model — in that case you
            build with pretrained_imagenet=False here and then separately
            load the SimCLR-pretrained backbone weights via
            `load_backbone_state_dict`, rather than mixing ImageNet init
            with SimCLR init.
        """
        super().__init__()
        self.backbone, feature_dim = build_resnet18_backbone(pretrained_imagenet)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)

    def load_backbone_state_dict(self, state_dict: dict) -> None:
        self.backbone.load_state_dict(state_dict)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
