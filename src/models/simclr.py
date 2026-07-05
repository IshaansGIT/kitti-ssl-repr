"""
SimCLR model = backbone encoder + small MLP projection head.

Standard SimCLR design (Chen et al., 2020): the contrastive loss is applied
on the projection head's output (`z`), not on the raw backbone features
(`h`). The projection head is discarded after pretraining — only the
backbone gets carried forward into fine-tuning / from-scratch comparison /
embedding visualization. This matters: the projection head is free to throw
away information that's useful for the pretext task but not for downstream
tasks, so evaluating on its output would understate the backbone's quality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import build_resnet18_backbone


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(
        self,
        pretrained_imagenet: bool = True,
        hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.backbone, feature_dim = build_resnet18_backbone(pretrained_imagenet)
        self.projection_head = ProjectionHead(feature_dim, hidden_dim, projection_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h, z): backbone features and L2-normalized projection."""
        h = self.backbone(x)
        z = self.projection_head(h)
        z = F.normalize(z, dim=1)
        return h, z
