"""
ResNet-18 backbone shared by both the SimCLR pretraining stage and the
from-scratch baseline. Keeping this in one place guarantees both models use
an *identical* architecture, which is what makes the fine-tune vs
from-scratch comparison fair — the only difference between them will be the
initial weights, not the network shape.
"""
import torch.nn as nn
from torchvision import models


def build_resnet18_backbone(pretrained_imagenet: bool = True) -> tuple[nn.Module, int]:
    """
    Returns (backbone, feature_dim).

    The backbone is a ResNet-18 with its classification head removed — output
    is the 512-d pooled feature vector, not class logits. `pretrained_imagenet`
    controls the starting weights:
      - True:  torchvision ImageNet weights (recommended; see README for why
               starting from-scratch contrastive pretraining on ~300 unlabeled
               images does not converge to anything meaningful).
      - False: random init (useful for the from-scratch baseline, or for an
               ablation on the pretrained_imagenet choice itself).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
    resnet = models.resnet18(weights=weights)
    feature_dim = resnet.fc.in_features  # 512 for resnet18
    resnet.fc = nn.Identity()  # strip classification head; keep global-pooled features
    return resnet, feature_dim
