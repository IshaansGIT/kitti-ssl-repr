"""
NT-Xent loss (Normalized Temperature-scaled Cross Entropy), as used in
SimCLR (Chen et al., 2020).

Given a batch of N images, each augmented twice -> 2N embeddings. For each
embedding, its positive is the *other* view of the same image; all other
2N-2 embeddings in the batch serve as negatives. This is why SimCLR benefits
from larger batch sizes (more negatives) — with our small dataset we keep
batch_size modest (see config), which is a real limitation worth naming in
the report rather than hiding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (batch_size, projection_dim), each already L2-normalized.
        z1[i] and z2[i] are the two augmented views of the same source image.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Stack to (2*batch_size, dim): [z1_0..z1_{N-1}, z2_0..z2_{N-1}]
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix (2N, 2N); embeddings are already normalized
        # so a plain matmul gives cosine similarity directly.
        sim = torch.matmul(z, z.T) / self.temperature

        # Mask out self-similarity (diagonal) so a sample is never its own
        # positive/negative candidate.
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # For row i in [0, N), its positive is row i+N, and vice versa.
        positive_idx = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
        ).to(device)

        # Standard cross-entropy: treat this as a 2N-way classification
        # problem where the correct "class" for each row is its positive.
        loss = F.cross_entropy(sim, positive_idx)
        return loss
