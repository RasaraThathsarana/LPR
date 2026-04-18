"""Identity adapter implementation."""

import torch
from typing import List
from ..base import Adapter


class IdentityAdapter(Adapter):
    """A no-op adapter that passes features through unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return features
