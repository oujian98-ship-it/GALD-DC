import torch
from torch.utils.data import Dataset
from typing import Tuple


class LatentDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.size(0) == labels.size(
            0
        ), f"Feature count {features.size(0)} must match label count {labels.size(0)}"

        self.features = features
        self.labels = labels
        self.num_samples = features.size(0)
        self.feature_dim = features.size(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    def get_stats(self) -> dict:
        memory_mb = (
            self.features.element_size() * self.features.nelement()
            + self.labels.element_size() * self.labels.nelement()
        ) / (1024**2)

        return {
            "num_samples": self.num_samples,
            "feature_dim": self.feature_dim,
            "memory_mb": memory_mb,
            "dtype": str(self.features.dtype),
        }
