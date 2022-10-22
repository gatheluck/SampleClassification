import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final, List, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VisionDataset


@dataclass(frozen=True)
class DatasetStats(ABC):
    num_classes: int = MISSING
    input_size: int = MISSING
    mean: Tuple[float, float, float] = MISSING
    std: Tuple[float, float, float] = MISSING
    labels: List[str] = MISSING


class BaseDataModule(pl.LightningDataModule, ABC):
    """Base class for all 2d image LightningDataModule.

    A datamodule encapsulates the five steps involved in data processing
    in PyTorch:

    - Download / tokenize / process.
    - Clean and (maybe) save to disk.
    - Load inside Dataset.
    - Apply transforms (rotate, tokenize, etc…).
    - Wrap inside a DataLoader.

    For more detail, please check official docs:
    https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
    #what-is-a-datamodule

    Attributes:
        batch_size (int): A size of input image.
        num_workers (int): A number of workers.
        dataset_stats (DatasetStats): A class holds dataset statistics.
        train_dataset: (Dataset): A dataset for training.
        val_dataset: (Dataset): A dataset for validation.

    """

    def __init__(self, batch_size: int, num_workers: int, root: pathlib.Path) -> None:
        super().__init__()
        self.batch_size: Final[int] = batch_size
        self.num_workers: Final[int] = num_workers
        self.dataset_stats: DatasetStats
        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.test_dataset: Dataset

    @abstractmethod
    def prepare_data(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Use this method to do things that might write to disk or that
        need to be done only from a single GPU in distributed settings.
        """
        raise NotImplementedError()

    @abstractmethod
    def setup(self, stage=None, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """There are also data operations you might want to perform on
        every GPU."""
        raise NotImplementedError()

    def train_dataloader(self, *args, **kwargs) -> DataLoader:  # type: ignore[no-untyped-def]
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:  # type: ignore[no-untyped-def]
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, num_samples: int = -1, *args, **kwargs) -> DataLoader:  # type: ignore[no-untyped-def]
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_subset(self, dataset: VisionDataset, num_samples: int) -> VisionDataset:
        if num_samples != -1:
            num_samples = min(num_samples, len(dataset))
            indices = [i for i in range(num_samples)]
            return torch.utils.data.Subset(dataset, indices)
        else:
            return dataset

    @classmethod
    def get_num_classes(cls) -> int:
        return cls.dataset_stats.num_classes

    @classmethod
    def get_input_size(cls) -> int:
        return cls.dataset_stats.input_size

    @classmethod
    def get_mean(cls) -> Tuple[float, float, float]:
        return cls.dataset_stats.mean

    @classmethod
    def get_std(cls) -> Tuple[float, float, float]:
        return cls.dataset_stats.std

    @classmethod
    def get_labels(cls) -> List[str]:
        return cls.dataset_stats.labels
