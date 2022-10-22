from __future__ import annotations

import pathlib
from typing import Dict, Final, List, Optional, Tuple

import albumentations
import numpy as np
import pydantic
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig
from PIL import Image
from pydantic.dataclasses import dataclass
from torch.nn.modules.loss import _Loss
from torchmetrics.classification.stat_scores import StatScores


class Classifier(pl.LightningModule):
    """Lightning Module for supervised image classfication.
    Attributes:
        encoder (nn.Module): The encoder to extract feature for classification.
        optimizer_cfg (DictConfig): The config for optimizer.
        scheduler_cfg (DictConfig): The config for sheduler.
        criterion (_Loss): The loss used by optimizer.
        num_classes (int): The number of class.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_classes: int,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        scheduler_monitor: str = "train/accuracy",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.optimizer_cfg: Final = optimizer_cfg
        self.scheduler_cfg: Final = scheduler_cfg
        self.scheduler_monitor: Final = scheduler_monitor

        self.criterion: Final[_Loss] = torch.nn.CrossEntropyLoss()
        self.metrics: Final = self._get_metrics(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # type: ignore[no-any-return]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single training step.
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): An input tensor and label.
            batch_index (int): An index of the batch.
        Returns:
            torch.Tensor: A loss tensor.
                If multiple nodes are used for training, return type should be Dict[str, torch.Tensor].
        """
        x, t = batch  # DO NOT need to send GPUs manually.
        logits = self.forward(x)
        loss = self.criterion(logits, t)

        # Logging metrics
        preds: Final = torch.argmax(logits, 1)
        self._log_metrics("train", preds, t, loss.detach(), self.metrics)

        return loss  # type: ignore[no-any-return]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Single validation step.
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The tuple of input tensor and label.
            batch_index (int): The index of batch.
        Returns:
            Dict[str, torch.Tensor]: The dict of log info.
        """
        x, t = batch  # DO NOT need to send GPU manually.
        logits = self.forward(x)
        loss = self.criterion(logits, t)

        # Logging metrics
        preds: Final = torch.argmax(logits, 1)
        self._log_metrics("valid", preds, t, loss.detach(), self.metrics)

        return {"targets": t, "logits": logits.detach()}

    def configure_optimizers(self) -> None:
        """setup optimzier and scheduler."""
        # optimizers are not needed for inference
        pass

    def _get_metrics(self, num_classes: int) -> Dict[str, StatScores]:
        return {
            "train/accuracy": torchmetrics.Accuracy(),
            "valid/accuracy": torchmetrics.Accuracy(),
            "valid/precision_micro": torchmetrics.Precision(average="micro"),
            "valid/precision_macro": torchmetrics.Precision(
                average="macro", num_classes=num_classes
            ),
            "valid/recall_micro": torchmetrics.Recall(average="micro"),
            "valid/recall_macro": torchmetrics.Recall(
                average="macro", num_classes=num_classes
            ),
            "valid/f1": torchmetrics.F1Score(num_classes=num_classes),
        }

    def _log_metrics(
        self,
        stage: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        metrics: Dict[str, StatScores],
    ) -> None:

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True)

        for k, metric in metrics.items():
            if k.startswith(stage):
                value = metric(preds.cpu(), targets.cpu())
                self.log(k, value, prog_bar=True, on_epoch=True)


@dataclass(frozen=True, config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class ClassificationPredictorInputDto:
    input: torch.Tensor
    topk: Optional[int]

    def to_cuda(self) -> ClassificationPredictorInputDto:
        return ClassificationPredictorInputDto(
            input=self.input.to("cuda"), topk=self.topk
        )

    @classmethod
    def from_image_files(
        cls,
        paths: List[pathlib.Path],
        transform: albumentations.Compose,
        topk: int = 3,
    ) -> ClassificationPredictorInputDto:
        return cls.from_images(
            images=[Image.open(path).convert("RGB") for path in paths],
            transform=transform,
            topk=topk,
        )

    @classmethod
    def from_images(
        cls,
        images: List[Image.Image],
        transform: albumentations.Compose,
        topk: int = 3,
    ) -> ClassificationPredictorInputDto:
        tensors: Final = [
            transform(image=np.asarray(image.convert("RGB")))["image"]
            for image in images
        ]
        return cls(input=torch.stack(tensors), topk=topk)


@dataclass(frozen=True)
class ClassificationPredictorOutputDto:
    probabilities: List[Dict[str, float]]

    def to_json(self) -> Dict[str, List[Dict[str, float]]]:
        return {
            "probabilities": self.probabilities,
        }


class ClassificationPredictor:
    def __init__(self, classifier: pl.LightningModule, labels: List[str]) -> None:
        self.classifier: Final = classifier
        self.classifier.eval()
        self.classifier.freeze()
        self.softmax: Final = torch.nn.Softmax(dim=1)
        self.labels: Final = labels

    def predict(
        self, input_dto: ClassificationPredictorInputDto
    ) -> ClassificationPredictorOutputDto:
        k: Final = (
            input_dto.input.size(-1) if input_dto.topk is None else input_dto.topk
        )
        logits: Final = self.classifier(input_dto.input)
        probs: Final = self.softmax(logits)

        predictions: List[Dict[str, float]] = list()
        for prob in probs.split(split_size=1, dim=0):
            value, index = prob.squeeze().topk(k=k)
            _lables: List[str] = [self.labels[i] for i in index.tolist()]
            _probs: List[float] = value.tolist()
            predictions.append(dict(zip(_lables, _probs)))

        return ClassificationPredictorOutputDto(predictions)
