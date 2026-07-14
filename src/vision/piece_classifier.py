"""13-class piece classifier per square patch."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from src.chess_core.fen_utils import CLASS_TO_PIECE, PIECE_CLASSES, PIECE_TO_CLASS
from src.vision.square_diff import MAX_INCREMENTAL_SQUARES, diff_square_indices

NUM_CLASSES = 13
AMBIGUITY_THRESHOLD = 0.65


class SmallCNN(nn.Module):
    """Lightweight CNN for 64x64 square classification."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_classifier(architecture: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build piece classifier model."""
    if architecture == "small_cnn":
        return SmallCNN(num_classes)
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {architecture}")


class PieceClassifier:
    """Wrapper for per-square piece classification."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        architecture: str = "resnet18",
        device: str | None = None,
        ambiguity_threshold: float = AMBIGUITY_THRESHOLD,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ambiguity_threshold = ambiguity_threshold
        self.architecture = architecture
        self.model = build_classifier(architecture).to(self.device)
        self.model.eval()
        self._use_amp = self.device == "cuda"
        if self._use_amp:
            torch.backends.cudnn.benchmark = True

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._mean = mean
        self._std = std

        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                ckpt = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            arch = ckpt.get("architecture", architecture)
            if arch != architecture:
                self.architecture = arch
                self.model = build_classifier(arch).to(self.device)
                self.model.eval()
            self.model.load_state_dict(state)
            if "architecture" in ckpt:
                self.architecture = ckpt["architecture"]
        elif checkpoint_path:
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {checkpoint_path}"
            )

    @property
    def device_label(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"

    def warmup(self, batch_size: int = 64) -> None:
        """Prime CUDA kernels / CPU thread pools before the first real capture."""
        dummy = np.zeros((batch_size, 64, 64, 3), dtype=np.uint8)
        self.predict_squares(dummy)

    def predict_squares(
        self,
        squares: np.ndarray,
        *,
        previous_squares: np.ndarray | None = None,
        previous_result: dict | None = None,
        max_incremental: int = MAX_INCREMENTAL_SQUARES,
    ) -> dict:
        """
        Classify 64 square patches.

        When ``previous_squares`` and ``previous_result`` are supplied, only
        re-classifies squares that changed (typical chess move: 2-4 squares).
        """
        if (
            previous_squares is not None
            and previous_result is not None
            and previous_squares.shape == squares.shape
        ):
            changed = diff_square_indices(previous_squares, squares)
            if not changed:
                result = deepcopy(previous_result)
                result["inference_mode"] = "cached"
                result["squares_updated"] = 0
                return result
            if len(changed) <= max_incremental:
                return self._predict_indices(squares, changed, previous_result)

        tensor = self._preprocess(squares)
        probs = self._forward(tensor)
        return self._result_from_probs(probs, inference_mode="full", squares_updated=64)

    def _predict_indices(
        self,
        squares: np.ndarray,
        indices: list[int],
        previous_result: dict,
    ) -> dict:
        batch = self._preprocess(squares[np.array(indices, dtype=np.int64)])
        partial_probs = self._forward(batch)
        probs = previous_result["probs"].copy()
        for j, idx in enumerate(indices):
            probs[idx] = partial_probs[j]
        return self._result_from_probs(
            probs,
            inference_mode="incremental",
            squares_updated=len(indices),
        )

    def _forward(self, tensor: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            if self._use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            return torch.softmax(logits, dim=1).float().cpu().numpy()

    def _result_from_probs(
        self,
        probs: np.ndarray,
        *,
        inference_mode: str,
        squares_updated: int,
    ) -> dict:
        predictions = probs.argmax(axis=1)
        confidence = probs.max(axis=1)

        ambiguous: list[tuple[int, int]] = []
        class_matrix = np.zeros((8, 8), dtype=np.int64)
        conf_matrix = np.zeros((8, 8), dtype=np.float32)
        board_matrix: list[list[str | None]] = []

        for rank in range(8):
            row: list[str | None] = []
            for file in range(8):
                idx = rank * 8 + file
                cls = int(predictions[idx])
                class_matrix[rank, file] = cls
                conf_matrix[rank, file] = confidence[idx]
                if confidence[idx] < self.ambiguity_threshold:
                    ambiguous.append((rank, file))
                row.append(CLASS_TO_PIECE[cls])
            board_matrix.append(row)

        return {
            "board_matrix": board_matrix,
            "class_matrix": class_matrix,
            "confidence": conf_matrix,
            "ambiguous_squares": ambiguous,
            "probs": probs,
            "inference_mode": inference_mode,
            "squares_updated": squares_updated,
        }

    def predict_squares_legacy(self, squares: np.ndarray) -> dict:
        """Alias kept for callers that do not use incremental mode."""
        return self.predict_squares(squares)

    def _preprocess(self, squares: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(squares[:, :, :, ::-1].copy()).to(self.device, non_blocking=True)
        tensor = rgb.permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        return tensor

    @staticmethod
    def class_to_piece(cls: int) -> str | None:
        return CLASS_TO_PIECE.get(cls)

    @staticmethod
    def piece_to_class(piece: str | None) -> int:
        if piece is None:
            return 0
        return PIECE_TO_CLASS[piece]
