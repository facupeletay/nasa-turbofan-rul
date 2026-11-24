"""Modelos secuenciales en PyTorch (LSTM y CNN 1D) y loop de entrenamiento.

Contrato común de los modelos:
    entrada: float32 `(batch_size, window_size, num_features)`
    salida:  float32 `(batch_size,)`, RUL en ciclos

Cada modelo resuelve internamente cualquier reordenamiento de ejes, de modo que
el DataLoader y el loop de entrenamiento no dependen de la arquitectura.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparámetros del entrenamiento."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    grad_clip: float = 1.0
    seed: int = config.RANDOM_SEED


@dataclass(frozen=True)
class TrainHistory:
    """Evolución por época y punto del mejor checkpoint."""

    train_loss: list[float]
    val_rmse: list[float]
    val_score: list[float]
    best_epoch: int


# --------------------------------------------------------------------------
# Arquitecturas
# --------------------------------------------------------------------------
class TurbofanLSTM(nn.Module):
    """LSTM apilada; una cabeza densa lee el último estado oculto.

    (B, W, F) -> LSTM -> (B, W, H) -> último paso -> (B, H) -> MLP -> (B,)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TurbofanCNN1D(nn.Module):
    """Convoluciones temporales y pooling global, con cabeza densa.

    (B, W, F) -> permute -> (B, F, W) -> Conv1d x N -> (B, C, W) -> pool -> (B, C)
    -> MLP -> (B,)
    """

    def __init__(
        self,
        num_features: int,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": TurbofanLSTM,
    "cnn1d": TurbofanCNN1D,
}


# --------------------------------------------------------------------------
# Entrenamiento e inferencia
# --------------------------------------------------------------------------
def seed_everything(seed: int = config.RANDOM_SEED) -> None:
    """Fija las semillas de `random`, NumPy y PyTorch (CPU y CUDA)."""
    raise NotImplementedError


def get_device() -> torch.device:
    """Devuelve `cuda` si está disponible y `cpu` en caso contrario."""
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Ejecuta una época y devuelve la pérdida media por muestra."""
    raise NotImplementedError


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Inferencia en modo `eval` y sin gradientes.

    Returns:
        `(y_true, y_pred)`, ambos `(n,)`, en ciclos.
    """
    raise NotImplementedError


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_config: TrainConfig,
    checkpoint_path: Path,
) -> TrainHistory:
    """Entrena con early stopping por RMSE de Val.

    Guarda el `state_dict` en cada mejora (Colab puede cortar la sesión) y al
    terminar deja cargados en `model` los mejores pesos.
    """
    raise NotImplementedError


def run_training(
    model_name: str,
    subset: str = config.DEFAULT_SUBSET,
    train_config: TrainConfig = TrainConfig(),
) -> dict[str, dict[str, float]]:
    """Carga artefactos, entrena y evalúa en Val y en la última ventana de Test.

    Returns:
        `{"val": {"rmse": ..., "nasa_score": ...}, "test": {...}}`
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelos secuenciales para RUL")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    args = parser.parse_args()
    print(run_training(args.model, args.subset))


if __name__ == "__main__":
    main()
