"""Ventanas deslizantes 3D y Dataset/DataLoader de PyTorch.

Contrato de tensores:
    X: float32 `(batch_size, window_size, num_features)`
    y: float32 `(batch_size,)`, RUL en ciclos al último paso de la ventana

Reglas anti-leakage:
    - Una ventana nunca cruza el límite entre dos motores.
    - Una ventana solo contiene ciclos anteriores o iguales al de su etiqueta.
    - Las ventanas se arman después del split por `unit_id`, nunca antes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Materializa todas las ventanas deslizantes (stride 1) de cada motor.

    Un motor con `n` ciclos aporta `max(0, n - window_size + 1)` ventanas. Sirve
    para pruebas y análisis; el entrenamiento usa `TurbofanWindowDataset`.

    Returns:
        X `(n_windows, window_size, num_features)` e y `(n_windows,)`, en float32.
    """
    raise NotImplementedError


def make_last_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae la última ventana de cada motor, para la evaluación oficial en Test.

    Si un motor tiene menos de `window_size` ciclos, se completa a la izquierda
    repitiendo su primer ciclo. No ocurre en FD001 (mínimo 31 ciclos), pero sí en
    FD002 (mínimo 21) y FD004 (mínimo 19).

    Returns:
        X `(n_units, window_size, num_features)`, y `(n_units,)` y unit_ids
        `(n_units,)`.
    """
    raise NotImplementedError


class TurbofanWindowDataset(Dataset):
    """Dataset que arma cada ventana al vuelo a partir de la tabla 2D.

    Guarda una sola matriz `(n_rows, num_features)` y el índice de fin de cada
    ventana válida. `__getitem__` devuelve un slice, sin duplicar datos.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = config.WINDOW_SIZE,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve `(X, y)` con formas `(window_size, num_features)` y `()`."""
        raise NotImplementedError


def build_dataloaders(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int = 256,
    window_size: int = config.WINDOW_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader]:
    """Construye los DataLoaders de Train (con shuffle) y Val (sin shuffle).

    Mezclar ventanas dentro de Train es válido porque el split ya separó los
    motores. El shuffle debe usar un `torch.Generator` con semilla fija.
    """
    raise NotImplementedError
