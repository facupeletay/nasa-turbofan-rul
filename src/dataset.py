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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src.etl import load_artifacts

DEFAULT_BATCH_SIZE = 256

WindowArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------
# Construcción de ventanas
# --------------------------------------------------------------------------
def _windows_ending_at(
    values: np.ndarray, ends: np.ndarray, window_size: int
) -> np.ndarray:
    """Ventanas de un único motor que terminan en las filas `ends`.

    Si a una ventana le faltan ciclos previos, se completa a la izquierda repitiendo
    el primer ciclo del motor (edge padding). Un relleno con ceros equivaldría, tras
    el `MinMaxScaler`, a sensores en su mínimo: un estado físico inexistente.

    Args:
        values: `(n_cycles, num_features)`, ordenado por ciclo.
        ends: índices (base 0) del último ciclo de cada ventana.

    Returns:
        `(len(ends), window_size, num_features)`.
    """
    padded = np.pad(values, ((window_size - 1, 0), (0, 0)), mode="edge")
    # Vista (n_cycles, num_features, window_size): la ventana `i` termina en la fila `i`
    view = np.lib.stride_tricks.sliding_window_view(padded, window_size, axis=0)
    return view[ends].transpose(0, 2, 1)


def _build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    select_ends: Callable[[np.ndarray], np.ndarray],
    target_col: str,
) -> WindowArrays:
    """Arma ventanas motor por motor.

    Args:
        select_ends: recibe los ciclos de un motor y devuelve los índices de fila
            donde debe terminar cada una de sus ventanas.

    Returns:
        X `(N, window_size, num_features)` e y `(N,)` en float32, y unit_ids `(N,)`
        en int64, ordenados por motor y ciclo.
    """
    if window_size < 1:
        raise ValueError(f"window_size debe ser >= 1; se recibió {window_size}")

    df = df.sort_values(config.INDEX_COLS, ignore_index=True)
    X_blocks, y_blocks, unit_blocks = [], [], []
    for unit_id, group in df.groupby(config.UNIT_COL, sort=True):
        ends = select_ends(group[config.CYCLE_COL].to_numpy())
        if len(ends) == 0:
            continue
        values = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)
        X_blocks.append(_windows_ending_at(values, ends, window_size))
        y_blocks.append(targets[ends])
        unit_blocks.append(np.full(len(ends), unit_id, dtype=np.int64))

    if not X_blocks:
        raise ValueError("No se generó ninguna ventana")
    X = np.ascontiguousarray(np.concatenate(X_blocks))
    y = np.concatenate(y_blocks)
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Hay NaN o infinitos en las features o en el target")
    return X, y, np.concatenate(unit_blocks)


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
    stride: int = 1,
    target_col: str = config.TARGET_COL,
) -> WindowArrays:
    """Ventanas deslizantes completas de cada motor.

    Un motor con `L >= window_size` ciclos aporta `(L - window_size) // stride + 1`
    ventanas. Con `stride > 1` se anclan al final de la trayectoria, para conservar
    siempre el último ciclo, que es el más informativo sobre la falla. Un motor con
    `L < window_size` aporta una sola ventana, con relleno a la izquierda.

    Returns:
        X `(N, window_size, num_features)` float32, y `(N,)` float32 con el RUL del
        último ciclo de cada ventana, y unit_ids `(N,)` int64.
    """
    if stride < 1:
        raise ValueError(f"stride debe ser >= 1; se recibió {stride}")

    def select_ends(cycles: np.ndarray) -> np.ndarray:
        last = len(cycles) - 1
        first = min(window_size - 1, last)
        return np.arange(last, first - 1, -stride)[::-1]

    return _build_windows(df, feature_cols, window_size, select_ends, target_col)


def build_last_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
    target_col: str = config.TARGET_COL,
) -> WindowArrays:
    """Última ventana de cada motor, para `val_last_cycle` y `test_last_cycle`.

    Si un motor tiene menos de `window_size` ciclos, se completa a la izquierda. No
    ocurre en FD001 con ventana 30 (mínimo 31 ciclos en Test), pero sí en FD002
    (mínimo 21) y FD004 (mínimo 19).

    Returns:
        X `(n_units, window_size, num_features)`, y `(n_units,)` y unit_ids
        `(n_units,)`, con los motores en orden ascendente.
    """
    return _build_windows(
        df,
        feature_cols,
        window_size,
        lambda cycles: np.array([len(cycles) - 1]),
        target_col,
    )


def build_truncated_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = config.WINDOW_SIZE,
    min_cycle: int = config.WINDOW_SIZE,
    target_col: str = config.TARGET_COL,
) -> WindowArrays:
    """Una ventana por cada ciclo `>= min_cycle`, para evaluar `val_truncated`.

    Cubre exactamente las filas que evalúa el baseline XGBoost, sea cual sea
    `window_size`: si la ventana es más larga que la historia disponible, se completa
    a la izquierda en lugar de perder la fila. Las predicciones se agregan con
    `metrics.truncated_metrics`, que es la función que usa el baseline.
    """
    return _build_windows(
        df,
        feature_cols,
        window_size,
        lambda cycles: np.flatnonzero(cycles >= min_cycle),
        target_col,
    )


# --------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------
class RULWindowDataset(Dataset):
    """Expone ventanas y targets como tensores float32.

    Los tensores viven en RAM. `unit_ids` sigue el mismo orden que `X` e `y`, por lo
    que en un loader sin shuffle queda alineado con las predicciones.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, unit_ids: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(f"X debe tener 3 dimensiones; se recibió {X.shape}")
        if y.shape != (len(X),) or unit_ids.shape != (len(X),):
            raise ValueError(
                f"Formas incompatibles: X={X.shape}, y={y.shape}, "
                f"unit_ids={unit_ids.shape}"
            )
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.unit_ids = unit_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve `(X, y)` con formas `(window_size, num_features)` y `()`."""
        return self.X[idx], self.y[idx]


@dataclass(frozen=True)
class RULDataLoaders:
    """Loaders de un subset y los datos necesarios para construir y evaluar modelos.

    `test_last_rul_raw` es el RUL sin recortar, alineado con `test_last`, para
    reportar `test_last_cycle_raw`.
    """

    train: DataLoader
    val_all: DataLoader
    val_last: DataLoader
    val_truncated: DataLoader
    test_last: DataLoader
    test_last_rul_raw: np.ndarray
    num_features: int
    window_size: int
    feature_cols: list[str]


def create_dataloaders(
    subset: str = config.DEFAULT_SUBSET,
    window_size: int = config.WINDOW_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stride: int = 1,
    seed: int = config.RANDOM_SEED,
) -> RULDataLoaders:
    """Construye los DataLoaders a partir de los artefactos del ETL.

    Solo `train` usa shuffle, con un `torch.Generator` fijado en `seed`. Mezclar
    ventanas es válido porque el split ya separó los motores. `stride` también se
    aplica solo a `train`: los conjuntos de evaluación no se submuestrean.
    """
    data = load_artifacts(subset)
    cols = data.feature_cols

    def loader(arrays: WindowArrays, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            RULWindowDataset(*arrays),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
            pin_memory=torch.cuda.is_available(),
        )

    _, test_last_rul_raw, _ = build_last_windows(
        data.test, cols, window_size, target_col=config.TARGET_RAW_COL
    )
    return RULDataLoaders(
        train=loader(
            build_sliding_windows(data.train, cols, window_size, stride), shuffle=True
        ),
        val_all=loader(build_sliding_windows(data.val, cols, window_size)),
        val_last=loader(build_last_windows(data.val, cols, window_size)),
        val_truncated=loader(build_truncated_windows(data.val, cols, window_size)),
        test_last=loader(build_last_windows(data.test, cols, window_size)),
        test_last_rul_raw=test_last_rul_raw,
        num_features=len(cols),
        window_size=window_size,
        feature_cols=cols,
    )
