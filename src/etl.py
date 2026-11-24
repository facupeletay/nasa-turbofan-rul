"""ETL de NASA C-MAPSS: Extract (Kaggle), Transform (limpieza y RUL), Load (Parquet).

Reglas anti-leakage que este módulo debe garantizar:
    1. El split Train/Val es por `unit_id`: ningún motor aparece en ambos.
    2. La lista de sensores de varianza cero se decide mirando solo Train.
    3. El `MinMaxScaler` se ajusta solo en Train y se aplica a Val y Test.
    4. Test no participa de ninguna decisión.

Ver `docs/02_process_architecture.md`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config


@dataclass(frozen=True)
class ETLResult:
    """Salida del ETL para un subconjunto de C-MAPSS.

    Los DataFrames tienen las columnas
    `[unit_id, cycle, *feature_cols, rul, rul_raw]`, con las features escaladas.
    """

    subset: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    scaler: MinMaxScaler


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------
def download_dataset() -> Path:
    """Descarga el dataset con `kagglehub` al disco temporal de la máquina.

    Returns:
        Carpeta que contiene los `.txt`. En Kaggle están en la subcarpeta `CMaps/`,
        por lo que hay que ubicarlos con una búsqueda recursiva.
    """
    raise NotImplementedError


def backup_raw(source_dir: Path, raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Copia los `.txt` a `data/raw/` como respaldo inmutable.

    Nunca sobrescribe. Si el archivo ya existe, compara SHA-256 y lanza
    `RuntimeError` ante cualquier diferencia.

    Returns:
        Rutas de los archivos presentes en `raw_dir`.
    """
    raise NotImplementedError


def read_raw_table(path: Path) -> pd.DataFrame:
    """Lee un `train_*.txt` o `test_*.txt` y asigna los 26 encabezados.

    Cada línea termina en dos espacios, así que con `sep=" "` pandas crea dos
    columnas vacías (26 y 27). Usar `sep=r"\\s+"` o descartarlas, y validar que
    queden exactamente 26 columnas sin nulos.
    """
    raise NotImplementedError


def read_test_rul(path: Path) -> pd.Series:
    """Lee `RUL_*.txt`: RUL real al último ciclo de cada motor de Test.

    Returns:
        Serie indexada por `unit_id` (1..n, en el orden del archivo).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def add_train_rul(df: pd.DataFrame, clip: int = config.RUL_CLIP) -> pd.DataFrame:
    """Agrega `rul_raw` y `rul` a trayectorias que llegan hasta la falla.

    rul_raw = max(cycle del motor) - cycle
    rul     = min(rul_raw, clip)
    """
    raise NotImplementedError


def add_test_rul(
    df: pd.DataFrame, final_rul: pd.Series, clip: int = config.RUL_CLIP
) -> pd.DataFrame:
    """Agrega `rul_raw` y `rul` a trayectorias truncadas antes de la falla.

    rul_raw = final_rul[unit_id] + max(cycle del motor) - cycle
    rul     = min(rul_raw, clip)
    """
    raise NotImplementedError


def split_by_unit(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa Train/Val con `GroupShuffleSplit` agrupando por `unit_id`.

    Debe verificar que la intersección de motores entre ambos sea vacía.
    """
    raise NotImplementedError


def select_feature_columns(
    train: pd.DataFrame, tol: float = config.ZERO_VARIANCE_TOL
) -> list[str]:
    """Devuelve los sensores con varianza mayor a `tol`, calculada solo en Train.

    Las condiciones operativas no se incluyen: en FD001 hay un único régimen y solo
    aportan ruido.
    """
    raise NotImplementedError


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    """Ajusta un `MinMaxScaler` a [0, 1] usando exclusivamente Train."""
    raise NotImplementedError


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]
) -> pd.DataFrame:
    """Devuelve una copia de `df` con `feature_cols` escaladas.

    Val y Test pueden quedar levemente fuera de [0, 1]. No se recorta: un valor
    fuera de rango es información legítima sobre el motor.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def save_artifacts(result: ETLResult) -> None:
    """Persiste los tres `.parquet`, el scaler y la metadata del ETL.

    La metadata (`etl_metadata_<subset>.json`) registra `feature_cols`, los
    `unit_id` de cada split, la semilla, `RUL_CLIP` y la fecha de ejecución.
    """
    raise NotImplementedError


def load_artifacts(subset: str = config.DEFAULT_SUBSET) -> ETLResult:
    """Recarga desde disco lo que guardó `save_artifacts`."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run_etl(subset: str = config.DEFAULT_SUBSET) -> ETLResult:
    """Ejecuta el ETL completo para un subconjunto.

    Orden obligatorio:
        1. `ensure_directories` -> `download_dataset` -> `backup_raw`
        2. `read_raw_table` (train, test) y `read_test_rul`
        3. `add_train_rul` y `add_test_rul`
        4. `split_by_unit` sobre train
        5. `select_feature_columns` sobre la partición Train
        6. `fit_scaler` en Train; `apply_scaler` en Train, Val y Test
        7. `save_artifacts`
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de NASA C-MAPSS")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    args = parser.parse_args()
    run_etl(args.subset)


if __name__ == "__main__":
    main()
