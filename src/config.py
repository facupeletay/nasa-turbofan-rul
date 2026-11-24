"""Configuración central: rutas dinámicas (Colab Drive vs. local) y constantes.

Importar este módulo no tiene efectos secundarios sobre el disco. Las carpetas
se crean solo al llamar a `ensure_directories()`.

Resolución de la raíz de almacenamiento (`STORAGE_ROOT`), por prioridad:
    1. Variable de entorno `NASA_RUL_ROOT`, si está definida.
    2. `/content/drive/MyDrive/nasa_rul`, si corre en Google Colab.
    3. La raíz del repositorio, en cualquier otro caso.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Entorno
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_MY_DRIVE = DRIVE_MOUNT_POINT / "MyDrive"
DRIVE_STORAGE_ROOT = DRIVE_MY_DRIVE / "nasa_rul"

STORAGE_ROOT_ENV_VAR = "NASA_RUL_ROOT"


def is_colab() -> bool:
    """Indica si el proceso corre en Google Colab.

    `google.colab` solo está en `sys.modules` dentro del kernel del notebook. La
    variable de entorno cubre los subprocesos (`!python -m src.etl`).
    """
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def resolve_storage_root() -> Path:
    """Devuelve la raíz donde persisten datos y modelos."""
    override = os.environ.get(STORAGE_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    if is_colab():
        return DRIVE_STORAGE_ROOT
    return PROJECT_ROOT


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
IN_COLAB = is_colab()
STORAGE_ROOT = resolve_storage_root()

DATA_DIR = STORAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = STORAGE_ROOT / "models"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
KAGGLE_DATASET = "behrad3d/nasa-cmaps"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_SUBSET = "FD001"

UNIT_COL = "unit_id"
CYCLE_COL = "cycle"
INDEX_COLS = [UNIT_COL, CYCLE_COL]
OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
RAW_COLS = INDEX_COLS + OP_SETTING_COLS + SENSOR_COLS  # 26 columnas

TARGET_COL = "rul"          # RUL lineal a tramos, con techo en RUL_CLIP
TARGET_RAW_COL = "rul_raw"  # RUL lineal sin techo, solo para auditoría y reporte

SENSOR_NAMES = {
    "s_1": "T2", "s_2": "T24", "s_3": "T30", "s_4": "T50", "s_5": "P2",
    "s_6": "P15", "s_7": "P30", "s_8": "Nf", "s_9": "Nc", "s_10": "epr",
    "s_11": "Ps30", "s_12": "phi", "s_13": "NRf", "s_14": "NRc", "s_15": "BPR",
    "s_16": "farB", "s_17": "htBleed", "s_18": "Nf_dmd", "s_19": "PCNfR_dmd",
    "s_20": "W31", "s_21": "W32",
}

# --------------------------------------------------------------------------
# Hiperparámetros del problema
# --------------------------------------------------------------------------
RUL_CLIP = 125
WINDOW_SIZE = 30
VAL_SIZE = 0.2
RANDOM_SEED = 42

# Tolerancia y no igualdad a cero: s_5 y s_16 son constantes en FD001, pero su
# varianza en punto flotante da ~1e-29 y ~1e-35. La menor varianza real entre los
# sensores que sí varían es la de s_6 (~2e-6).
ZERO_VARIANCE_TOL = 1e-10

# Ventanas (en ciclos) de las rolling features del baseline tabular
ROLLING_WINDOWS = (5, 10, 20, 30)

# Constantes de decaimiento del NASA Asymmetric Score
SCORE_A_EARLY = 13.0  # d < 0: predicción temprana
SCORE_A_LATE = 10.0   # d >= 0: predicción tardía


# --------------------------------------------------------------------------
# Constructores de rutas
# --------------------------------------------------------------------------
def validate_subset(subset: str) -> str:
    """Verifica que `subset` sea un subconjunto válido de C-MAPSS."""
    if subset not in SUBSETS:
        raise ValueError(f"Subset desconocido: {subset!r}. Opciones: {SUBSETS}")
    return subset


def raw_path(kind: str, subset: str = DEFAULT_SUBSET) -> Path:
    """Ruta del `.txt` crudo. `kind` es `train`, `test` o `RUL`."""
    if kind not in ("train", "test", "RUL"):
        raise ValueError(f"kind inválido: {kind!r}. Opciones: train, test, RUL")
    return RAW_DIR / f"{kind}_{validate_subset(subset)}.txt"


def processed_path(split: str, subset: str = DEFAULT_SUBSET) -> Path:
    """Ruta del `.parquet` procesado. `split` es `train`, `val` o `test`."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split inválido: {split!r}. Opciones: train, val, test")
    return PROCESSED_DIR / f"{split}_{validate_subset(subset)}.parquet"


def scaler_path(subset: str = DEFAULT_SUBSET) -> Path:
    """Ruta del `MinMaxScaler` ajustado en Train."""
    return MODELS_DIR / f"scaler_{validate_subset(subset)}.joblib"


def metadata_path(subset: str = DEFAULT_SUBSET) -> Path:
    """Ruta de la metadata del ETL (features, unidades por split, semilla)."""
    return PROCESSED_DIR / f"etl_metadata_{validate_subset(subset)}.json"


def model_path(name: str, subset: str = DEFAULT_SUBSET, ext: str = "pt") -> Path:
    """Ruta de un modelo entrenado, p. ej. `models/lstm_FD001.pt`."""
    return MODELS_DIR / f"{name}_{validate_subset(subset)}.{ext}"


def ensure_directories() -> None:
    """Crea las carpetas de almacenamiento si no existen.

    En Colab exige que Drive ya esté montado. Crear `/content/drive/MyDrive` sin
    montar escribiría en el disco efímero de la VM y haría fallar el montaje
    posterior.
    """
    uses_drive = STORAGE_ROOT.is_relative_to(DRIVE_MOUNT_POINT)
    if uses_drive and not DRIVE_MY_DRIVE.is_dir():
        raise RuntimeError(
            "Google Drive no está montado. Ejecutá primero: "
            "from google.colab import drive; drive.mount('/content/drive')"
        )
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Resumen legible de la configuración activa."""
    return "\n".join([
        f"Entorno      : {'Google Colab' if IN_COLAB else 'Local'}",
        f"Código       : {PROJECT_ROOT}",
        f"Storage      : {STORAGE_ROOT}",
        f"Datos crudos : {RAW_DIR}",
        f"Procesados   : {PROCESSED_DIR}",
        f"Modelos      : {MODELS_DIR}",
        f"RUL_CLIP={RUL_CLIP}  WINDOW_SIZE={WINDOW_SIZE}  "
        f"VAL_SIZE={VAL_SIZE}  SEED={RANDOM_SEED}",
    ])


if __name__ == "__main__":
    print(describe())
