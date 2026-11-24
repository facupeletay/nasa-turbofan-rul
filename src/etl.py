"""ETL de NASA C-MAPSS: Extract (Kaggle), Transform (limpieza y RUL), Load (Parquet).

Reglas anti-leakage que este módulo garantiza:
    1. El split Train/Val es por `unit_id`: ningún motor aparece en ambos.
    2. La lista de sensores de varianza cero se decide mirando solo Train.
    3. El `MinMaxScaler` se ajusta solo en Train y se aplica a Val y Test.
    4. Test no participa de ninguna decisión.

Ver `docs/02_process_architecture.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from src import config

logger = logging.getLogger(__name__)

RAW_FILENAMES = tuple(
    f"{kind}_{subset}.txt"
    for subset in config.SUBSETS
    for kind in ("train", "test", "RUL")
)

MULTI_REGIME_SUBSETS = ("FD002", "FD004")


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
        por lo que se ubican con una búsqueda recursiva.
    """
    # Import local: si `data/raw/` ya está completo, kagglehub no hace falta
    import kagglehub

    root = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    matches = sorted(root.rglob(RAW_FILENAMES[0]))
    if not matches:
        raise FileNotFoundError(f"No se encontró {RAW_FILENAMES[0]} dentro de {root}")
    return matches[0].parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_raw(source_dir: Path, raw_dir: Path = config.RAW_DIR) -> list[Path]:
    """Copia los 12 `.txt` de datos a `data/raw/` como respaldo inmutable.

    Nunca sobrescribe. Si el archivo ya existe, compara SHA-256 y lanza
    `RuntimeError` ante cualquier diferencia. La copia pasa por un archivo temporal
    y se verifica antes de renombrar, de modo que un corte de sesión no deja un
    respaldo a medio escribir.

    Returns:
        Rutas de los archivos presentes en `raw_dir`.
    """
    missing = [name for name in RAW_FILENAMES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos en {source_dir}: {missing}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in RAW_FILENAMES:
        source, target = source_dir / name, raw_dir / name
        checksum = _sha256(source)
        if target.exists():
            if _sha256(target) != checksum:
                raise RuntimeError(
                    f"{target} difiere de la descarga. El respaldo es inmutable: "
                    "revisá ambos archivos antes de continuar."
                )
        else:
            tmp = target.with_name(f"{name}.tmp")
            shutil.copyfile(source, tmp)
            if _sha256(tmp) != checksum:
                raise RuntimeError(f"La copia de {name} a {raw_dir} quedó corrupta")
            os.replace(tmp, target)
        paths.append(target)
    return paths


def extract(raw_dir: Path = config.RAW_DIR) -> None:
    """Deja `data/raw/` completo. No descarga nada si ya lo está."""
    if all((raw_dir / name).is_file() for name in RAW_FILENAMES):
        logger.info("Datos crudos ya presentes en %s; se omite la descarga", raw_dir)
        return
    source_dir = download_dataset()
    backup_raw(source_dir, raw_dir)
    logger.info("Datos crudos respaldados en %s", raw_dir)


def read_raw_table(path: Path) -> pd.DataFrame:
    """Lee un `train_*.txt` o `test_*.txt` y asigna los 26 encabezados.

    Cada línea termina en dos espacios: con `sep=" "` pandas crearía dos columnas
    vacías (26 y 27). Con `sep=r"\\s+"` no aparecen. Valida 26 columnas, ausencia de
    nulos y ciclos consecutivos desde 1 en cada motor.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(config.RAW_COLS):
        raise ValueError(
            f"{path.name}: se esperaban {len(config.RAW_COLS)} columnas y hay "
            f"{df.shape[1]}"
        )
    df.columns = config.RAW_COLS
    if df.isna().any(axis=None):
        raise ValueError(f"{path.name}: contiene valores nulos")

    df = df.sort_values(config.INDEX_COLS, ignore_index=True)
    expected_cycle = df.groupby(config.UNIT_COL).cumcount() + 1
    if not (df[config.CYCLE_COL] == expected_cycle).all():
        raise ValueError(f"{path.name}: hay motores con ciclos no consecutivos")
    return df


def read_test_rul(path: Path) -> pd.Series:
    """Lee `RUL_*.txt`: RUL real al último ciclo de cada motor de Test.

    Returns:
        Serie indexada por `unit_id` (1..n, en el orden del archivo).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path.name}: se esperaba 1 columna y hay {df.shape[1]}")
    rul = df[0]
    if rul.isna().any() or (rul < 0).any():
        raise ValueError(f"{path.name}: contiene RUL nulos o negativos")
    rul.index = pd.RangeIndex(1, len(rul) + 1, name=config.UNIT_COL)
    return rul.rename(config.TARGET_RAW_COL)


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def _add_rul_columns(df: pd.DataFrame, rul_raw: pd.Series, clip: int) -> pd.DataFrame:
    return df.assign(
        **{config.TARGET_COL: rul_raw.clip(upper=clip), config.TARGET_RAW_COL: rul_raw}
    )


def _cycles_to_last_record(df: pd.DataFrame) -> pd.Series:
    cycles = df[config.CYCLE_COL]
    return cycles.groupby(df[config.UNIT_COL]).transform("max") - cycles


def add_train_rul(df: pd.DataFrame, clip: int = config.RUL_CLIP) -> pd.DataFrame:
    """Agrega `rul_raw` y `rul` a trayectorias que llegan hasta la falla.

    rul_raw = max(cycle del motor) - cycle
    rul     = min(rul_raw, clip)
    """
    return _add_rul_columns(df, _cycles_to_last_record(df), clip)


def add_test_rul(
    df: pd.DataFrame, final_rul: pd.Series, clip: int = config.RUL_CLIP
) -> pd.DataFrame:
    """Agrega `rul_raw` y `rul` a trayectorias truncadas antes de la falla.

    rul_raw = final_rul[unit_id] + max(cycle del motor) - cycle
    rul     = min(rul_raw, clip)
    """
    if set(df[config.UNIT_COL]) != set(final_rul.index):
        raise ValueError("Los motores de Test no coinciden con los del archivo de RUL")
    rul_raw = df[config.UNIT_COL].map(final_rul) + _cycles_to_last_record(df)
    return _add_rul_columns(df, rul_raw, clip)


def split_by_unit(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa Train/Val con `GroupShuffleSplit` agrupando por `unit_id`.

    Verifica que la intersección de motores entre ambos sea vacía.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[config.UNIT_COL]))
    train = df.iloc[train_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)

    shared = set(train[config.UNIT_COL]) & set(val[config.UNIT_COL])
    if shared:
        raise RuntimeError(f"Data leakage: motores en Train y Val: {sorted(shared)}")
    return train, val


def select_feature_columns(
    train: pd.DataFrame, tol: float = config.ZERO_VARIANCE_TOL
) -> list[str]:
    """Devuelve los sensores con varianza mayor a `tol`, calculada solo en Train.

    Las condiciones operativas no se incluyen: en FD001 hay un único régimen y solo
    aportan ruido.
    """
    variance = train[config.SENSOR_COLS].var()
    feature_cols = [col for col in config.SENSOR_COLS if variance[col] > tol]
    if not feature_cols:
        raise ValueError("Ningún sensor supera la tolerancia de varianza")
    return feature_cols


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    """Ajusta un `MinMaxScaler` a [0, 1] usando exclusivamente Train."""
    return MinMaxScaler().fit(train[feature_cols])


def apply_scaler(
    df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list[str]
) -> pd.DataFrame:
    """Devuelve `df` con `feature_cols` escaladas y las columnas del contrato.

    Val y Test pueden quedar levemente fuera de [0, 1]. No se recorta: un valor
    fuera de rango es información legítima sobre el motor.
    """
    scaled = pd.DataFrame(
        scaler.transform(df[feature_cols]), columns=feature_cols, index=df.index
    )
    targets = df[[config.TARGET_COL, config.TARGET_RAW_COL]]
    return pd.concat([df[config.INDEX_COLS], scaled, targets], axis=1)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def save_artifacts(result: ETLResult) -> None:
    """Persiste los tres `.parquet`, el scaler y la metadata del ETL.

    La metadata (`etl_metadata_<subset>.json`) registra `feature_cols`, los
    `unit_id` de cada split, la semilla, `RUL_CLIP` y la fecha de ejecución. Se borra
    al empezar y se escribe al final: si existe, los demás artefactos son completos.
    """
    config.ensure_directories()
    metadata_path = config.metadata_path(result.subset)
    metadata_path.unlink(missing_ok=True)

    splits = {"train": result.train, "val": result.val, "test": result.test}
    for split, df in splits.items():
        df.to_parquet(config.processed_path(split, result.subset), index=False)
    joblib.dump(result.scaler, config.scaler_path(result.subset))

    metadata = {
        "subset": result.subset,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rul_clip": config.RUL_CLIP,
        "val_size": config.VAL_SIZE,
        "random_seed": config.RANDOM_SEED,
        "zero_variance_tol": config.ZERO_VARIANCE_TOL,
        "feature_cols": result.feature_cols,
        "rows": {split: len(df) for split, df in splits.items()},
        "units": {
            split: df[config.UNIT_COL].unique().tolist() for split, df in splits.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_artifacts(subset: str = config.DEFAULT_SUBSET) -> ETLResult:
    """Recarga desde disco lo que guardó `save_artifacts`.

    Lanza `ValueError` si los artefactos se generaron con parámetros distintos de los
    de `config`: entrenar con ellos mezclaría dos configuraciones sin aviso.
    """
    metadata_path = config.metadata_path(subset)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No hay artefactos de {subset} en {config.PROCESSED_DIR}. "
            f"Ejecutá primero el ETL: python -m src.etl --subset {subset}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    current = {
        "rul_clip": config.RUL_CLIP,
        "val_size": config.VAL_SIZE,
        "random_seed": config.RANDOM_SEED,
        "zero_variance_tol": config.ZERO_VARIANCE_TOL,
    }
    stale = {k: (metadata[k], v) for k, v in current.items() if metadata[k] != v}
    if stale:
        raise ValueError(
            f"Artefactos de {subset} desactualizados (guardado, actual): {stale}. "
            "Volvé a ejecutar el ETL."
        )

    return ETLResult(
        subset=subset,
        train=pd.read_parquet(config.processed_path("train", subset)),
        val=pd.read_parquet(config.processed_path("val", subset)),
        test=pd.read_parquet(config.processed_path("test", subset)),
        feature_cols=metadata["feature_cols"],
        scaler=joblib.load(config.scaler_path(subset)),
    )


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run_etl(subset: str = config.DEFAULT_SUBSET) -> ETLResult:
    """Ejecuta el ETL completo para un subconjunto.

    Orden obligatorio:
        1. `ensure_directories` -> `extract` (descarga y respaldo)
        2. `read_raw_table` (train, test) y `read_test_rul`
        3. `add_train_rul` y `add_test_rul`
        4. `split_by_unit` sobre train
        5. `select_feature_columns` sobre la partición Train
        6. `fit_scaler` en Train; `apply_scaler` en Train, Val y Test
        7. `save_artifacts`
    """
    config.validate_subset(subset)
    if subset in MULTI_REGIME_SUBSETS:
        logger.warning(
            "%s tiene seis regímenes operativos y este ETL no normaliza por régimen: "
            "la condición de vuelo tapará la degradación",
            subset,
        )

    config.ensure_directories()
    extract()

    full_train = add_train_rul(read_raw_table(config.raw_path("train", subset)))
    test = add_test_rul(
        read_raw_table(config.raw_path("test", subset)),
        read_test_rul(config.raw_path("RUL", subset)),
    )

    train, val = split_by_unit(full_train)
    feature_cols = select_feature_columns(train)
    scaler = fit_scaler(train, feature_cols)

    result = ETLResult(
        subset=subset,
        train=apply_scaler(train, scaler, feature_cols),
        val=apply_scaler(val, scaler, feature_cols),
        test=apply_scaler(test, scaler, feature_cols),
        feature_cols=feature_cols,
        scaler=scaler,
    )
    save_artifacts(result)
    logger.info(
        "%s: %d features | filas train=%d val=%d test=%d",
        subset, len(feature_cols), len(train), len(val), len(test),
    )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ETL de NASA C-MAPSS")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    args = parser.parse_args()
    run_etl(args.subset)


if __name__ == "__main__":
    main()
