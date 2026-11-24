"""Baseline tabular: XGBoost sobre rolling features por motor.

Reglas anti-leakage:
    - Las rolling features son causales (solo pasado y presente) y se calculan
      dentro de cada `unit_id`.
    - `unit_id`, `cycle` y las columnas de RUL nunca son features.
    - El early stopping mira Val; Test solo se evalúa.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config
from src.etl import load_artifacts
from src.metrics import nasa_asymmetric_score, rmse, truncated_metrics

logger = logging.getLogger(__name__)

MODEL_NAME = "xgb_baseline"
TOP_FEATURES = 15

NON_FEATURE_COLS = [*config.INDEX_COLS, config.TARGET_COL, config.TARGET_RAW_COL]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_estimators": 2000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "early_stopping_rounds": 50,
    "random_state": config.RANDOM_SEED,
}


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _validate_windows(windows: Iterable[int]) -> tuple[int, ...]:
    """Devuelve las ventanas ordenadas y sin repetir.

    Exige `w >= 2`: con un solo ciclo la media repite el sensor y el desvío es 0.
    """
    result = tuple(sorted(set(windows)))
    if not result or result[0] < 2:
        raise ValueError(f"Las ventanas deben ser enteros >= 2; se recibió {result}")
    return result


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: Iterable[int] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Agrega media y desvío estándar móviles de cada sensor, por motor.

    Las ventanas son causales: la fila del ciclo `t` usa como mucho los ciclos
    `[t - w + 1, t]` de su propio motor. En los primeros ciclos se usan los
    disponibles, y el desvío de una única observación se define como 0.

    Returns:
        Copia de `df` ordenada por (`unit_id`, `cycle`), con índice reiniciado y las
        columnas nuevas `<sensor>_mean_<w>` y `<sensor>_std_<w>`.
    """
    out = df.sort_values(config.INDEX_COLS, ignore_index=True)
    grouped = out.groupby(config.UNIT_COL, sort=False)[feature_cols]

    blocks = [out]
    for w in _validate_windows(windows):
        rolling = grouped.rolling(w, min_periods=1)
        stats = {"mean": rolling.mean(), "std": rolling.std().fillna(0.0)}
        for name, block in stats.items():
            block = block.reset_index(level=config.UNIT_COL, drop=True)
            blocks.append(block.add_suffix(f"_{name}_{w}"))
    return pd.concat(blocks, axis=1)


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de entrada del modelo: todas menos identificadores y targets."""
    return [col for col in df.columns if col not in NON_FEATURE_COLS]


def _check_features(df: pd.DataFrame, model_feature_cols: list[str], name: str) -> None:
    """XGBoost acepta NaN como dato faltante: sin este control pasarían sin aviso."""
    if not np.isfinite(df[model_feature_cols].to_numpy()).all():
        raise ValueError(f"{name}: hay NaN o infinitos en las features")


# --------------------------------------------------------------------------
# Entrenamiento y evaluación
# --------------------------------------------------------------------------
def train_xgb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_feature_cols: list[str],
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    """Entrena con early stopping sobre Val. El target es `config.TARGET_COL`."""
    model = XGBRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(
        train[model_feature_cols],
        train[config.TARGET_COL],
        eval_set=[(val[model_feature_cols], val[config.TARGET_COL])],
        verbose=False,
    )
    return model


def predict_rul(
    model: XGBRegressor, df: pd.DataFrame, model_feature_cols: list[str]
) -> np.ndarray:
    """Predice en ciclos y recorta a `[0, RUL_CLIP]`.

    Con early stopping, `predict` usa solo los árboles hasta la mejor iteración.
    """
    return np.clip(model.predict(df[model_feature_cols]), 0.0, config.RUL_CLIP)


def last_cycle_mask(df: pd.DataFrame) -> np.ndarray:
    """Máscara booleana con el último ciclo registrado de cada motor."""
    cycles = df[config.CYCLE_COL]
    return (cycles == cycles.groupby(df[config.UNIT_COL]).transform("max")).to_numpy()


def _score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_asymmetric_score(y_true, y_pred),
    }


def evaluate_xgb(
    model: XGBRegressor,
    val: pd.DataFrame,
    test: pd.DataFrame,
    model_feature_cols: list[str],
) -> dict[str, dict[str, float | int]]:
    """Calcula RMSE y NASA score según el protocolo de evaluación.

    El NASA score es una suma: solo es comparable entre filas con el mismo `n`.

    Returns:
        Un diccionario por conjunto:
        - `val_all`: todos los ciclos de Val.
        - `val_last_cycle`: último ciclo de cada motor de Val. El RUL real siempre es
          0, así que solo mide el error en el instante de la falla.
        - `val_truncated`: esperanza con truncamiento al azar, análoga a Test.
        - `test_last_cycle`: métrica oficial, contra el RUL recortado.
        - `test_last_cycle_raw`: igual, contra el RUL sin recortar.
    """
    val_pred = predict_rul(model, val, model_feature_cols)
    val_last = last_cycle_mask(val)
    val_eligible = (val[config.CYCLE_COL] >= config.WINDOW_SIZE).to_numpy()

    test_last = test[last_cycle_mask(test)]
    test_pred = predict_rul(model, test_last, model_feature_cols)

    return {
        "val_all": _score(val[config.TARGET_COL], val_pred),
        "val_last_cycle": _score(
            val.loc[val_last, config.TARGET_COL], val_pred[val_last]
        ),
        "val_truncated": truncated_metrics(
            val.loc[val_eligible, config.TARGET_COL],
            val_pred[val_eligible],
            val.loc[val_eligible, config.UNIT_COL],
        ),
        "test_last_cycle": _score(test_last[config.TARGET_COL], test_pred),
        "test_last_cycle_raw": _score(test_last[config.TARGET_RAW_COL], test_pred),
    }


def top_features_by_gain(
    model: XGBRegressor, top_n: int = TOP_FEATURES
) -> list[dict[str, float | str]]:
    """Las `top_n` features con mayor ganancia media por división."""
    gain = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(gain.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {
            "feature": name,
            "sensor": config.SENSOR_NAMES["_".join(name.split("_")[:2])],
            "gain": float(value),
        }
        for name, value in ranked
    ]


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run_baseline(
    subset: str = config.DEFAULT_SUBSET,
    params: dict[str, Any] | None = None,
    windows: Iterable[int] = config.ROLLING_WINDOWS,
) -> dict[str, Any]:
    """Carga artefactos, entrena, evalúa y guarda el modelo y el reporte.

    Args:
        subset: subconjunto de C-MAPSS con el ETL ya ejecutado.
        params: hiperparámetros que reemplazan a los de `DEFAULT_PARAMS`.
        windows: ventanas de las rolling features, en ciclos.

    Returns:
        El mismo reporte que se guarda en `metrics/xgb_baseline_<subset>.json`.
    """
    windows = _validate_windows(windows)
    data = load_artifacts(subset)
    train, val, test = (
        add_rolling_features(df, data.feature_cols, windows)
        for df in (data.train, data.val, data.test)
    )

    model_feature_cols = get_model_feature_columns(train)
    for name, df in (("train", train), ("val", val), ("test", test)):
        _check_features(df, model_feature_cols, name)

    model = train_xgb(train, val, model_feature_cols, params)
    if model.best_iteration == model.n_estimators - 1:
        logger.warning(
            "El early stopping no actuó: la mejor iteración es la última (%d). "
            "Conviene aumentar n_estimators",
            model.n_estimators,
        )

    report = {
        "model": MODEL_NAME,
        "subset": subset,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {**DEFAULT_PARAMS, **(params or {})},
        "windows": list(windows),
        "rul_clip": config.RUL_CLIP,
        "n_features": len(model_feature_cols),
        "best_iteration": int(model.best_iteration),
        "metrics": evaluate_xgb(model, val, test, model_feature_cols),
        "top_features_gain": top_features_by_gain(model),
    }

    config.ensure_directories()
    model.save_model(config.model_path(MODEL_NAME, subset, ext="json"))
    config.metrics_path(MODEL_NAME, subset).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    """Resumen legible del reporte."""
    lines = [
        f"{report['model']} {report['subset']} | {report['n_features']} features | "
        f"mejor iteración: {report['best_iteration']}",
        f"{'conjunto':<22}{'n':>7}{'RMSE':>9}{'NASA score':>13}",
    ]
    for name, m in report["metrics"].items():
        lines.append(f"{name:<22}{m['n']:>7}{m['rmse']:>9.2f}{m['nasa_score']:>13.1f}")
    lines.append(f"Top {len(report['top_features_gain'])} features por gain:")
    for item in report["top_features_gain"]:
        lines.append(f"  {item['feature']:<14}{item['sensor']:<9}{item['gain']:>12.1f}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Baseline XGBoost para RUL")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    parser.add_argument(
        "--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"]
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"]
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=list(config.ROLLING_WINDOWS),
        help="ventanas de las rolling features, en ciclos",
    )
    args = parser.parse_args()

    report = run_baseline(
        args.subset,
        params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        windows=args.windows,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
