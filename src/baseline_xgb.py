"""Baseline tabular: XGBoost sobre rolling features por motor.

Reglas anti-leakage:
    - Las rolling features son causales (solo pasado y presente) y se calculan
      dentro de cada `unit_id`.
    - El early stopping mira Val; Test se evalúa una única vez.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

from src import config

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "random_state": config.RANDOM_SEED,
}


def build_rolling_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: tuple[int, ...] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Agrega media, desvío y pendiente móviles de cada sensor, por motor.

    Con `F` sensores y `W` ventanas produce `F * (1 + 3 * W)` features: el valor
    instantáneo más tres estadísticos por ventana.

    Returns:
        DataFrame con el mismo índice y cantidad de filas que `df`.
    """
    raise NotImplementedError


def train_xgb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_feature_cols: list[str],
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    """Entrena con early stopping sobre Val. El target es `config.TARGET_COL`."""
    raise NotImplementedError


def evaluate_xgb(
    model: XGBRegressor, df: pd.DataFrame, model_feature_cols: list[str]
) -> dict[str, float]:
    """Calcula RMSE y NASA score sobre las filas recibidas.

    Para comparar con los modelos secuenciales hay que evaluar las mismas filas:
    `cycle >= WINDOW_SIZE` en Val y el último ciclo de cada motor en Test.
    """
    raise NotImplementedError


def run_baseline(subset: str = config.DEFAULT_SUBSET) -> dict[str, dict[str, float]]:
    """Carga artefactos, entrena, evalúa en Val y Test y guarda el modelo.

    Returns:
        `{"val": {"rmse": ..., "nasa_score": ...}, "test": {...}}`
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline XGBoost para RUL")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    args = parser.parse_args()
    print(run_baseline(args.subset))


if __name__ == "__main__":
    main()
