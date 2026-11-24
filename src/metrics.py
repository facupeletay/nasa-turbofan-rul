"""Métricas de evaluación de RUL, vectorizadas en NumPy.

Convención de signo en todo el proyecto:

    d = RUL_predicho - RUL_real

    d < 0  -> predicción temprana (se interviene antes de lo necesario)
    d > 0  -> predicción tardía  (el motor puede fallar antes de intervenir)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from src.config import SCORE_A_EARLY, SCORE_A_LATE


def _prediction_error(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Valida las entradas y devuelve `d = y_pred - y_true` en float64.

    Acepta vectores `(n,)` o columna `(n, 1)`. Las formas se unifican antes de
    restar: con `(n,)` y `(n, 1)`, NumPy haría broadcasting a `(n, n)` sin avisar.
    """
    arrays = []
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(
                f"{name} debe tener forma (n,) o (n, 1); se recibió {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contiene valores NaN o infinitos")
        arrays.append(arr)

    true, pred = arrays
    if true.shape != pred.shape:
        raise ValueError(
            f"Longitudes distintas: y_true={true.shape[0]}, y_pred={pred.shape[0]}"
        )
    if true.size == 0:
        raise ValueError("y_true e y_pred están vacíos")
    return pred - true


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Raíz del error cuadrático medio, en ciclos de vuelo."""
    d = _prediction_error(y_true, y_pred)
    return float(np.sqrt(np.mean(d**2)))


def nasa_asymmetric_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    a_early: float = SCORE_A_EARLY,
    a_late: float = SCORE_A_LATE,
) -> float:
    """NASA Asymmetric Score (PHM08). Menor es mejor; 0 es perfecto.

        s = sum_i  exp(-d_i / a_early) - 1   si d_i <  0
                   exp( d_i / a_late)  - 1   si d_i >= 0

    Es una suma y no un promedio: solo es comparable entre evaluaciones sobre la
    misma cantidad de motores. La evaluación estándar usa una predicción por motor.

    Args:
        y_true: RUL real, en ciclos.
        y_pred: RUL predicho, en ciclos.
        a_early: constante de decaimiento para predicciones tempranas.
        a_late: constante para predicciones tardías. Con `a_late < a_early`,
            llegar tarde cuesta más que llegar temprano.
    """
    if a_early <= 0 or a_late <= 0:
        raise ValueError("a_early y a_late deben ser positivos")
    d = _prediction_error(y_true, y_pred)
    exponent = np.where(d < 0, -d / a_early, d / a_late)
    return float(np.sum(np.expm1(exponent)))


def truncated_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, unit_ids: ArrayLike
) -> dict[str, float | int]:
    """Métricas esperadas si cada motor se truncara en un ciclo al azar.

    Test se evalúa en un ciclo anterior a la falla, mientras que el último ciclo de
    Val siempre tiene RUL 0. Aquí cada motor puede cortarse en cualquiera de las filas
    recibidas con igual probabilidad. Se calcula la esperanza exacta, sin sorteo: se
    promedia dentro de cada motor y luego se agrega entre motores.

    El llamador elige las filas. Para comparar modelos deben ser las mismas:
    `cycle >= WINDOW_SIZE`.
    """
    true, pred, units = (np.asarray(a).reshape(-1) for a in (y_true, y_pred, unit_ids))
    if not len(true) == len(pred) == len(units):
        raise ValueError(
            f"Longitudes distintas: y_true={len(true)}, y_pred={len(pred)}, "
            f"unit_ids={len(units)}"
        )
    mse_by_unit, score_by_unit = [], []
    for unit in np.unique(units):
        mask = units == unit
        mse_by_unit.append(rmse(true[mask], pred[mask]) ** 2)
        score_by_unit.append(nasa_asymmetric_score(true[mask], pred[mask]) / mask.sum())
    if not mse_by_unit:
        raise ValueError("No hay filas para evaluar")
    return {
        "n": len(mse_by_unit),
        "rmse": float(np.sqrt(np.mean(mse_by_unit))),
        "nasa_score": float(np.sum(score_by_unit)),
    }
