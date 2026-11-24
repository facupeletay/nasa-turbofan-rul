"""Ensemble convexo de redes entrenadas, con pesos ajustados en validación.

La predicción combinada es `sum_i w_i * pred_i`, con `w_i >= 0` y `sum_i w_i = 1`,
recortada a `[0, RUL_CLIP]`.

Reglas anti-leakage:
    - Los pesos se ajustan solo con `val_truncated`.
    - Las predicciones de Test se generan recién con los pesos ya fijados.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.dataset import RULDataLoaders, create_dataloaders
from src.metrics import truncated_metrics
from src.models_dl import (
    MODEL_REGISTRY,
    get_device,
    predict,
    read_checkpoint,
    score_predictions,
)

logger = logging.getLogger(__name__)

# Opción de `--optimize` -> métrica de `truncated_metrics` que se minimiza
OBJECTIVES = {"rmse": "rmse", "score": "nasa_score"}


@dataclass(frozen=True)
class Member:
    """Modelo recargado y los loaders armados con su propio `window_size`."""

    name: str
    model: nn.Module
    metadata: dict[str, Any]
    loaders: RULDataLoaders

    @property
    def window_size(self) -> int:
        return self.loaders.window_size


@dataclass(frozen=True)
class EvalSet:
    """Predicciones de todos los miembros sobre las mismas filas.

    `preds` tiene forma `(n, n_miembros)`; `y_true` y `unit_ids`, `(n,)`.
    """

    y_true: np.ndarray
    preds: np.ndarray
    unit_ids: np.ndarray


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def _read_member_files(
    name: str, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Lee pesos y metadata. Si el `.pt` no la incluye, usa su reporte JSON."""
    path = config.MODELS_DIR / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint {path}")
    state_dict, metadata = read_checkpoint(path, device)
    if metadata is None:
        report_path = config.METRICS_DIR / f"{name}.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"{path.name} no incluye metadata y falta su reporte {report_path}"
            )
        logger.info("%s: checkpoint sin metadata; se usa %s", name, report_path.name)
        metadata = json.loads(report_path.read_text(encoding="utf-8"))
    return state_dict, metadata


def load_members(
    names: list[str], subset: str, device: torch.device
) -> list[Member]:
    """Reconstruye cada modelo y verifica que sea compatible con los datos actuales."""
    if len(names) < 2:
        raise ValueError("El ensemble necesita al menos dos modelos")
    if len(set(names)) != len(names):
        raise ValueError(f"Hay modelos repetidos: {names}")

    loaders_by_window: dict[int, RULDataLoaders] = {}
    members = []
    for name in names:
        state_dict, metadata = _read_member_files(name, device)
        if metadata["subset"] != subset:
            raise ValueError(
                f"{name} se entrenó con {metadata['subset']} y no con {subset}"
            )
        # `rul_clip` es también la escala de salida de las redes
        if metadata["rul_clip"] != config.RUL_CLIP:
            raise ValueError(
                f"{name} usa rul_clip={metadata['rul_clip']} y config tiene "
                f"{config.RUL_CLIP}"
            )
        window_size = metadata["params"]["train"]["window_size"]
        if window_size not in loaders_by_window:
            loaders_by_window[window_size] = create_dataloaders(
                subset, window_size=window_size
            )
        loaders = loaders_by_window[window_size]
        if metadata["feature_cols"] != loaders.feature_cols:
            raise ValueError(
                f"{name} se entrenó con otras features que las del ETL actual"
            )

        model = MODEL_REGISTRY[metadata["model"]](**metadata["params"]["model"])
        model.load_state_dict(state_dict)
        members.append(Member(name, model.to(device), metadata, loaders))
    return members


# --------------------------------------------------------------------------
# Predicción
# --------------------------------------------------------------------------
def _last_rows_by_unit(unit_ids: np.ndarray, counts: dict[int, int]) -> np.ndarray:
    """Índices de las últimas `counts[motor]` filas de cada motor."""
    return np.concatenate(
        [np.flatnonzero(unit_ids == unit)[-n:] for unit, n in counts.items()]
    )


def collect_predictions(
    members: list[Member],
    select_loader: Callable[[RULDataLoaders], DataLoader],
    device: torch.device,
) -> EvalSet:
    """Predice con todos los miembros y alinea las filas.

    Con distinto `window_size`, `val_all` tiene más o menos filas por miembro. Como
    las ventanas terminan siempre en el último ciclo del motor, las filas comunes son
    las últimas de cada uno. Los demás conjuntos no dependen de la ventana.
    """
    outputs = []
    for member in members:
        loader = select_loader(member.loaders)
        outputs.append((*predict(member.model, loader, device),
                        loader.dataset.unit_ids))

    counts: dict[int, int] = {}
    for _, _, unit_ids in outputs:
        for unit, n in zip(*np.unique(unit_ids, return_counts=True)):
            counts[int(unit)] = min(int(n), counts.get(int(unit), int(n)))

    trues, preds, units = [], [], []
    for y_true, y_pred, unit_ids in outputs:
        rows = _last_rows_by_unit(unit_ids, counts)
        trues.append(y_true[rows])
        preds.append(y_pred[rows])
        units.append(unit_ids[rows])
    for member, y_true, unit_ids in zip(members, trues, units):
        if not (
            np.array_equal(y_true, trues[0]) and np.array_equal(unit_ids, units[0])
        ):
            raise RuntimeError(f"Las filas de {member.name} no quedaron alineadas")
    return EvalSet(
        y_true=trues[0],
        # float64: en float32 las diferencias finitas del optimizador son ruido
        preds=np.stack(preds, axis=1).astype(np.float64),
        unit_ids=units[0],
    )


def combine(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combinación convexa recortada a `[0, RUL_CLIP]`: `(n, m)` y `(m,)` -> `(n,)`."""
    return np.clip(preds @ weights, 0.0, config.RUL_CLIP)


# --------------------------------------------------------------------------
# Pesos
# --------------------------------------------------------------------------
def optimize_weights(val_truncated: EvalSet, optimize: str = "rmse") -> np.ndarray:
    """Pesos convexos que minimizan la métrica elegida en `val_truncated`.

    Ambos objetivos son convexos en los pesos, por lo que SLSQP no tiene mínimos
    locales. Se compara además con cada modelo por separado y con el promedio simple:
    el resultado nunca es peor que ellos en `val_truncated`.

    Args:
        val_truncated: único conjunto que esta función puede ver.
        optimize: clave de `OBJECTIVES`.
    """
    if optimize not in OBJECTIVES:
        raise ValueError(
            f"optimize inválido: {optimize!r}. Opciones: {sorted(OBJECTIVES)}"
        )
    metric = OBJECTIVES[optimize]

    def objective(weights: np.ndarray) -> float:
        y_pred = combine(val_truncated.preds, weights)
        return truncated_metrics(
            val_truncated.y_true, y_pred, val_truncated.unit_ids
        )[metric]

    n_members = val_truncated.preds.shape[1]
    uniform = np.full(n_members, 1.0 / n_members)
    candidates = [uniform, *np.eye(n_members)]

    # Se normaliza para que `ftol` signifique lo mismo en RMSE y en score
    scale = objective(uniform) or 1.0
    result = minimize(
        lambda w: objective(w) / scale,
        uniform,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_members,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if result.success:
        solution = np.clip(result.x, 0.0, None)
        candidates.append(solution / solution.sum())
    else:
        logger.warning("SLSQP no convergió (%s)", result.message)
    return min(candidates, key=objective)


# --------------------------------------------------------------------------
# Evaluación y orquestación
# --------------------------------------------------------------------------
def evaluate_ensemble(
    members: list[Member], weights: np.ndarray, device: torch.device
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Métricas de cada miembro y del ensemble, con los pesos ya fijados.

    Returns:
        `{modelo: {conjunto: métricas}}`, con la clave `ensemble` al final y los
        mismos conjuntos que `models_dl.evaluate_model`.
    """
    val_truncated = collect_predictions(members, lambda d: d.val_truncated, device)
    val_all = collect_predictions(members, lambda d: d.val_all, device)
    val_last = collect_predictions(members, lambda d: d.val_last, device)
    test_last = collect_predictions(members, lambda d: d.test_last, device)
    test_raw = members[0].loaders.test_last_rul_raw

    def evaluate(weights: np.ndarray) -> dict[str, dict[str, float | int]]:
        test_pred = combine(test_last.preds, weights)
        return {
            "val_all": score_predictions(
                val_all.y_true, combine(val_all.preds, weights)
            ),
            "val_last_cycle": score_predictions(
                val_last.y_true, combine(val_last.preds, weights)
            ),
            "val_truncated": truncated_metrics(
                val_truncated.y_true,
                combine(val_truncated.preds, weights),
                val_truncated.unit_ids,
            ),
            "test_last_cycle": score_predictions(test_last.y_true, test_pred),
            "test_last_cycle_raw": score_predictions(test_raw, test_pred),
        }

    metrics = {
        member.name: evaluate(one_hot)
        for member, one_hot in zip(members, np.eye(len(members)))
    }
    return {**metrics, "ensemble": evaluate(weights)}


def run_ensemble(
    model_names: list[str],
    subset: str = config.DEFAULT_SUBSET,
    optimize: str = "rmse",
) -> dict[str, Any]:
    """Recarga los modelos, ajusta los pesos, evalúa y guarda el reporte.

    Args:
        model_names: nombres base de los checkpoints, p. ej. `lstm_FD001_asym`.
        subset: subconjunto de C-MAPSS con el que se entrenaron todos.
        optimize: clave de `OBJECTIVES`.

    Returns:
        El mismo reporte que se guarda en `metrics/ensemble_<subset>.json`.
    """
    config.ensure_directories()
    device = get_device()
    members = load_members(model_names, subset, device)
    if len({member.window_size for member in members}) > 1:
        logger.warning(
            "Los modelos usan distinto window_size: val_all se evalúa solo en las "
            "filas comunes"
        )

    val_truncated = collect_predictions(members, lambda d: d.val_truncated, device)
    weights = optimize_weights(val_truncated, optimize)

    report = {
        "model": "ensemble",
        "subset": subset,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "optimize": optimize,
        "weights_fitted_on": "val_truncated",
        "rul_clip": config.RUL_CLIP,
        "members": [
            {
                "name": member.name,
                "weight": float(weight),
                "arch": member.metadata["model"],
                "tag": member.metadata.get("tag", ""),
                "params": member.metadata["params"],
            }
            for member, weight in zip(members, weights)
        ],
        "device": device.type,
        "torch_version": str(torch.__version__),
        "metrics": evaluate_ensemble(members, weights, device),
    }
    config.metrics_path("ensemble", subset).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    """Pesos y tabla comparativa, en formato `RMSE / NASA score`."""
    width = max(len(name) for name in report["metrics"]) + 2
    lines = [
        f"ensemble {report['subset']} | pesos ajustados en "
        f"{report['weights_fitted_on']} (objetivo: {report['optimize']})",
        *(f"  {m['name']:<{width}}{m['weight']:.4f}" for m in report["members"]),
        "",
    ]
    for dataset, ensemble in report["metrics"]["ensemble"].items():
        lines.append(f"{dataset} (n={ensemble['n']})")
        lines.append(f"  {'modelo':<{width}}{'RMSE':>9}{'NASA score':>13}")
        for name, by_dataset in report["metrics"].items():
            m = by_dataset[dataset]
            lines.append(
                f"  {name:<{width}}{m['rmse']:>9.2f}{m['nasa_score']:>13.1f}"
            )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ensemble convexo para RUL")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    parser.add_argument(
        "--models", nargs="+", required=True, metavar="NOMBRE",
        help="nombres base de los checkpoints, p. ej. lstm_FD001 lstm_FD001_asym",
    )
    parser.add_argument(
        "--optimize", default="rmse", choices=sorted(OBJECTIVES),
        help="métrica de val_truncated que minimizan los pesos",
    )
    args = parser.parse_args()
    print(format_report(run_ensemble(args.models, args.subset, args.optimize)))


if __name__ == "__main__":
    main()
