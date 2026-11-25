"""Gráficos de evaluación a partir de los reportes y modelos ya entrenados.

Genera en `reports/figures/`:
    - `model_comparison_<subset>.png`: RMSE y NASA score de Test por modelo.
    - `rul_pred_vs_true_<subset>.png`: RUL real y predicho de cada motor de Test.
    - `error_distribution_<subset>.png`: histograma de `d = predicho - real` en Test.
    - `training_curves_<subset>.png`: RMSE de `val_truncated` por época.

Ningún paso entrena ni ajusta nada. Los reportes no guardan predicciones, así que los
dos gráficos por motor las recalculan con los modelos de `models/`. Se grafica lo que
haya disponible: un reporte o un modelo faltante se omite con un aviso.

Reglas anti-leakage:
    - El modelo de los gráficos por motor se elige por `val_truncated`, nunca por Test.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from xgboost import XGBRegressor

from src import baseline_xgb, config
from src.ensemble import collect_predictions, combine, load_members
from src.etl import load_artifacts
from src.metrics import rmse
from src.models_dl import get_device

logger = logging.getLogger(__name__)

DPI = 150

# Ancho de los intervalos del histograma de errores, en ciclos
ERROR_BIN_WIDTH = 5

# Diferencia de RMSE, en ciclos, entre las predicciones recalculadas y el reporte a
# partir de la cual se avisa. CPU y GPU difieren bastante menos que esto.
RMSE_TOL = 0.001

# Paleta distinguible con daltonismo. El color sigue al modelo, no a su posición.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
RED = "#e34948"

STYLE = {
    **sns.axes_style("whitegrid"),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": "#c3c2b7",
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": True,  # fondo liso: la grilla no cruza el texto
    "legend.facecolor": SURFACE,
    "legend.edgecolor": "none",
    "legend.framealpha": 1.0,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
}


@dataclass(frozen=True)
class RunSpec:
    """Corrida que se sabe graficar. `name` y `tag` son los de `config.run_name`."""

    name: str
    label: str
    tag: str = ""
    curve_color: str = ""  # vacío si el reporte no tiene curvas de entrenamiento


XGB = RunSpec(baseline_xgb.MODEL_NAME, "XGBoost (baseline)")
CNN1D = RunSpec("cnn1d", "CNN 1D", curve_color="#1baf7a")
LSTM_ASYM = RunSpec("lstm", "LSTM (MSE asimétrico)", "asym", curve_color="#eb6834")
LSTM = RunSpec("lstm", "LSTM (MSE)", curve_color=BLUE)
ENSEMBLE = RunSpec("ensemble", "Ensemble")

# En el orden en que se grafican
RUNS = (XGB, CNN1D, LSTM_ASYM, LSTM, ENSEMBLE)


@dataclass(frozen=True)
class TestPredictions:
    """Predicciones sobre el último ciclo de cada motor de Test, en ciclos.

    `y_true` es el RUL recortado a `RUL_CLIP`; `y_raw`, sin recortar.
    """

    spec: RunSpec
    y_true: np.ndarray
    y_raw: np.ndarray
    y_pred: np.ndarray


# --------------------------------------------------------------------------
# Reportes
# --------------------------------------------------------------------------
def run_metrics(spec: RunSpec, report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Métricas por conjunto. El reporte del ensemble incluye además sus miembros."""
    return report["metrics"]["ensemble"] if spec is ENSEMBLE else report["metrics"]


def load_reports(subset: str) -> dict[RunSpec, dict[str, Any]]:
    """Lee los reportes de `RUNS` que existan; los ilegibles se omiten con un aviso."""
    reports = {}
    for spec in RUNS:
        path = config.metrics_path(spec.name, subset, spec.tag)
        if not path.is_file():
            logger.info("%s: no existe %s", spec.label, path.name)
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            metrics = run_metrics(spec, report)
            for dataset in ("val_truncated", "test_last_cycle"):
                metrics[dataset]["rmse"], metrics[dataset]["nasa_score"]
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("%s: se omite %s (%r)", spec.label, path.name, error)
            continue
        reports[spec] = report
    return reports


# --------------------------------------------------------------------------
# Predicciones de Test
# --------------------------------------------------------------------------
def _predict_xgb(subset: str, report: dict[str, Any]) -> TestPredictions:
    path = config.model_path(XGB.name, subset, ext="json")
    if not path.is_file():
        raise FileNotFoundError(f"No existe el modelo {path}")
    model = XGBRegressor()
    model.load_model(path)

    data = load_artifacts(subset)
    test = baseline_xgb.add_rolling_features(
        data.test, data.feature_cols, report["windows"]
    )
    cols = baseline_xgb.get_model_feature_columns(test)
    last = test[baseline_xgb.last_cycle_mask(test)]
    return TestPredictions(
        spec=XGB,
        y_true=last[config.TARGET_COL].to_numpy(),
        y_raw=last[config.TARGET_RAW_COL].to_numpy(),
        y_pred=baseline_xgb.predict_rul(model, last, cols),
    )


def _predict_nets(
    spec: RunSpec, subset: str, report: dict[str, Any]
) -> TestPredictions:
    """Una red sola se trata como un ensemble de un miembro con peso 1."""
    weights = (
        {member["name"]: member["weight"] for member in report["members"]}
        if spec is ENSEMBLE
        else {config.run_name(spec.name, subset, spec.tag): 1.0}
    )
    device = get_device()
    members = load_members(list(weights), subset, device)
    test = collect_predictions(members, lambda d: d.test_last, device)
    return TestPredictions(
        spec=spec,
        y_true=test.y_true,
        y_raw=members[0].loaders.test_last_rul_raw,
        y_pred=combine(test.preds, np.array(list(weights.values()))),
    )


def predict_test(
    spec: RunSpec, subset: str, report: dict[str, Any]
) -> TestPredictions:
    """Recalcula las predicciones de Test y las contrasta con el RMSE del reporte."""
    preds = (
        _predict_xgb(subset, report)
        if spec is XGB
        else _predict_nets(spec, subset, report)
    )
    reported = run_metrics(spec, report)["test_last_cycle"]["rmse"]
    actual = rmse(preds.y_true, preds.y_pred)
    if abs(actual - reported) > RMSE_TOL:
        logger.warning(
            "%s: el RMSE de Test recalculado (%.4f) no coincide con el del reporte "
            "(%.4f). ¿El modelo o los datos son de otra corrida?",
            spec.label, actual, reported,
        )
    return preds


def rank_runs(reports: dict[RunSpec, dict[str, Any]]) -> list[RunSpec]:
    """Corridas en orden de preferencia: el ensemble y luego por `val_truncated`."""
    return sorted(
        reports,
        key=lambda spec: (
            spec is not ENSEMBLE,
            run_metrics(spec, reports[spec])["val_truncated"]["rmse"],
        ),
    )


def predict_best(
    reports: dict[RunSpec, dict[str, Any]], subset: str
) -> TestPredictions | None:
    """Predicciones de la mejor corrida cuyos modelos se puedan cargar."""
    for spec in rank_runs(reports):
        try:
            return predict_test(spec, subset, reports[spec])
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            logger.warning("%s: no se pudo predecir Test (%s)", spec.label, error)
    return None


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def _set_titles(ax: Axes, title: str, subtitle: str) -> None:
    """Título del gráfico con una línea secundaria debajo."""
    ax.set_title(title, pad=24, fontsize=12, fontweight="bold")
    ax.text(
        0, 1.02, subtitle, transform=ax.transAxes, va="bottom",
        fontsize=9, color=INK_SECONDARY,
    )


def plot_model_comparison(
    reports: dict[RunSpec, dict[str, Any]], subset: str
) -> Figure:
    """Barras de RMSE y NASA score de Test, en paneles separados por su escala."""
    specs = list(reports)
    labels = [textwrap.fill(spec.label, 14) for spec in specs]
    # La altura crece con los modelos para que el grosor de las barras no cambie
    fig = Figure(figsize=(10, 1.6 + 0.55 * len(specs)), layout="constrained")
    for ax, metric, title, fmt in zip(
        fig.subplots(1, 2, sharey=True),
        ("rmse", "nasa_score"),
        ("RMSE (ciclos)", "NASA score"),
        ("%.2f", "%.1f"),
    ):
        values = [
            run_metrics(spec, reports[spec])["test_last_cycle"][metric]
            for spec in specs
        ]
        bars = ax.barh(
            labels, values, height=0.5, linewidth=0,
            color=[MUTED if spec is XGB else BLUE for spec in specs],
        )
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_xlim(0, max(values) * 1.15 or 1)
        ax.grid(axis="y", visible=False)
        ax.spines["bottom"].set_visible(False)
    ax.set_ylim(len(specs) - 0.5, -0.5)  # eje compartido: afecta a los dos paneles
    fig.suptitle(
        f"{subset} · Desempeño en Test (último ciclo de cada motor, menor es mejor)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    return fig


def plot_pred_vs_true(preds: TestPredictions, subset: str) -> Figure:
    """RUL real y predicho por motor, ordenados por RUL real creciente."""
    order = np.argsort(preds.y_raw, kind="stable")
    x = np.arange(1, len(order) + 1)
    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.subplots()
    if (preds.y_raw > preds.y_true).any():
        ax.plot(x, preds.y_raw[order], color=MUTED, linewidth=1.2,
                label="RUL real sin recortar")
    ax.plot(x, preds.y_true[order], color=INK, linewidth=2,
            label=f"RUL real (techo en {config.RUL_CLIP})")
    ax.plot(x, preds.y_pred[order], color=BLUE, linewidth=1, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
            label=f"RUL predicho: {preds.spec.label}")
    _set_titles(
        ax,
        f"{subset} · RUL real y predicho en Test",
        f"{preds.spec.label} · {len(x)} motores, un punto por motor · "
        f"RMSE {rmse(preds.y_true, preds.y_pred):.2f} ciclos",
    )
    ax.set_xlabel("Motores de Test, ordenados por RUL real")
    ax.set_ylabel("RUL (ciclos)")
    ax.set_xlim(0, len(x) + 1)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return fig


def plot_error_distribution(preds: TestPredictions, subset: str) -> Figure:
    """Histograma de `d = predicho - real`, con las zonas temprana y tardía.

    El error exactamente nulo, que el recorte en `RUL_CLIP` vuelve posible, cuenta como
    zona segura: no cuesta nada en el NASA score.
    """
    d = np.asarray(preds.y_pred, dtype=np.float64) - preds.y_true
    late = d > 0
    # Los intervalos parten de 0: ninguno mezcla las dos zonas
    n_bins = max(1, int(np.ceil(np.abs(d).max() / ERROR_BIN_WIDTH)))
    edges = np.arange(n_bins + 1) * ERROR_BIN_WIDTH

    fig = Figure(figsize=(9, 4.8), layout="constrained")
    ax = fig.subplots()
    for errors, bins, color, label in (
        (d[~late], -edges[::-1], BLUE, "Zona segura: temprana o exacta (d ≤ 0)"),
        (d[late], edges, RED, "Zona peligrosa: tardía (d > 0)"),
    ):
        ax.hist(errors, bins=bins, color=color, edgecolor=SURFACE, linewidth=1.5,
                label=f"{label}: {len(errors)} motores")
    ax.axvline(0, color=INK, linewidth=1.2)
    _set_titles(
        ax,
        f"{subset} · Distribución del error en Test",
        f"{preds.spec.label} · {len(d)} motores · error medio {d.mean():+.1f} ciclos",
    )
    ax.set_xlabel("Error d = RUL predicho − RUL real (ciclos)")
    ax.set_ylabel("Motores")
    # Espacio para que la leyenda no tape barras
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    return fig


def plot_training_curves(
    reports: dict[RunSpec, dict[str, Any]], subset: str
) -> Figure | None:
    """RMSE de `val_truncated` por época; `None` si ningún reporte tiene curvas."""
    runs = {
        spec: report["history"]["val_rmse"]
        for spec, report in reports.items()
        if spec.curve_color and report.get("history", {}).get("val_rmse")
    }
    if not runs:
        return None
    fig = Figure(figsize=(9, 4.8), layout="constrained")
    ax = fig.subplots()
    for spec, values in runs.items():
        best = int(np.argmin(values))
        ax.plot(
            np.arange(1, len(values) + 1), values, color=spec.curve_color,
            linewidth=2,
            label=f"{spec.label}: {values[best]:.2f} en la época {best + 1}",
        )
        ax.plot(best + 1, values[best], "o", color=spec.curve_color, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    _set_titles(
        ax,
        f"{subset} · RMSE de val_truncated por época",
        "El punto marca la mejor época, que es la del checkpoint guardado",
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE (ciclos)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return fig


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run_visualize(subset: str = config.DEFAULT_SUBSET) -> list[Path]:
    """Genera todos los gráficos posibles y devuelve las rutas de los guardados."""
    config.ensure_directories()
    reports = load_reports(subset)
    if not reports:
        raise FileNotFoundError(
            f"No hay reportes de {subset} en {config.METRICS_DIR}. "
            "Entrená primero algún modelo"
        )
    preds = predict_best(reports, subset)

    paths = []
    # `Figure` sin pyplot no depende del backend ni altera el estado del notebook
    with matplotlib.rc_context(STYLE):
        figures = {
            "model_comparison": plot_model_comparison(reports, subset),
            "rul_pred_vs_true": preds and plot_pred_vs_true(preds, subset),
            "error_distribution": preds and plot_error_distribution(preds, subset),
            "training_curves": plot_training_curves(reports, subset),
        }
        for name, fig in figures.items():
            path = config.figure_path(name, subset)
            if fig is None:
                logger.warning("No hay datos para %s", path.name)
                continue
            fig.savefig(path, dpi=DPI)
            paths.append(path)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Gráficos de evaluación")
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    args = parser.parse_args()
    for path in run_visualize(args.subset):
        print(path)


if __name__ == "__main__":
    main()
