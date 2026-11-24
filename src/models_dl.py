"""Modelos secuenciales en PyTorch (LSTM y CNN 1D) y loop de entrenamiento.

Contrato común de los modelos:
    entrada: float32 `(batch_size, window_size, num_features)`
    salida:  float32 `(batch_size,)`, RUL en ciclos

Cada modelo resuelve internamente cualquier reordenamiento de ejes, de modo que
el DataLoader y el loop de entrenamiento no dependen de la arquitectura.

Reglas anti-leakage:
    - El scheduler, el early stopping y el checkpoint miran solo `val_truncated`.
    - Test se evalúa una única vez, con el mejor checkpoint ya elegido.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.dataset import DEFAULT_BATCH_SIZE, RULDataLoaders, create_dataloaders
from src.metrics import nasa_asymmetric_score, rmse, truncated_metrics

logger = logging.getLogger(__name__)

# Las redes trabajan en RUL / RUL_CLIP y devuelven ciclos. Con el target en ciclos los
# gradientes superan siempre `grad_clip` y la LSTM pasa varias épocas prediciendo la
# media, donde el early stopping puede cortarla.
TARGET_SCALE = float(config.RUL_CLIP)

# Error, en ciclos, a partir del cual la pérdida Huber pasa de cuadrática a lineal
HUBER_DELTA_CYCLES = 10.0

LOSSES = ("mse", "huber", "asymmetric")


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparámetros del entrenamiento."""

    epochs: int = 50
    batch_size: int = DEFAULT_BATCH_SIZE
    window_size: int = config.WINDOW_SIZE
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss: str = "mse"
    late_weight: float = 2.0  # solo se usa con loss="asymmetric"
    patience: int = 10
    lr_patience: int = 4
    lr_factor: float = 0.5
    grad_clip: float = 1.0
    seed: int = config.RANDOM_SEED

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1:
            raise ValueError(
                f"epochs y patience deben ser >= 1; se recibió "
                f"{self.epochs} y {self.patience}"
            )
        if self.loss not in LOSSES:
            raise ValueError(f"loss inválida: {self.loss!r}. Opciones: {LOSSES}")
        if not (math.isfinite(self.late_weight) and self.late_weight > 0):
            raise ValueError(
                f"late_weight debe ser positivo y finito; se recibió {self.late_weight}"
            )


@dataclass
class TrainHistory:
    """Evolución por época y punto del mejor checkpoint.

    `train_loss` está en unidades de target normalizado y se mide con dropout activo;
    `val_rmse` y `val_score` son los de `val_truncated`, en ciclos.
    """

    train_loss: list[float] = field(default_factory=list)
    val_rmse: list[float] = field(default_factory=list)
    val_score: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    best_epoch: int = 0  # base 1; 0 si todavía no hay checkpoint


# --------------------------------------------------------------------------
# Arquitecturas
# --------------------------------------------------------------------------
def _regression_head(in_features: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class LSTMRegressor(nn.Module):
    """LSTM apilada; una cabeza densa lee el último estado oculto.

    (B, W, F) -> LSTM -> (B, W, H) -> último paso -> (B, H) -> MLP -> (B,)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        head_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hparams = {
            "num_features": num_features,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "head_size": head_size,
            "dropout": dropout,
        }
        self.lstm = nn.LSTM(
            num_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # nn.LSTM aplica dropout solo entre capas
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _regression_head(hidden_size, head_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1) * TARGET_SCALE


class CNN1DRegressor(nn.Module):
    """Convoluciones temporales y pooling global, con cabeza densa.

    (B, W, F) -> permute -> (B, F, W) -> Conv1d x N -> (B, C, W) -> pool -> (B, C)
    -> MLP -> (B,)
    """

    def __init__(
        self,
        num_features: int,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        head_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hparams = {
            "num_features": num_features,
            "channels": list(channels),
            "kernel_size": kernel_size,
            "head_size": head_size,
            "dropout": dropout,
        }
        blocks: list[nn.Module] = []
        in_channels = num_features
        for out_channels in channels:
            blocks += [
                nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = _regression_head(in_channels, head_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.conv(x.permute(0, 2, 1))).squeeze(-1)
        return self.head(pooled).squeeze(-1) * TARGET_SCALE


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMRegressor,
    "cnn1d": CNN1DRegressor,
}


# --------------------------------------------------------------------------
# Entrenamiento e inferencia
# --------------------------------------------------------------------------
def seed_everything(seed: int = config.RANDOM_SEED) -> None:
    """Fija las semillas de `random`, NumPy y PyTorch (CPU y CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # incluye todas las GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Devuelve `cuda` si está disponible y `cpu` en caso contrario."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AsymmetricMSELoss(nn.Module):
    """MSE que multiplica por `late_weight` el error de las predicciones tardías.

        l_i = w_i * (pred_i - target_i)^2,  w_i = late_weight  si pred_i >  target_i
                                                  1            si pred_i <= target_i

    Promedia sobre el batch sin dividir por la suma de pesos: con `late_weight = 1`
    coincide con `nn.MSELoss`. Es diferenciable en todo punto, porque ambas ramas
    tienen gradiente 0 en `pred == target`.
    """

    def __init__(self, late_weight: float = 2.0) -> None:
        super().__init__()
        self.late_weight = late_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"Formas distintas: pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}"
            )
        squared = (pred - target) ** 2
        return torch.where(pred > target, self.late_weight * squared, squared).mean()


def build_loss(name: str, late_weight: float = 2.0) -> nn.Module:
    """Pérdida sobre el target normalizado. `name` es una de `LOSSES`.

    `late_weight` solo afecta a `asymmetric`.
    """
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA_CYCLES / TARGET_SCALE)
    if name == "asymmetric":
        return AsymmetricMSELoss(late_weight)
    raise ValueError(f"loss inválida: {name!r}. Opciones: {LOSSES}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Ejecuta una época y devuelve la pérdida media por muestra."""
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x) / TARGET_SCALE, y / TARGET_SCALE)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Inferencia en modo `eval` y sin gradientes, recortada a `[0, RUL_CLIP]`.

    Requiere un loader sin shuffle para que el orden coincida con `unit_ids`.

    Returns:
        `(y_true, y_pred)`, ambos `(n,)`, en ciclos.
    """
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device, non_blocking=True)).cpu())
            trues.append(y)
    y_pred = torch.cat(preds).numpy()
    return torch.cat(trues).numpy(), np.clip(y_pred, 0.0, config.RUL_CLIP)


def score_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float | int]:
    """Métricas por fila, con el formato que usan los reportes."""
    return {
        "n": len(y_true),
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_asymmetric_score(y_true, y_pred),
    }


def _score_truncated(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float | int]:
    """Métricas de `val_truncated`, ponderadas por motor."""
    return truncated_metrics(*predict(model, loader, device), loader.dataset.unit_ids)


def save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    """Guarda pesos y metadata; un corte a mitad de escritura no pisa el anterior.

    `metadata` solo puede tener tipos básicos de Python: el archivo se lee con
    `weights_only=True`.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, tmp)
    tmp.replace(path)


def read_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Devuelve `(state_dict, metadata)` de un checkpoint.

    `metadata` es `None` en los checkpoints anteriores a su incorporación, que
    contienen solo el `state_dict`.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" not in payload:
        return payload, None
    return payload["state_dict"], payload["metadata"]


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Carga en `model` los pesos guardados por `save_checkpoint`."""
    state_dict, _ = read_checkpoint(path, device)
    model.load_state_dict(state_dict)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_config: TrainConfig,
    checkpoint_path: Path,
    checkpoint_metadata: dict[str, Any],
) -> TrainHistory:
    """Entrena con early stopping por RMSE de `val_truncated`.

    Guarda un checkpoint en cada mejora (Colab puede cortar la sesión) y al terminar
    deja cargados en `model` los mejores pesos. El criterio es el RMSE cualquiera sea
    la pérdida, para que las corridas sean comparables.

    Args:
        val_loader: el loader `val_truncated`; nunca uno de Test.
        checkpoint_metadata: lo necesario para reconstruir el modelo desde el archivo.
    """
    device = next(model.parameters()).device
    loss_fn = build_loss(train_config.loss, train_config.late_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config.lr_factor,
        patience=train_config.lr_patience,
    )

    history = TrainHistory()
    best_rmse = math.inf
    for epoch in range(1, train_config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, train_config.grad_clip
        )
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Época {epoch}: la pérdida de Train no es finita")
        val = _score_truncated(model, val_loader, device)
        scheduler.step(val["rmse"])

        history.train_loss.append(train_loss)
        history.val_rmse.append(val["rmse"])
        history.val_score.append(val["nasa_score"])
        history.learning_rate.append(lr)

        improved = val["rmse"] < best_rmse
        if improved:
            best_rmse = val["rmse"]
            history.best_epoch = epoch
            save_checkpoint(model, checkpoint_path, checkpoint_metadata)
        logger.info(
            "Época %3d/%d | train_loss=%.5f | val_truncated rmse=%.2f score=%.1f | "
            "lr=%.1e%s",
            epoch, train_config.epochs, train_loss, val["rmse"], val["nasa_score"],
            lr, " *" if improved else "",
        )
        if epoch - history.best_epoch >= train_config.patience:
            logger.info("Early stopping: %d épocas sin mejora", train_config.patience)
            break

    load_checkpoint(model, checkpoint_path, device)
    return history


# --------------------------------------------------------------------------
# Evaluación y orquestación
# --------------------------------------------------------------------------
def evaluate_model(
    model: nn.Module, loaders: RULDataLoaders, device: torch.device
) -> dict[str, dict[str, float | int]]:
    """Evalúa con el protocolo y los nombres que usa `baseline_xgb.evaluate_xgb`.

    `val_all` no es comparable con el del baseline: las redes necesitan una ventana
    completa y no predicen los primeros ciclos de cada motor.
    """
    test_true, test_pred = predict(model, loaders.test_last, device)
    return {
        "val_all": score_predictions(*predict(model, loaders.val_all, device)),
        "val_last_cycle": score_predictions(*predict(model, loaders.val_last, device)),
        "val_truncated": _score_truncated(model, loaders.val_truncated, device),
        "test_last_cycle": score_predictions(test_true, test_pred),
        "test_last_cycle_raw": score_predictions(loaders.test_last_rul_raw, test_pred),
    }


def run_training(
    arch: str,
    subset: str = config.DEFAULT_SUBSET,
    train_config: TrainConfig = TrainConfig(),
    tag: str = "",
) -> dict[str, Any]:
    """Carga artefactos, entrena, evalúa y guarda el modelo y el reporte.

    Args:
        arch: clave de `MODEL_REGISTRY`.
        subset: subconjunto de C-MAPSS con el ETL ya ejecutado.
        tag: sufijo de los archivos de salida, para no pisar otra corrida de la misma
            arquitectura y subset.

    Returns:
        El mismo reporte que se guarda en `metrics/<arch>_<subset>[_<tag>].json`.
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Arquitectura desconocida: {arch!r}. Opciones: {sorted(MODEL_REGISTRY)}"
        )
    name = config.run_name(arch, subset, tag)
    config.ensure_directories()
    seed_everything(train_config.seed)
    device = get_device()
    loaders = create_dataloaders(
        subset,
        window_size=train_config.window_size,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    model = MODEL_REGISTRY[arch](loaders.num_features).to(device)
    params = {"model": model.hparams, "train": asdict(train_config)}
    logger.info("%s | dispositivo: %s", name, device)

    history = train_model(
        model,
        loaders.train,
        loaders.val_truncated,
        train_config,
        config.model_path(arch, subset, tag=tag),
        checkpoint_metadata={
            "model": arch,
            "subset": subset,
            "tag": tag,
            "params": params,
            "rul_clip": config.RUL_CLIP,
            "feature_cols": loaders.feature_cols,
        },
    )
    if history.best_epoch == train_config.epochs:
        logger.warning(
            "El early stopping no actuó: la mejor época es la última (%d). "
            "Conviene aumentar epochs",
            train_config.epochs,
        )

    curves = asdict(history)
    best_epoch = curves.pop("best_epoch")
    report = {
        "model": arch,
        "subset": subset,
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "rul_clip": config.RUL_CLIP,
        "n_features": loaders.num_features,
        "feature_cols": loaders.feature_cols,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "device": device.type,
        "torch_version": str(torch.__version__),
        "best_epoch": best_epoch,
        "epochs_run": len(history.val_rmse),
        "metrics": evaluate_model(model, loaders, device),
        "history": curves,
    }
    config.metrics_path(arch, subset, tag).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    """Resumen legible del reporte."""
    name = config.run_name(report["model"], report["subset"], report["tag"])
    lines = [
        f"{name} | {report['n_features']} features | "
        f"{report['n_parameters']} parámetros | mejor época: {report['best_epoch']} "
        f"de {report['epochs_run']}",
        f"{'conjunto':<22}{'n':>7}{'RMSE':>9}{'NASA score':>13}",
    ]
    for name, m in report["metrics"].items():
        lines.append(f"{name:<22}{m['n']:>7}{m['rmse']:>9.2f}{m['nasa_score']:>13.1f}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Modelos secuenciales para RUL")
    parser.add_argument("--arch", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument(
        "--subset", default=config.DEFAULT_SUBSET, choices=config.SUBSETS
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--patience", type=int, default=defaults.patience,
        help="épocas sin mejora de val_truncated antes de detener",
    )
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--loss", default=defaults.loss, choices=LOSSES)
    parser.add_argument(
        "--late-weight", type=float, default=defaults.late_weight,
        help="peso del error tardío; solo se usa con --loss asymmetric",
    )
    parser.add_argument(
        "--tag", default="",
        help="sufijo del checkpoint y del reporte, para no pisar otra corrida",
    )
    args = parser.parse_args()

    report = run_training(
        args.arch,
        args.subset,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            window_size=args.window_size,
            learning_rate=args.lr,
            loss=args.loss,
            late_weight=args.late_weight,
            patience=args.patience,
        ),
        tag=args.tag,
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
