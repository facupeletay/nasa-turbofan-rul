# NASA Turbofan RUL

Mantenimiento predictivo sobre motores turbofán: estimación de la vida útil remanente (*Remaining Useful Life*, RUL) con el dataset **NASA C-MAPSS**.

Dado el historial de sensores de un motor, el objetivo es predecir **cuántos ciclos de vuelo le quedan** antes de alcanzar el fin de vida. El error se mide no solo por su magnitud, sino también por su riesgo: predecir tarde cuesta más que predecir temprano.

## Estado del proyecto

| Componente | Estado |
|---|---|
| Documentación de dominio y arquitectura | ✅ Completa |
| `src/config.py`, `src/metrics.py`, `src/etl.py`, `src/dataset.py`, `src/baseline_xgb.py`, `src/models_dl.py` | ✅ Implementados |
| `notebooks/colab_runner.ipynb` | 🚧 Preparación, ETL, baseline y redes; EDA pendiente |

## Documentación

| Documento | Contenido |
|---|---|
| [01 · Teoría de dominio](docs/01_domain_theory.md) | Física del turbofán, sensores, degradación del HPC, RUL a tramos y NASA score |
| [02 · Arquitectura de procesos](docs/02_process_architecture.md) | Flujo ETL, protocolo de evaluación y ejecución en Colab |
| [03 · Arquitectura lógica](docs/03_logical_architecture.md) | Módulos, contratos de tensores, reglas anti-leakage y comparativa de modelos |

## Enfoque

- **Dataset:** subconjunto `FD001` (100 motores de entrenamiento y 100 de test, un régimen operativo, falla del HPC).
- **Target:** RUL lineal a tramos con techo en **125 ciclos**.
- **Split:** Train/Val estricto por motor, sin *data leakage*.
- **Modelos:** XGBoost con rolling features (baseline), LSTM y CNN 1D.
- **Métricas:** RMSE y NASA Asymmetric Score.

$$
s_i =
\begin{cases}
e^{-d_i/13} - 1, & d_i < 0 \quad \text{(temprana)} \\
e^{d_i/10} - 1, & d_i \ge 0 \quad \text{(tardía)}
\end{cases}
\qquad d_i = \widehat{RUL}_i - RUL_i
$$

## Stack

| Área | Herramientas |
|---|---|
| Datos | kagglehub, pandas, NumPy, PyArrow |
| Modelado | scikit-learn, XGBoost, PyTorch |
| Visualización | matplotlib, seaborn |
| Ejecución | Google Colab (CPU y GPU) con Google Drive como almacenamiento |

## Estructura

```
nasa-turbofan-rul/
├── docs/
│   ├── 01_domain_theory.md
│   ├── 02_process_architecture.md
│   └── 03_logical_architecture.md
├── notebooks/
│   └── colab_runner.ipynb      # Puente: monta Drive, actualiza el repo y ejecuta
├── src/
│   ├── config.py               # Rutas dinámicas y constantes
│   ├── etl.py                  # Extract, Transform, Load
│   ├── metrics.py              # RMSE y NASA Asymmetric Score
│   ├── dataset.py              # Ventanas 3D y DataLoader
│   ├── baseline_xgb.py         # Baseline tabular
│   └── models_dl.py            # LSTM, CNN 1D y entrenamiento
├── requirements.txt
└── README.md
```

Los datos, los modelos y las métricas **no se versionan**: viven en `/content/drive/MyDrive/nasa_rul/` y se regeneran con el código.

## Uso

### Google Colab

1. Abrir `notebooks/colab_runner.ipynb` desde GitHub en Colab.
2. Sesión **CPU**: preparación, ETL, EDA y baseline XGBoost.
3. Sesión **GPU**: preparación, LSTM y CNN 1D.

### Local

Requiere Python 3.10 o superior. Los comandos se ejecutan desde la raíz del repositorio.

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.config           # Muestra las rutas activas
python -m src.etl              # ETL de FD001 (--subset para otro)
python -m src.baseline_xgb     # Baseline XGBoost (--help para ver opciones)
python -m src.models_dl --arch lstm   # o cnn1d (--help para ver opciones)
```

Para guardar datos y modelos en otra ubicación, definir la variable de entorno `NASA_RUL_ROOT`.

```python
from src.metrics import nasa_asymmetric_score, rmse

rmse([100, 50], [110, 40])                   # 10.0
nasa_asymmetric_score([100, 50], [110, 40])  # 2.876...
```

## Referencias

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, 2008.
- Dataset: [behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) en Kaggle.
