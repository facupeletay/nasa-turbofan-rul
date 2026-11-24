# NASA Turbofan RUL

Mantenimiento predictivo sobre motores turbofán: estimación de la vida útil remanente (*Remaining Useful Life*, RUL) con el dataset **NASA C-MAPSS**.

Dado el historial de sensores de un motor, el objetivo es predecir **cuántos ciclos de vuelo le quedan** antes de alcanzar el fin de vida. El error se mide no solo por su magnitud, sino también por su riesgo: predecir tarde cuesta más que predecir temprano.

## Estado del proyecto

| Componente | Estado |
|---|---|
| Documentación de dominio y arquitectura | ✅ Completa |
| `src/config.py`, `src/metrics.py`, `src/etl.py`, `src/dataset.py`, `src/baseline_xgb.py`, `src/models_dl.py`, `src/ensemble.py`, `src/visualize.py` | ✅ Implementados |
| Evaluación en `FD001` | ✅ Cerrada ([resultados](#resultados-en-fd001)) |
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
- **Modelos:** XGBoost con rolling features (baseline), LSTM, CNN 1D y un ensemble convexo de las redes.
- **Métricas:** RMSE y NASA Asymmetric Score.

$$
s_i =
\begin{cases}
e^{-d_i/13} - 1, & d_i < 0 \quad \text{(temprana)} \\
e^{d_i/10} - 1, & d_i \ge 0 \quad \text{(tardía)}
\end{cases}
\qquad d_i = \widehat{RUL}_i - RUL_i
$$

## Resultados en FD001

Corrida oficial en Google Colab (GPU T4, semilla 42). Menor es mejor; las métricas están definidas en el [protocolo de evaluación](docs/02_process_architecture.md#5-protocolo-de-evaluación).

| Modelo | `val_truncated` RMSE | Test RMSE | Test NASA score | Test RMSE sin recortar |
|---|---:|---:|---:|---:|
| XGBoost (baseline) | 16.60 | 17.79 | 902.0 | 18.70 |
| CNN 1D | 15.10 | 16.27 | 535.3 | 17.69 |
| LSTM (MSE asimétrico, `late_weight=2.0`) | 12.61 | 14.03 | 356.5 | 15.28 |
| LSTM (MSE) | 12.00 | 13.63 | 330.8 | 14.93 |
| **Ensemble** | **11.96** | **13.56** | **326.9** | **14.88** |

El ensemble es `0.7825 · LSTM + 0.1938 · LSTM asimétrica + 0.0236 · CNN 1D`, con los pesos ajustados solo en `val_truncated`. Test son 100 motores, evaluados en su último ciclo registrado. La última columna usa el RUL sin el techo de 125 ciclos.

### Análisis

**Por qué la LSTM supera a XGBoost.** Respecto del baseline, la LSTM baja el RMSE de Test un 23.4 % y el NASA score un 63.3 %; con el ensemble, las bajas son de 23.8 % y **63.8 %**.

- **Dónde está la mejora.** El score crece de forma exponencial con el error, así que lo dominan unos pocos motores. En XGBoost, los 5 peores suman 565 de los 902 puntos; en la LSTM, 156 de 330.8. Los motores con predicción tardía por más de 20 ciclos bajan de 15 a 10, y el desvío del error, de 17.4 a 13.5 ciclos. Por eso el score cae mucho más que el RMSE: se redujeron los errores extremos, que son los peligrosos en operación.
- **Explicación propuesta.** XGBoost ve cada ciclo como una fila con medias y desvíos de 5 y 15 ciclos: conserva el nivel y la dispersión, pero no el orden dentro de la ventana. La LSTM recibe los 30 ciclos en secuencia y puede estimar la velocidad de degradación. La CNN 1D también ve la secuencia, pero su pooling global descarta en qué parte de la ventana apareció cada patrón. Es coherente con los resultados, pero no se aisló con un experimento.
- **La pérdida asimétrica no mejoró el score en esta corrida** (356.5 contra 330.8). Redujo el sesgo tardío medio de +2.1 a +1.5 ciclos, a costa de más dispersión. La diferencia no es concluyente: una repetición en CPU con la misma semilla invirtió el orden (274.7 contra 330.2).

**Qué aporta el ensemble.** Promediar modelos cuyos errores no están del todo correlacionados reduce la varianza. Aquí el margen es chico: los errores de las dos LSTM tienen correlación 0.94 (0.79 y 0.75 con la CNN), así que el desvío del error solo baja de 13.47 a 13.42 ciclos. La mejora sobre la LSTM es de 0.07 ciclos de RMSE y 1.2 % de score.

| Diferencia en Test | RMSE, IC 95 % | NASA score, IC 95 % |
|---|---:|---:|
| LSTM − XGBoost | −6.65 a −1.65 | −1064 a −159 |
| Ensemble − LSTM | −0.26 a +0.13 | −15 a +7 |

Los intervalos son de un bootstrap pareado sobre los 100 motores (20 000 réplicas) y no incluyen la variación entre semillas. **La ventaja sobre el baseline es clara; la del ensemble sobre la LSTM sola no se distingue de cero.**

**Limitaciones.** El ensemble todavía predice tarde en 60 de los 100 motores, con un error medio de +1.9 ciclos. La LSTM alcanzó su mejor época en la 46 de 50 sin que actuara el early stopping, por lo que podría mejorar con más épocas. Todo proviene de una sola semilla.

**Por qué no hay leakage.** Los pesos se ajustan solo con `val_truncated`, y Test se predice recién con los pesos fijos. El orden de los modelos en `val_truncated` es el mismo que en Test, por lo que elegir por validación lleva a la misma decisión. Los valores de `val_truncated` son algo optimistas, porque ese conjunto también elige la época y los pesos: la estimación limpia es la de Test.

### Reproducción

Desde la raíz del repositorio, en este orden. En Colab, anteponer `!` y usar runtime GPU para `src.models_dl`.

```bash
python -m src.etl --subset FD001
python -m src.baseline_xgb --subset FD001
python -m src.models_dl --arch cnn1d --subset FD001 --lr 0.0005
python -m src.models_dl --arch lstm --subset FD001
python -m src.models_dl --arch lstm --subset FD001 --loss asymmetric --late-weight 2.0 --tag asym
python -m src.ensemble --subset FD001 --models lstm_FD001 lstm_FD001_asym cnn1d_FD001 --optimize rmse
python -m src.visualize --subset FD001
```

Los hiperparámetros son los de los reportes oficiales. Los resultados varían según el hardware y las versiones de las dependencias. `src.visualize` guarda en `reports/figures/`:

| Gráfico | Contenido |
|---|---|
| `model_comparison_FD001.png` | RMSE y NASA score de Test por modelo |
| `rul_pred_vs_true_FD001.png` | RUL real y predicho de los 100 motores de Test |
| `error_distribution_FD001.png` | Histograma del error, con las zonas segura y peligrosa |
| `training_curves_FD001.png` | RMSE de `val_truncated` por época de las tres redes |

## Stack

| Área | Herramientas |
|---|---|
| Datos | kagglehub, pandas, NumPy, PyArrow |
| Modelado | scikit-learn, XGBoost, PyTorch, SciPy |
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
│   ├── models_dl.py            # LSTM, CNN 1D y entrenamiento
│   ├── ensemble.py             # Ensemble convexo de las redes
│   └── visualize.py            # Gráficos de evaluación
├── requirements.txt
└── README.md
```

Los datos, los modelos, las métricas y los gráficos **no se versionan**: viven en `/content/drive/MyDrive/nasa_rul/` y se regeneran con el código.

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
```

La secuencia completa está en [Reproducción](#reproducción); cada módulo acepta `--help`.

Para guardar datos y modelos en otra ubicación, definir la variable de entorno `NASA_RUL_ROOT`.

```python
from src.metrics import nasa_asymmetric_score, rmse

rmse([100, 50], [110, 40])                   # 10.0
nasa_asymmetric_score([100, 50], [110, 40])  # 2.876...
```

## Referencias

- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, 2008.
- Dataset: [behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) en Kaggle.
