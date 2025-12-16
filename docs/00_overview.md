# Visión General (vit-human-alignment-framework)

## Propósito
Framework modular en Python para evaluar **métricas de alineamiento humano** (p.ej. saliency) sobre modelos tipo ViT, con un flujo centrado en un **CSV único** que actúa como fuente de verdad (configuración + resultados).

## Estado actual (importante: lo que el repo hace hoy)
- **Ejecución**: desde Python (script `main.py` o instanciando `Runner`). **No hay CLI**.
- **Runner**: recorre un CSV, detecta columnas de métricas vacías con prefijo `metric_` y calcula únicamente lo pendiente.
- **Modelos**: registry muy simple en `src/models/__init__.py`. Implementado: `model_name=vit-b16` (carga `timm`).
- **Métricas**: registry en `src/metrics/__init__.py`. Implementadas: `saliency_auc_judd` y `saliency_pearson_correlation_coefficient`.
- **Dataset**: loader MIT1003 (torch) en `src/dataset_loaders/saliency.py`. La ruta del dataset está **hardcodeada** (pendiente de parametrizar).
- **Backends**: el enum soporta `torch/jax/numpy`, pero el flujo actual de cálculo de métricas está implementado para **Torch**; JAX está sin implementar en los calculators actuales.

## Componentes principales (reales)
- **Model adapters (`src/models/`)**: adaptadores con una interfaz común (`BaseModelAdapter`) que exponen `forward(..., return_features, return_saliency)` y manejan el dispositivo según backend.
- **Metrics (`src/metrics/`)**: métricas atómicas (clases) y “calculators” que agrupan la ejecución sobre un dataset (p.ej. `SaliencyMetricsCalculator`).
- **Dataset loaders (`src/dataset_loaders/`)**: loaders de datasets (torch) para iterar por batches y devolver tuplas de tensores.
- **Runner (`src/runner/`)**: orquestación basada en CSV; detecta pendientes y actualiza el CSV con los resultados.
- **Utils (`src/utils/`)**: enums, tipos y utilidades comunes (p.ej. `METRIC_PREFIX="metric_"`, selección de dispositivo).

## Idea clave del CSV único
- Cada fila del CSV define un **modelo** (`model_name`) y contiene columnas de métricas **prefijadas** con `metric_`.
- Una celda vacía en una columna `metric_*` significa “pendiente”: se calcula y se escribe el resultado en esa celda.

## Diagrama (visión simplificada)
```mermaid
flowchart LR
    CSV[CSV único\nmodel_name + metric_*] --> Runner[Runner.execute()]
    Runner -->|load_model| Model[BaseModelAdapter]
    Runner -->|load_metric| MetricFactory[metrics.load_metric]
    MetricFactory --> Calculator[BaseMetricCalculator\n(ej. SaliencyMetricsCalculator)]
    Calculator --> Loader[DatasetLoader\n(torch DataLoader)]
    Loader --> Model
    Model --> Calculator
    Calculator --> Runner
    Runner -->|escribe| CSV
```
