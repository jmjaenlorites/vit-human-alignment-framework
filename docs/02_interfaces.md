# Interfaces y contratos (según el código actual)

Este documento describe los **contratos reales** que existen hoy en el repo (principalmente `Protocol`s) y cómo se conectan.

## Backend enum y prefijos
- `BackendEnum`: `torch | jax | numpy` (aunque no todo está implementado en todos los backends).
- `METRIC_PREFIX="metric_"`: el runner considera “métricas” a las columnas del CSV que comienzan por este prefijo.

## Adaptador de modelo: `BaseModelAdapter`
Fuente: `src/models/base.py`.

Contrato principal:
- `forward(batch, layers=None, return_features=True, return_saliency=False) -> ForwardOutputs`
- `ForwardOutputs` contiene **`features`** y/o **`saliency`** (hoy no hay logits/attn en el contrato).

Notas de diseño:
- El adaptador decide dispositivo con `get_backend_device(backend)`.
- El runner asume que el modelo expone `transform` (en `ViT_B_16`) y lo pasa al dataset loader.
- `preprocess` está documentado pero no implementado en el adaptador actual: se suele delegar en `transform`.

## Métrica: `BaseMetric` y especialización `BaseSaliencyMetric`
Fuente: `src/metrics/base.py`.

Conceptos clave:
- `BaseMetric` es una métrica “atómica” que define:
  - `name: str` (sin prefijo `metric_`)
  - `type: Literal["saliency", "perceptual"]`
  - `calculate(batch, batch_results) -> list[float]` (devuelve valores por sample/batch; el “mean final” lo hace el calculator)
- `BaseSaliencyMetric` implementa un `calculate(...)` común que:
  - toma `batch_results["saliency"]`
  - calcula un `predicted_saliency_maps` (hoy **placeholder**, no attention rollout real)
  - redimensiona ground truth saliency/fixation a la resolución del mapa
  - llama a `_calculate_torch(...)` / `_calculate_jax(...)` por sample.

## Calculator: `BaseMetricCalculator`
Fuente: `src/metrics/base.py`.

Responsabilidades:
- crear el dataset loader (con `get_dataset_loader(transform=...)`)
- iterar por batches
- llamar al modelo con los flags necesarios (`return_saliency` / `return_features`)
- acumular valores por métrica y devolver un dict `{metric_name: mean_value}`.

Ejemplo real: `SaliencyMetricsCalculator` en `src/metrics/saliency.py`.

## Registries: `load_model` y `load_metric`
El framework usa registries explícitos:
- `models.load_model(model_name: str) -> BaseModelAdapter` (hoy soporta `"vit-b16"`)
- `metrics.load_metric(metric_name_or_column: str) -> Callable[[BackendEnum], BaseMetric]`

Detalle importante de `load_metric`:
- acepta tanto `metric_saliency_auc_judd` (columna del CSV) como `saliency_auc_judd` (nombre “interno”).
- si recibe un nombre con prefijo `metric_`, lo recorta para comparar con `BaseMetric.name`.

## Especificación de experimento: `ExperimentSpec`
Fuente: `src/utils/common_types.py`.

Campos reales:
- `model_name: str`
- `metric_columns: list[str]` (columnas `metric_*` presentes en el CSV)
- `pending_metrics: list[BaseMetric]` (en la práctica: “factories” devueltas por `load_metric`, que luego se instancian con backend)
- `extra_config: dict[str, Any]` (hoy: `row.to_dict()`, no se consume aún)
- `row_index: int`

## Runner: `Runner`
Fuente: `src/runner/base.py`.

Semántica actual:
- identifica columnas de métricas por `col.startswith(METRIC_PREFIX)`
- considera pendiente si la celda está vacía (`None`, `NaN`, `"nan"`, `"null"`, `""`, etc.)
- carga el modelo una vez por fila
- instancia métricas con el backend del modelo
- agrupa por tipo (saliency/perceptual) y ejecuta calculators
- escribe resultados en las columnas `metric_<metric.name>` y guarda el CSV

Limitación explícita:
- hoy no hay un sistema de reintentos/errores robusto en el runner (no hay `try/except` por métrica); si una métrica falla, la ejecución abortará.
