# Roadmap y Tareas

## Epic: End-to-end estable (MVP)
- **Runner robusto con manejo de errores**
  - Resultado: `try/except` por fila/métrica (o por calculator) para que una métrica no tumbe todo el run.
  - Salida: opcional `metric_error_<name>` / `metric_status_<name>` en CSV o logs estructurados.
- **Escritura segura del CSV**
  - Resultado: modo `tmp + atomic replace` y/o lock para evitar corrupción si se interrumpe el proceso.
- **Configuración de dataset sin rutas hardcodeadas**
  - Resultado: parametrizar `MIT1003_DATASET_PATH` (por CLI/env/columna CSV/config).

## Epic: Infraestructura de métricas
- **Implementar “saliency rollout” real**
  - Resultado: reemplazar el placeholder actual en `BaseSaliencyMetric.calculate` por attention rollout (o el método que se decida) con un contrato claro de salida.
- **Nuevos calculators y tipos**
  - Resultado: `PerceptualMetricsCalculator` (hoy comentado) y métricas base para esa categoría.
- **Normalización / contratos de tensores**
  - Resultado: documentar shape/dtype/rango esperados para `saliency`, `stimulus`, `fixation` y dónde se redimensiona.

## Epic: Runner + CSV único
- **Config por fila y por métrica**
  - Resultado: definir cómo pasar configs desde CSV a calculators/métricas (p.ej. columnas extra o un `config_json` global).
- **Selección de backend**
  - Resultado: soportar `BackendEnum.JAX` en calculators (o eliminarlo del path hasta implementarlo).

## Epic: CLI
- **Añadir CLI minimal**
  - Resultado: `run --csv ... --output ... --backend torch|jax` con Typer/Click (ya está como dependencia indirecta).
- **Ayudas de usuario**
  - Resultado: `--list-models`, `--list-metrics`, `--validate-csv`.

## Epic: Logging, errores y reanudación
- **Logging estructurado**
  - Resultado: logger con contexto (row_index, model_name, métricas), y stacktrace en log (no en CSV).
- **Resumen de run**
  - Resultado: conteo de métricas calculadas, pendientes, fallidas; tiempos por etapa.

## Epic: Calidad y utilidades
- **Tests mínimos**
  - Resultado: tests de registry, runner con dataset dummy, y “golden CSV” de ejemplo.
- **Packaging**
  - Resultado: actualizar `pyproject.toml` (descripción, scripts/entrypoints cuando exista CLI).
