# Preguntas Abiertas (versión actual)

1. **Convención de nombres de métricas en el CSV**
   - ¿El nombre de la columna debe coincidir exactamente con el `metric.name` registrado?
      + **Sí**: hoy `load_metric` espera que, tras quitar el prefijo `metric_`, el nombre coincida con `BaseMetric.name`.
   - ¿Prefijo obligatorio?
      + **Para columnas del CSV, sí**: el runner sólo considera métricas las columnas que empiezan con `metric_` (`METRIC_PREFIX`).
      + `load_metric` acepta también el nombre “interno” sin prefijo (útil para tests/uso directo).

2. **Mapeo columna → configuración de métrica**
   - Hoy `Runner` hace `extra_config = row.to_dict()` pero **no se usa** en calculators/métricas.
   - Opciones recomendadas para implementar configuración:
     - **Columnas extra por métrica**: `cfg_<metric>_<param>` (pro: simple en CSV; contra: columnas explotan).
     - **Columna JSON por fila**: `config_json` con estructura `{ "global": {...}, "metrics": { "<metric>": {...} } }` (pro: escalable; contra: valida peor en Excel).
     - **Archivo config externo** (path en CLI): pro: versionable; contra: más fricción operativa.

3. **Capas por métrica**
   - El contrato actual del modelo soporta `layers: Optional[list[int]]` en `forward`, pero el runner **no parsea** capas desde CSV todavía.
   - Si se implementa, opciones:
     - **`layers` global por fila** (pro: simple; contra: no flexible).
     - **`layers_<metric>` por métrica** (pro: flexible; contra: más columnas).
     - **`config_json` con `layers` por métrica** (pro: escalable; contra: requiere validación).

4. **Batch size dinámico**
   - Política deseada: ¿valor inicial por defecto y step de reducción? ¿batch mínimo aceptable antes de abortar?

5. **Estado/errores en CSV**
   - Hoy el runner no escribe estado/errores y tampoco maneja excepciones por métrica (si falla, aborta).
   - Recomendación:
     - mantener las celdas `metric_*` limpias (numéricas/NaN)
     - añadir opcionalmente `metric_status_<name>` y `metric_error_<name>` (string corto), y el stacktrace sólo en logs.

6. **Backend de cálculo de métricas**
   - Hoy las métricas de saliency están implementadas en Torch.
   - Decisión pendiente: si se soporta JAX, definir si los calculators devuelven siempre `numpy` o se mantienen en backend nativo.

7. **Saliency (attention rollout) detalle operativo**
   - En el código actual `BaseSaliencyMetric.calculate` contiene un TODO y usa un placeholder (producto de tensores) en vez de rollout real.
   - Decisión pendiente: definir contrato de `ForwardOutputs["saliency"]`:
     - shape (H×W vs tokens)
     - rango (0..1) y dtype
     - tamaño (resolución del modelo vs imagen original).

8. **CLI: manejo de filas parcialmente completas**
   - No hay CLI hoy. Cuando exista, decidir si se procesan siempre celdas vacías o si hay filtros (`--only-metrics`, `--skip-partial`, etc.).

9. **Persistencia de logs**
   - Ubicación recomendada para logs por experimento/métrica (`logs/{experiment_id}/{metric}.log`?). ¿Rotación/tamaño máximo necesario?

10. **Packaging del script de saliency**
    - Hoy no hay rollout real. Decidir si se integra como módulo interno (`src/saliency/...`) o si se consume de un paquete externo (mejor para reutilización/benchmarking).
