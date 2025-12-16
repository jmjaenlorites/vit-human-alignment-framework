# CSV Único y Runner

## Esquema mínimo del CSV (entrada = salida)
El runner actual detecta las métricas por el prefijo `metric_` (`METRIC_PREFIX`).

Requeridos hoy:
- `model_name`: nombre del adaptador a cargar (p.ej. `vit-b16`).
- **Columnas de métricas**: una o más columnas cuyo nombre empiece por `metric_`.

Opcionales (permitidos, pero hoy no consumidos por los calculators):
- cualquier otra columna que quieras usar como “config por fila”: se incluye en `extra_config` (e.g. rutas, batch_size, flags).

Ejemplo tabular (simplificado):
```csv
model_name,metric_saliency_auc_judd,metric_saliency_pearson_correlation_coefficient
vit-b16,,
```

Una celda vacía en una columna `metric_*` significa “pendiente” y se intentará calcular en la ejecución.

## Convención columna → métrica
El código soporta dos formas:
- columna de CSV: `metric_<metric_name>`
- nombre interno: `<metric_name>`

Ejemplos reales (hoy implementados):
- `metric_saliency_auc_judd` → `saliency_auc_judd`
- `metric_saliency_pearson_correlation_coefficient` → `saliency_pearson_correlation_coefficient`

## Flujo de actualización del CSV
1. Leer CSV con pandas (o polars) en memoria.
2. Para cada fila:
   - Identificar métricas con celda vacía (pendientes).
   - Instanciar modelo una sola vez por fila.
   - Ejecutar calculators por tipo (hoy: saliency).
   - Escribir los resultados en sus celdas `metric_<metric_name>`.
3. Escribir CSV de vuelta:
   - Modo seguro: escribir a `csv_path.tmp` y reemplazar original.
   - Modo in-place: sobrescribir si el FS lo permite.

> Importante: en el código actual la escritura es “in-place” usando `df.to_csv(...)`. No hay aún modo tmp+replace ni lock.

## Reanudación
- Una fila puede estar parcialmente completa: las métricas con valor se saltan; sólo se ejecutan las celdas vacías.
- No hay reintentos automáticos dentro de la misma ejecución.
- Tampoco hay (todavía) un mecanismo de “estado” o “error_id” en el CSV en el runner actual.

## Diagrama de runner y CSV
```mermaid
flowchart TD
    A[Leer CSV] --> B[Detectar filas con métricas vacías]
    B --> C[Instanciar modelo]
    C --> D[Agrupar métricas por tipo]
    D --> E[Ejecutar calculator (p.ej. saliency)]
    E --> F[Iterar batches + model.forward(...)]
    F --> G[Calcular métricas + agregación]
    G --> H[Actualizar celdas metric_*]
    H --> I[Escribir CSV]
    G -->|excepción| J[Actualmente: aborta ejecución]
```
