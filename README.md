## ViT Human Alignment Framework

Framework modular (Python) para **evaluar alineamiento humano** de modelos de visión tipo ViT a través de un flujo reproducible basado en **un CSV como “fuente de verdad”**.

### Estado actual (lo que hay implementado hoy)
- **Ejecución**: vía API Python / script (`main.py`). **No hay CLI todavía**.
- **Runner**: lee un CSV, detecta columnas de métricas vacías con prefijo `metric_`, ejecuta lo pendiente y **actualiza el mismo CSV**.
- **Backends**: Torch soportado en los componentes usados; JAX aparece como enum pero **no está implementado** en los calculators actuales.
- **Modelo**: `model_name=vit-b16` (se carga con `timm`, pesos preentrenados).
- **Métricas**: saliency (p.ej. AUC Judd, Pearson correlation) sobre dataset MIT1003.
- **Dataset**: MIT1003 loader está implementado, pero la **ruta está hardcodeada** en el loader (pendiente de parametrización).

### Quickstart

1) Ajusta el CSV (ejemplo mínimo):

```csv
model_name,metric_saliency_auc_judd,metric_saliency_pearson_correlation_coefficient
vit-b16,,
```

2) Ejecuta el runner (ejemplo actual):

- Edita `main.py` si quieres cambiar el `csv_path`.
- Ejecuta:

```bash
python main.py
```

### Documentación de diseño y guía de extensión
La documentación vive en `docs/`:
- `docs/00_overview.md`: visión general + estado actual.
- `docs/01_arquitectura.md`: estructura real de paquetes y responsabilidades.
- `docs/02_interfaces.md`: contratos (modelos, métricas, calculators, loader registry).
- `docs/03_csv_y_runner.md`: esquema del CSV y comportamiento del runner.
- `docs/04_flujos_de_ejecucion.md`: flujos de ejecución (Python) y puntos de extensión.
- `docs/05_roadmap.md`: roadmap realista basado en el estado del repo.
