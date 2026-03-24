## ViT Human Alignment Framework

Framework modular (Python) para **evaluar alineamiento humano** de modelos de visión tipo ViT a través de flujos reproducibles basados en **CSV legacy** o en **JSON + `results.csv`**.

### Estado actual (lo que hay implementado hoy)
- **Ejecución**: vía API Python / script (`main.py`). **No hay CLI todavía**.
- **Runner**:
  - flujo legacy: lee un CSV, detecta columnas de métricas vacías con prefijo `metric_`, ejecuta lo pendiente y **actualiza el mismo CSV**;
  - flujo nuevo: lee un `experiments.json`, expande experimentos y persiste estado/resultados en `results.csv`.
- **Backends**: Torch soportado en los componentes usados; JAX aparece como enum pero **no está implementado** en los calculators actuales.
- **Modelo**:
  - aliases internos: `vit-b16`, `vit-b32`, `vit-l14`, `vit-h14`;
  - flujo JSON: también acepta `timm::<model_name>` para modelos disponibles en `timm`.
- **Métricas**: saliency (p.ej. AUC Judd, Pearson correlation) sobre dataset MIT1003.
- **Dataset**: MIT1003 loader está implementado, pero la **ruta está hardcodeada** en el loader (pendiente de parametrización).

### Quickstart

#### Opción A: flujo CSV legacy

1) Ajusta el CSV (ejemplo mínimo):

```csv
model_name,metric_saliency_auc_judd,metric_saliency_pearson_correlation_coefficient
vit-b16,,
```

2) Ejecuta el runner:

- Edita `main.py` si quieres cambiar el `csv_path`.
- Ejecuta:

```bash
uv sync
uv run main.py
```

#### Opción B: flujo JSON recomendado

1) Usa `data/experiments.example.json` como punto de partida.

2) Ejecuta el runner desde Python:

```python
from src.runner.base import Runner

Runner(
    json_path="data/experiments.example.json",
    results_path="results.csv",
).execute()
```

3) El runner:
- valida modelos, métricas y config antes de ejecutar;
- reanuda automáticamente usando `results.csv`;
- no vuelve a ejecutar filas ya marcadas como `done`.

Ejemplo mínimo de JSON:

```json
{
  "defaults": {
    "batch_size": 2
  },
  "dataset_defaults": {
    "levels": {
      "levels_path": "tests/data/levels",
      "imagenet_path": "tests/data/levels/images"
    },
    "visturing": {
      "data_path": "tests/data/visturing",
      "gt_path": "tests/data/visturing"
    }
  },
  "models": ["vit-b16"],
  "experiments": [
    {
      "experiment_id": "levels-between",
      "metric": "levels_triplet_accuracy",
      "config": {"split": "between_class"}
    },
    {
      "experiment_id": "visturing-csf-rg",
      "metric": "visturing_csf_pearson",
      "config": {"channel": "red_green"}
    }
  ]
}
```

\* Asegurate de tener uv instalado

### Documentación de diseño y guía de extensión
La documentación vive en `docs/`:
- `docs/00_overview.md`: visión general + estado actual.
- `docs/01_arquitectura.md`: estructura real de paquetes y responsabilidades.
- `docs/02_interfaces.md`: contratos (modelos, métricas, calculators, loader registry).
- `docs/03_csv_y_runner.md`: flujo CSV legacy y flujo JSON nuevo.
- `docs/04_flujos_de_ejecucion.md`: flujos de ejecución (Python), reanudación y puntos de extensión.
- `docs/05_roadmap.md`: roadmap realista basado en el estado del repo.
