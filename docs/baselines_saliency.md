# Baselines de saliency MIT1003

Este repositorio incluye `data/baselines/saliency_mit1003_reference.json` como artefacto de procedencia para dos baselines históricos de MIT1003 (`IttiSaliencyMaps` y `JuddSaliencyPy`). Sus valores vienen del repositorio local hermano `ViT-saliency-alignment` y se etiquetan como `reference_from_legacy_repo`.

La correspondencia directa con métricas actuales del framework es:

| Métrica legacy | Métrica del framework | Estado |
| --- | --- | --- |
| `AUC_Judd` | `saliency_auc_judd` | referencia legacy, pendiente de recomputar |
| `Correlation_Pearson` | `saliency_pearson_correlation_coefficient` | referencia legacy, pendiente de recomputar |
| `NSS` | - | solo referencia legacy |
| `KL_divergence` | - | solo referencia legacy |

Las tablas del paper deben usar valores `framework_recomputed` cuando existan. Los valores `reference_from_legacy_repo` sirven solo como puente de procedencia y comprobación de cordura mientras se implementa o ejecuta una recomputación dentro de este framework.

El baseline `timm-random::vit_base_patch16_224` se calculó primero con el
runner JSON del framework en servidor compartido y quedó guardado en
`results-saliency-baselines.csv`. Sus mejores valores layer-wise en MIT1003 son:

| Baseline | `saliency_auc_judd` best/final | `saliency_pearson_correlation_coefficient` best/final | Estado |
| --- | ---: | ---: | --- |
| `timm-random::vit_base_patch16_224` | 0.5756 / 0.5756 | 0.1159 / 0.1159 | `framework_recomputed` |

Esto controla el sesgo arquitectónico y de attention rollout de un ViT sin
pesos preentrenados. Además, `data/experiments.framework-saliency-baselines.json`
incluye dos adapters saliency-only framework-native:

| Baseline | `saliency_auc_judd` best/final | `saliency_pearson_correlation_coefficient` best/final | Estado |
| --- | ---: | ---: | --- |
| `saliency-baseline::uniform` | 0.5029 / 0.5029 | NaN / NaN | `framework_recomputed` |
| `saliency-baseline::center-gaussian` | 0.8310 / 0.8310 | 0.4869 / 0.4869 | `framework_recomputed` |

El baseline gaussiano centrado deja claro que MIT1003 contiene un prior espacial
fuerte; por tanto, las curvas de saliency de modelos entrenados deben
interpretarse junto a controles de centro, no solo junto a otros ViTs. Los
baselines clásicos `IttiSaliencyMaps` y `JuddSaliencyPy` siguen siendo
referencias legacy hasta que sus mapas se integren como loaders/cálculos
reproducibles dentro del framework.

Para la revisión mayor del artículo, `data/experiments.framework-baselines.json`
amplía este patrón al grid compacto del paper: saliency, TID, NIGHTS, THINGS,
LEVELS y una métrica psicofísica custom. Esa configuración debe ejecutarse como
un experimento explícito y persistir con sidecars unitarios; no se deben añadir
columnas mágicas al CSV principal. Mientras no esté cerrado el output v3, el
manuscrito debe distinguir entre:

- baselines framework-native ya ejecutados;
- referencias legacy con procedencia y hashes verificables;
- controles chance/null que son valores analíticos de la familia métrica, no
  outputs de modelo.

Desde la revisión mayor, los modelos `timm-random::` y `timm-untrained::` se
inicializan con semilla fija `random_seed=0` durante la creación del modelo y
restauran después el estado RNG del proceso. Si se comparan baselines random en
el paper, usar resultados generados con ese contrato semillado.

El primer run semillado multi-familia se conserva como procedencia en
`results-framework-baselines-seed0.csv`. Completó saliency AUC-Judd, TID,
NIGHTS, THINGS y LEVELS between-class; falló de forma no destructiva en
`levels-class-border-random-vit` con `Broken pipe` antes de llegar a las filas
restantes. Por tanto, el paper no debe usarlo como tabla final. La tabla
principal debe basarse en `results-framework-baselines-seed0-v3.csv`,
`unit-results-framework-baselines-seed0-v3/` y los resúmenes bootstrap/null
derivados.

El script `scripts/recompute_saliency_baselines.py` no ejecuta cómputo pesado por defecto. Solo resume el artefacto, puede verificar hashes de los ficheros legacy locales con `--check-source-files`, y puede emitir una plantilla de resultados recomputados sin asumir acceso a servidores remotos.
