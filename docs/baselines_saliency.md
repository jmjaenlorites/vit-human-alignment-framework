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

El baseline `timm-random::vit_base_patch16_224` ya se calculó con el runner JSON del framework en servidor compartido y quedó guardado en `results-saliency-baselines.csv`. Sus mejores valores layer-wise en MIT1003 son:

| Baseline | `saliency_auc_judd` best/final | `saliency_pearson_correlation_coefficient` best/final | Estado |
| --- | ---: | ---: | --- |
| `timm-random::vit_base_patch16_224` | 0.5756 / 0.5756 | 0.1159 / 0.1159 | `framework_recomputed` |

Esto controla el sesgo arquitectónico y de attention rollout de un ViT sin pesos preentrenados. Los baselines clásicos `IttiSaliencyMaps` y `JuddSaliencyPy` siguen siendo referencias legacy hasta que sus mapas se integren como loaders/cálculos reproducibles dentro del framework.

El script `scripts/recompute_saliency_baselines.py` no ejecuta cómputo pesado por defecto. Solo resume el artefacto, puede verificar hashes de los ficheros legacy locales con `--check-source-files`, y puede emitir una plantilla de resultados recomputados sin asumir acceso a servidores remotos.
