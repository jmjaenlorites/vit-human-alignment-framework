from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    family: str
    allowed_keys: set[str] = field(default_factory=set)
    value_options: dict[str, set[str]] = field(default_factory=dict)


LEVELS_SPLITS = {"between_class", "class_border", "within_class"}
CONTRAST_CURVES_FREQS = {"all", "1p5", "3", "6", "12", "24"}
MASK_FREQS = {"all", "1p5", "3", "6", "12", "24"}
MASK_CONTRASTS = {"0075", "0150", "0225", "0300"}
MASK_ORIENTATIONS = {"0", "22p5", "45", "67p5", "90", "112p5", "135"}
LOW_HIGH_FREQS = {"all", "low", "high"}
PATH_FIELDS = {"levels_path", "imagenet_path", "data_path", "gt_path", "dataset_path"}
DATASET_DEFAULT_GROUPS = {"levels", "visturing", "tid", "nights", "saliency"}


METRIC_SPECS: dict[str, MetricSpec] = {
    "saliency_auc_judd": MetricSpec(
        family="saliency",
        allowed_keys={"dataset_path", "batch_size"},
    ),
    "saliency_pearson_correlation_coefficient": MetricSpec(
        family="saliency",
        allowed_keys={"dataset_path", "batch_size"},
    ),
    "tid_spearman_mos": MetricSpec(
        family="tid",
        allowed_keys={"dataset_path", "batch_size"},
    ),
    "nights_preference_accuracy": MetricSpec(
        family="nights",
        allowed_keys={"dataset_path", "batch_size"},
    ),
    "levels_triplet_accuracy": MetricSpec(
        family="levels",
        allowed_keys={"split", "levels_path", "imagenet_path", "batch_size"},
        value_options={"split": LEVELS_SPLITS},
    ),
    "visturing_spectral_sensitivity": MetricSpec(
        family="visturing",
        allowed_keys={"data_path", "gt_path", "batch_size"},
    ),
    "visturing_weber_law_pearson": MetricSpec(
        family="visturing",
        allowed_keys={"data_path", "gt_path", "batch_size"},
    ),
    "visturing_weber_law_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"data_path", "gt_path", "batch_size"},
    ),
    "visturing_csf_pearson": MetricSpec(
        family="visturing",
        allowed_keys={"data_path", "gt_path", "batch_size"},
    ),
    "visturing_csf_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"data_path", "gt_path", "batch_size"},
    ),
    "visturing_campbell_blakemore_pearson": MetricSpec(
        family="visturing",
        allowed_keys={"mask_freq", "data_path", "gt_path", "batch_size"},
        value_options={"mask_freq": {"all", "3", "6", "12"}},
    ),
    "visturing_campbell_blakemore_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"mask_freq", "data_path", "gt_path", "batch_size"},
        value_options={"mask_freq": {"all", "3", "6", "12"}},
    ),
    "visturing_contrast_curves_pearson": MetricSpec(
        family="visturing",
        allowed_keys={"freq", "data_path", "gt_path", "batch_size"},
        value_options={"freq": CONTRAST_CURVES_FREQS},
    ),
    "visturing_contrast_curves_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"freq", "data_path", "gt_path", "batch_size"},
        value_options={"freq": CONTRAST_CURVES_FREQS},
    ),
    "visturing_contrast_masking_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"freq", "mask_contrast", "data_path", "gt_path", "batch_size"},
        value_options={"freq": LOW_HIGH_FREQS, "mask_contrast": MASK_CONTRASTS},
    ),
    "visturing_frequency_masking_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"freq", "mask_freq", "data_path", "gt_path", "batch_size"},
        value_options={"freq": LOW_HIGH_FREQS, "mask_freq": MASK_FREQS},
    ),
    "visturing_orientation_masking_kendall": MetricSpec(
        family="visturing",
        allowed_keys={"freq", "mask_orientation", "data_path", "gt_path", "batch_size"},
        value_options={
            "freq": LOW_HIGH_FREQS,
            "mask_orientation": MASK_ORIENTATIONS,
        },
    ),
}


def get_metric_spec(metric_name: str) -> MetricSpec:
    try:
        return METRIC_SPECS[metric_name]
    except KeyError as exc:
        raise ValueError(f"Metric {metric_name} not supported") from exc


def list_supported_metric_names() -> list[str]:
    return sorted(METRIC_SPECS)
