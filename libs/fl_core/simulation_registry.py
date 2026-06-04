from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    modality: str
    task_type: str
    input_shape: tuple[int, ...]
    num_classes: int
    default_model: str
    compatible_models: tuple[str, ...]
    supported_splits: tuple[str, ...] = ("iid", "non_iid")
    loader: str = "torchvision"
    requires_local_data: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: tuple[str, ...]
    modalities: tuple[str, ...]
    task_types: tuple[str, ...]
    description: str = ""


LINEAR_AGGREGATIONS = ("fedavg", "weighted_avg", "simple_avg")
EQUAL_WEIGHT_AGGREGATIONS = ("fedavg", "simple_avg")
EXECUTABLE_AGGREGATIONS = (*LINEAR_AGGREGATIONS,)
COMPRESSION_METHODS = ("global_topk", "random_k", "threshold", "sign", "quant_int8")


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="Mclr_Logistic",
        aliases=("mclr", "logistic", "logistic_regression", "MCLR"),
        modalities=("image", "text"),
        task_types=("classification",),
        description="Multiclass logistic regression over flattened features.",
    ),
    ModelSpec(
        name="LeNet",
        aliases=("lenet",),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="DNN",
        aliases=("dnn", "mlp"),
        modalities=("image", "text"),
        task_types=("classification",),
        description="Two-layer dense network with configurable flattened input.",
    ),
    ModelSpec(
        name="FedAvgCNN",
        aliases=("cnn", "simple_cnn", "fedavg_cnn"),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="ResNet18",
        aliases=("resnet", "resnet18"),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="ResNet34",
        aliases=("resnet34",),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="ResNet50",
        aliases=("resnet50",),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="AlexNet",
        aliases=("alexnet",),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="MobileNet",
        aliases=("mobilenet", "mobilenet_v2", "mobilenetv2"),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="GoogleNet",
        aliases=("googlenet", "google_net"),
        modalities=("image",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="TextLogistic",
        aliases=("text_logistic",),
        modalities=("text",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="TextDNN",
        aliases=("text_dnn",),
        modalities=("text",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="TextCNN",
        aliases=("text_cnn",),
        modalities=("text",),
        task_types=("classification",),
    ),
    ModelSpec(
        name="CharLSTM",
        aliases=("char_lstm", "lstm", "char_rnn"),
        modalities=("sequence",),
        task_types=("next_char",),
    ),
)


DIGIT_MODELS = ("Mclr_Logistic", "LeNet", "DNN")
SMALL_IMAGE_MODELS = (
    "Mclr_Logistic",
    "FedAvgCNN",
    "DNN",
    "ResNet18",
    "AlexNet",
    "MobileNet",
    "GoogleNet",
)
GENERAL_IMAGE_MODELS = (
    "FedAvgCNN",
    "DNN",
    "ResNet18",
    "ResNet34",
    "AlexNet",
    "MobileNet",
    "GoogleNet",
)
TEXT_MODELS = ("TextLogistic", "TextDNN", "TextCNN")
SEQUENCE_MODELS = ("CharLSTM",)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="MNIST",
        aliases=("mnist",),
        modality="image",
        task_type="classification",
        input_shape=(1, 28, 28),
        num_classes=10,
        default_model="LeNet",
        compatible_models=DIGIT_MODELS,
    ),
    DatasetSpec(
        name="EMNIST",
        aliases=("emnist",),
        modality="image",
        task_type="classification",
        input_shape=(1, 28, 28),
        num_classes=47,
        default_model="LeNet",
        compatible_models=DIGIT_MODELS,
        metadata={"split": "balanced"},
    ),
    DatasetSpec(
        name="FEMNIST",
        aliases=("femnist",),
        modality="image",
        task_type="classification",
        input_shape=(1, 28, 28),
        num_classes=62,
        default_model="LeNet",
        compatible_models=DIGIT_MODELS,
        loader="leaf_femnist",
        requires_local_data=True,
    ),
    DatasetSpec(
        name="Fashion-MNIST",
        aliases=("fashion-mnist", "fashion_mnist", "fashionmnist"),
        modality="image",
        task_type="classification",
        input_shape=(1, 28, 28),
        num_classes=10,
        default_model="LeNet",
        compatible_models=DIGIT_MODELS,
    ),
    DatasetSpec(
        name="CIFAR-10",
        aliases=("cifar10", "cifar-10", "Cifar10"),
        modality="image",
        task_type="classification",
        input_shape=(3, 32, 32),
        num_classes=10,
        default_model="FedAvgCNN",
        compatible_models=SMALL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="CIFAR-100",
        aliases=("cifar100", "cifar-100", "Cifar100"),
        modality="image",
        task_type="classification",
        input_shape=(3, 32, 32),
        num_classes=100,
        default_model="FedAvgCNN",
        compatible_models=SMALL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="AG News",
        aliases=("ag_news", "agnews"),
        modality="text",
        task_type="classification",
        input_shape=(5000,),
        num_classes=4,
        default_model="TextDNN",
        compatible_models=TEXT_MODELS,
        loader="csv_text",
        requires_local_data=True,
        metadata={"feature_dim": 5000},
    ),
    DatasetSpec(
        name="Sogou News",
        aliases=("sogou_news", "sogou"),
        modality="text",
        task_type="classification",
        input_shape=(8000,),
        num_classes=5,
        default_model="TextDNN",
        compatible_models=TEXT_MODELS,
        loader="csv_text",
        requires_local_data=True,
        metadata={"feature_dim": 8000},
    ),
    DatasetSpec(
        name="Tiny-ImageNet",
        aliases=("tiny_imagenet", "tinyimagenet", "tiny-imagenet"),
        modality="image",
        task_type="classification",
        input_shape=(3, 32, 32),
        num_classes=200,
        default_model="FedAvgCNN",
        compatible_models=SMALL_IMAGE_MODELS,
        loader="tiny_imagenet",
        requires_local_data=True,
    ),
    DatasetSpec(
        name="Country211",
        aliases=("country211",),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=211,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="Flowers102",
        aliases=("flowers102", "flowers-102"),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=102,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="GTSRB",
        aliases=("gtsrb",),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=43,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="Shakespeare",
        aliases=("shakespeare",),
        modality="sequence",
        task_type="next_char",
        input_shape=(80,),
        num_classes=128,
        default_model="CharLSTM",
        compatible_models=SEQUENCE_MODELS,
        loader="shakespeare",
        requires_local_data=True,
        metadata={"sequence_length": 80, "vocab_size": 128},
    ),
    DatasetSpec(
        name="Stanford Cars",
        aliases=("stanford_cars", "stanford-cars", "cars"),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=196,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
    ),
    DatasetSpec(
        name="COVIDx",
        aliases=("covidx", "covid-x"),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=3,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
        loader="local_image_folder",
        requires_local_data=True,
    ),
    DatasetSpec(
        name="Kvasir",
        aliases=("kvasir",),
        modality="image",
        task_type="classification",
        input_shape=(3, 64, 64),
        num_classes=8,
        default_model="MobileNet",
        compatible_models=GENERAL_IMAGE_MODELS,
        loader="local_image_folder",
        requires_local_data=True,
    ),
)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


DATASET_BY_NAME: dict[str, DatasetSpec] = {
    _normalize_key(spec.name): spec for spec in DATASET_SPECS
}
for _spec in DATASET_SPECS:
    for _alias in _spec.aliases:
        DATASET_BY_NAME[_normalize_key(_alias)] = _spec


MODEL_BY_NAME: dict[str, ModelSpec] = {
    _normalize_key(spec.name): spec for spec in MODEL_SPECS
}
for _spec in MODEL_SPECS:
    for _alias in _spec.aliases:
        MODEL_BY_NAME[_normalize_key(_alias)] = _spec


def canonical_dataset_name(name: str) -> str:
    spec = get_dataset_spec(name)
    if spec is None:
        return name
    return spec.name


def canonical_model_name(name: str) -> str:
    spec = get_model_spec(name)
    if spec is None:
        return name
    return spec.name


def get_dataset_spec(name: str | None) -> DatasetSpec | None:
    if not name:
        return None
    return DATASET_BY_NAME.get(_normalize_key(str(name)))


def get_model_spec(name: str | None) -> ModelSpec | None:
    if not name:
        return None
    return MODEL_BY_NAME.get(_normalize_key(str(name)))


def list_dataset_names() -> list[str]:
    return [spec.name for spec in DATASET_SPECS]


def list_model_names() -> list[str]:
    return [spec.name for spec in MODEL_SPECS]


def is_model_compatible_with_dataset(model_name: str, dataset_name: str) -> bool:
    dataset = get_dataset_spec(dataset_name)
    model = get_model_spec(model_name)
    if dataset is None or model is None:
        return False
    return model.name in dataset.compatible_models


def get_compatible_models(dataset_name: str) -> list[str]:
    dataset = get_dataset_spec(dataset_name)
    if dataset is None:
        return []
    return list(dataset.compatible_models)


def resolve_model_name_for_dataset(model_name: str | None, dataset_name: str) -> str:
    dataset = get_dataset_spec(dataset_name)
    if model_name is None or not str(model_name).strip() or str(model_name).strip().lower() == "auto":
        return dataset.default_model if dataset is not None else "FedAvgCNN"
    return canonical_model_name(str(model_name))


def validate_training_combination(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    federated_cfg = config.get("federated") if isinstance(config.get("federated"), dict) else {}
    privacy_cfg = config.get("privacy") if isinstance(config.get("privacy"), dict) else {}
    compression_cfg = config.get("compression") if isinstance(config.get("compression"), dict) else {}

    dataset_name = str(dataset_cfg.get("name", ""))
    raw_model_name = model_cfg.get("name")
    aggregation = str(federated_cfg.get("aggregation", "fedavg")).lower()

    dataset = get_dataset_spec(dataset_name)
    model_name = resolve_model_name_for_dataset(raw_model_name, dataset_name)
    model = get_model_spec(model_name)

    if dataset is None:
        errors.append(f"Unsupported dataset: {dataset_name}")
    if model is None:
        errors.append(f"Unsupported model: {model_name}")
    if dataset is not None and model is not None and model.name not in dataset.compatible_models:
        allowed = ", ".join(dataset.compatible_models)
        errors.append(f"Model {model.name} is not compatible with dataset {dataset.name}. Allowed models: {allowed}")

    if aggregation not in EXECUTABLE_AGGREGATIONS:
        allowed = ", ".join(EXECUTABLE_AGGREGATIONS)
        errors.append(f"Aggregation {aggregation} is not executable in simulation. Allowed aggregations: {allowed}")

    he_cfg = privacy_cfg.get("homomorphic_encryption") if isinstance(privacy_cfg, dict) else {}
    ckks_enabled = isinstance(he_cfg, dict) and he_cfg.get("enable") is True
    dp_cfg = privacy_cfg.get("differential_privacy") if isinstance(privacy_cfg, dict) else {}
    dp_enabled = isinstance(dp_cfg, dict) and dp_cfg.get("enable") is True
    secure_agg_cfg = privacy_cfg.get("secure_aggregation") if isinstance(privacy_cfg, dict) else {}
    secure_agg_enabled = isinstance(secure_agg_cfg, dict) and secure_agg_cfg.get("enable") is True
    sparsification_cfg = compression_cfg.get("sparsification") if isinstance(compression_cfg, dict) else {}
    sparsification_enabled = isinstance(sparsification_cfg, dict) and sparsification_cfg.get("enable") is True
    compression_method = str(sparsification_cfg.get("method", "global_topk")).lower() if isinstance(sparsification_cfg, dict) else "global_topk"

    if ckks_enabled and aggregation not in LINEAR_AGGREGATIONS:
        allowed = ", ".join(LINEAR_AGGREGATIONS)
        errors.append(f"CKKS is only compatible with linear aggregations: {allowed}")
    if ckks_enabled and sparsification_enabled:
        errors.append("CKKS homomorphic encryption and sparsification are not compatible yet")
    if ckks_enabled and secure_agg_enabled:
        errors.append("CKKS homomorphic encryption and secure aggregation masking cannot be enabled together")
    if secure_agg_enabled and aggregation not in EQUAL_WEIGHT_AGGREGATIONS:
        allowed = ", ".join(EQUAL_WEIGHT_AGGREGATIONS)
        errors.append(f"Secure aggregation masking is only compatible with equal-weight aggregations: {allowed}")
    if secure_agg_enabled and sparsification_enabled:
        errors.append("Secure aggregation masking and compression are not compatible yet")
    if sparsification_enabled and aggregation not in LINEAR_AGGREGATIONS:
        allowed = ", ".join(LINEAR_AGGREGATIONS)
        errors.append(f"Sparsification is only compatible with linear aggregations: {allowed}")
    if sparsification_enabled and compression_method not in COMPRESSION_METHODS:
        allowed = ", ".join(COMPRESSION_METHODS)
        errors.append(f"Unsupported compression method: {compression_method}. Allowed methods: {allowed}")
    if dp_enabled:
        clipping_norm = dp_cfg.get("clipping_norm", 1.0)
        noise_multiplier = dp_cfg.get("noise_multiplier", 0.0)
        if isinstance(clipping_norm, bool) or not isinstance(clipping_norm, (int, float)) or clipping_norm <= 0:
            errors.append("Differential privacy clipping_norm must be greater than 0")
        if isinstance(noise_multiplier, bool) or not isinstance(noise_multiplier, (int, float)) or noise_multiplier < 0:
            errors.append("Differential privacy noise_multiplier must be greater than or equal to 0")

    return errors


def capabilities_payload() -> dict[str, Any]:
    return {
        "datasets": list_dataset_names(),
        "distributions": ["iid", "non_iid"],
        "models": ["Auto", *list_model_names()],
        "dataset_model_compatibility": {
            spec.name: list(spec.compatible_models) for spec in DATASET_SPECS
        },
        "dataset_defaults": {
            spec.name: {
                "default_model": spec.default_model,
                "modality": spec.modality,
                "task_type": spec.task_type,
                "input_shape": list(spec.input_shape),
                "num_classes": spec.num_classes,
                "requires_local_data": spec.requires_local_data,
            }
            for spec in DATASET_SPECS
        },
        "aggregations": list(EXECUTABLE_AGGREGATIONS),
        "privacy": {
            "homomorphic_encryption": {
                "methods": ["ckks"],
                "compatible_aggregations": list(LINEAR_AGGREGATIONS),
                "incompatible_with": ["compression.sparsification", "privacy.secure_aggregation"],
            },
            "differential_privacy": {
                "methods": ["clip_and_noise"],
                "compatible_aggregations": list(EXECUTABLE_AGGREGATIONS),
                "compatible_with": [
                    "privacy.homomorphic_encryption",
                    "privacy.secure_aggregation",
                    "compression.sparsification",
                ],
            },
            "secure_aggregation": {
                "methods": ["additive_masking_simulation"],
                "compatible_aggregations": list(EQUAL_WEIGHT_AGGREGATIONS),
                "incompatible_with": ["privacy.homomorphic_encryption", "compression.sparsification"],
            },
        },
        "compression": {
            "sparsification": {
                "methods": list(COMPRESSION_METHODS),
                "compatible_aggregations": list(LINEAR_AGGREGATIONS),
                "incompatible_with": ["privacy.homomorphic_encryption", "privacy.secure_aggregation"],
            }
        },
        "metrics": {
            "global_results": ["rounds", "global_loss", "global_accuracy"],
            "client_results": ["train_loss", "train_acc", "test_loss", "test_acc"],
        },
    }
