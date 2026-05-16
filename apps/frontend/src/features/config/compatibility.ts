import { getValueByPath, isRecord, type SchemaNode } from "../simulation/utils";

const DATASET_MODEL_COMPATIBILITY: Record<string, string[]> = {
  MNIST: ["Mclr_Logistic", "LeNet", "DNN"],
  EMNIST: ["Mclr_Logistic", "LeNet", "DNN"],
  FEMNIST: ["Mclr_Logistic", "LeNet", "DNN"],
  "Fashion-MNIST": ["Mclr_Logistic", "LeNet", "DNN"],
  "CIFAR-10": ["Mclr_Logistic", "FedAvgCNN", "DNN", "ResNet18", "AlexNet", "MobileNet", "GoogleNet"],
  "CIFAR-100": ["Mclr_Logistic", "FedAvgCNN", "DNN", "ResNet18", "AlexNet", "MobileNet", "GoogleNet"],
  "AG News": ["TextLogistic", "TextDNN", "TextCNN"],
  "Sogou News": ["TextLogistic", "TextDNN", "TextCNN"],
  "Tiny-ImageNet": ["Mclr_Logistic", "FedAvgCNN", "DNN", "ResNet18", "AlexNet", "MobileNet", "GoogleNet"],
  Country211: ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
  Flowers102: ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
  GTSRB: ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
  Shakespeare: ["CharLSTM"],
  "Stanford Cars": ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
  COVIDx: ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
  Kvasir: ["FedAvgCNN", "DNN", "ResNet18", "ResNet34", "AlexNet", "MobileNet", "GoogleNet"],
};

const DATASET_DEFAULT_MODELS: Record<string, string> = {
  MNIST: "LeNet",
  EMNIST: "LeNet",
  FEMNIST: "LeNet",
  "Fashion-MNIST": "LeNet",
  "CIFAR-10": "FedAvgCNN",
  "CIFAR-100": "FedAvgCNN",
  "AG News": "TextDNN",
  "Sogou News": "TextDNN",
  "Tiny-ImageNet": "FedAvgCNN",
  Country211: "MobileNet",
  Flowers102: "MobileNet",
  GTSRB: "MobileNet",
  Shakespeare: "CharLSTM",
  "Stanford Cars": "MobileNet",
  COVIDx: "MobileNet",
  Kvasir: "MobileNet",
};

const LINEAR_AGGREGATIONS = new Set(["fedavg", "weighted_avg", "simple_avg"]);
const EQUAL_WEIGHT_AGGREGATIONS = new Set(["fedavg", "simple_avg"]);

export type OptionState = {
  disabled: boolean;
  reason?: string;
};

function boolAt(properties: Record<string, unknown>, path: string): boolean {
  return getValueByPath(properties, path) === true;
}

function stringAt(properties: Record<string, unknown>, path: string, fallback = ""): string {
  const value = getValueByPath(properties, path);
  return typeof value === "string" ? value : fallback;
}

export function optionMeta(definition: SchemaNode, option: unknown): Record<string, unknown> {
  const ui = definition.ui;
  if (!isRecord(ui)) return {};
  const options = ui.options;
  if (!isRecord(options)) return {};
  const meta = options[String(option)];
  return isRecord(meta) ? meta : {};
}

export function optionLabel(definition: SchemaNode, option: unknown): string {
  const meta = optionMeta(definition, option);
  return typeof meta.label === "string" && meta.label.trim() ? meta.label : String(option);
}

export function staticOptionDisabled(definition: SchemaNode, option: unknown): boolean {
  return optionMeta(definition, option).disabled === true;
}

export function compatibleModelsForDataset(datasetName: string): string[] {
  return DATASET_MODEL_COMPATIBILITY[datasetName] ?? [];
}

export function defaultModelForDataset(datasetName: string): string | null {
  return DATASET_DEFAULT_MODELS[datasetName] ?? null;
}

export function isModelCompatibleWithDataset(modelName: unknown, datasetName: string): boolean {
  if (modelName === "Auto") return true;
  if (typeof modelName !== "string") return false;
  const models = compatibleModelsForDataset(datasetName);
  return models.length === 0 || models.includes(modelName);
}

export function dynamicOptionState(
  path: string,
  option: unknown,
  properties: Record<string, unknown>,
): OptionState {
  const optionText = String(option);
  const systemMode = stringAt(properties, "system.mode", "simulation");
  const aggregation = stringAt(properties, "federated.aggregation", "fedavg");
  const ckksEnabled = boolAt(properties, "privacy.homomorphic_encryption.enable");
  const dpEnabled = boolAt(properties, "privacy.differential_privacy.enable");
  const compressionEnabled = boolAt(properties, "compression.sparsification.enable");
  const secureAggEnabled = boolAt(properties, "privacy.secure_aggregation.enable");

  if (path === "task.type" && optionText === "llm_peft_sft" && systemMode === "distributed") {
    return {
      disabled: true,
      reason: "LLM PEFT is currently supported in simulation mode only.",
    };
  }

  if (path === "model.name") {
    const datasetName = stringAt(properties, "dataset.name", "CIFAR-10");
    if (!isModelCompatibleWithDataset(optionText, datasetName)) {
      return {
        disabled: true,
        reason: `${optionText} is not compatible with ${datasetName}.`,
      };
    }
  }

  if (path === "federated.aggregation") {
    if ((ckksEnabled || compressionEnabled) && !LINEAR_AGGREGATIONS.has(optionText)) {
      return {
        disabled: true,
        reason: "CKKS and compression only support linear aggregation.",
      };
    }
    if (secureAggEnabled && !EQUAL_WEIGHT_AGGREGATIONS.has(optionText)) {
      return {
        disabled: true,
        reason: "Secure aggregation masking requires equal-weight aggregation.",
      };
    }
  }

  if (path === "privacy.homomorphic_encryption.method" && optionText !== "ckks") {
    return { disabled: true, reason: "Only CKKS is currently executable." };
  }

  if (path === "compression.sparsification.method") {
    if (ckksEnabled) return { disabled: true, reason: "Turn off CKKS before enabling compression." };
    if (secureAggEnabled) return { disabled: true, reason: "Turn off secure aggregation before enabling compression." };
  }

  if (path === "privacy.differential_privacy.enable" && dpEnabled) {
    return { disabled: false };
  }

  void aggregation;
  return { disabled: false };
}

export function dynamicBooleanState(path: string, value: unknown, properties: Record<string, unknown>): OptionState {
  if (value === true) return { disabled: false };

  const aggregation = stringAt(properties, "federated.aggregation", "fedavg");
  const ckksEnabled = boolAt(properties, "privacy.homomorphic_encryption.enable");
  const compressionEnabled = boolAt(properties, "compression.sparsification.enable");
  const secureAggEnabled = boolAt(properties, "privacy.secure_aggregation.enable");

  if (path === "privacy.homomorphic_encryption.enable") {
    if (!LINEAR_AGGREGATIONS.has(aggregation)) {
      return { disabled: true, reason: "CKKS requires fedavg, weighted_avg, or simple_avg." };
    }
    if (compressionEnabled) return { disabled: true, reason: "Turn off compression before enabling CKKS." };
    if (secureAggEnabled) return { disabled: true, reason: "Turn off secure aggregation before enabling CKKS." };
  }

  if (path === "compression.sparsification.enable") {
    if (!LINEAR_AGGREGATIONS.has(aggregation)) {
      return { disabled: true, reason: "Compression requires fedavg, weighted_avg, or simple_avg." };
    }
    if (ckksEnabled) return { disabled: true, reason: "Turn off CKKS before enabling compression." };
    if (secureAggEnabled) return { disabled: true, reason: "Turn off secure aggregation before enabling compression." };
  }

  if (path === "privacy.secure_aggregation.enable") {
    if (!EQUAL_WEIGHT_AGGREGATIONS.has(aggregation)) {
      return { disabled: true, reason: "Secure aggregation requires fedavg or simple_avg." };
    }
    if (ckksEnabled) return { disabled: true, reason: "Turn off CKKS before enabling secure aggregation." };
    if (compressionEnabled) return { disabled: true, reason: "Turn off compression before enabling secure aggregation." };
  }

  return { disabled: false };
}

export function fieldHint(path: string, properties: Record<string, unknown>): string | null {
  if (path === "model.name") {
    const datasetName = stringAt(properties, "dataset.name", "CIFAR-10");
    const models = compatibleModelsForDataset(datasetName);
    if (models.length === 0) return null;
    const defaultModel = defaultModelForDataset(datasetName);
    const defaultText = defaultModel ? ` Default: ${defaultModel}.` : "";
    return `${datasetName} supports: Auto, ${models.join(", ")}.${defaultText}`;
  }

  if (path === "federated.aggregation") {
    const ckksEnabled = boolAt(properties, "privacy.homomorphic_encryption.enable");
    const compressionEnabled = boolAt(properties, "compression.sparsification.enable");
    const secureAggEnabled = boolAt(properties, "privacy.secure_aggregation.enable");
    if (secureAggEnabled) return "Secure aggregation masking allows fedavg or simple_avg.";
    if (ckksEnabled || compressionEnabled) return "CKKS and compression allow fedavg, weighted_avg, or simple_avg.";
  }

  return null;
}
