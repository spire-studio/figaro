from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fl_core.simulation_registry import (
    DatasetSpec,
    canonical_dataset_name,
    get_dataset_spec,
    list_dataset_names,
)


def _safe_dataset_class(name: str):
    return getattr(torchvision.datasets, name, None)


def _image_transform(input_shape: tuple[int, int, int], mean: tuple[float, ...] | None = None, std: tuple[float, ...] | None = None):
    channels, height, width = input_shape
    steps: list[Any] = []
    if (height, width) not in {(28, 28), (32, 32)}:
        steps.append(transforms.Resize((height, width)))
    steps.append(transforms.Grayscale(num_output_channels=channels) if channels == 1 else transforms.Lambda(lambda image: image.convert("RGB")))
    steps.append(transforms.ToTensor())
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def _hash_text_to_vector(text: str, feature_dim: int) -> np.ndarray:
    vector = np.zeros(feature_dim, dtype=np.float32)
    for token in text.lower().split():
        if not token:
            continue
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        vector[int(digest[:8], 16) % feature_dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


class CsvTextClassificationDataset(Dataset):
    def __init__(self, path: Path, *, feature_dim: int):
        if not path.exists():
            raise FileNotFoundError(f"Text dataset file not found: {path}")
        self.samples: list[tuple[np.ndarray, int]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    label = int(row[0])
                except ValueError:
                    continue
                if label > 0:
                    label -= 1
                text = " ".join(row[1:])
                self.samples.append((_hash_text_to_vector(text, feature_dim), label))
        if not self.samples:
            raise ValueError(f"No text samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        data, label = self.samples[index]
        return torch.from_numpy(data), int(label)


class ShakespeareDataset(Dataset):
    def __init__(self, path: Path, *, sequence_length: int = 80, vocab_size: int = 128):
        if not path.exists():
            raise FileNotFoundError(f"Shakespeare text file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        encoded = np.array([min(ord(ch), vocab_size - 1) for ch in text], dtype=np.int64)
        if len(encoded) <= sequence_length:
            raise ValueError(f"Shakespeare file {path} is too short for sequence_length={sequence_length}")
        self.x = []
        self.y = []
        for start in range(0, len(encoded) - sequence_length):
            end = start + sequence_length
            self.x.append(encoded[start:end])
            self.y.append(int(encoded[end]))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.x[index]), self.y[index]


class LeafFEMNISTDataset(Dataset):
    def __init__(self, root: Path, split: str):
        files = sorted((root / split).glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"FEMNIST LEAF files not found. Expected JSON files under {root / split}"
            )
        data: list[np.ndarray] = []
        labels: list[int] = []
        for path in files:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            user_data = loaded.get("user_data", {})
            for user_payload in user_data.values():
                xs = user_payload.get("x", [])
                ys = user_payload.get("y", [])
                for x, y in zip(xs, ys):
                    arr = np.asarray(x, dtype=np.float32).reshape(1, 28, 28)
                    data.append(arr)
                    labels.append(int(y))
        if not data:
            raise ValueError(f"No FEMNIST samples found under {root / split}")
        self.data = np.stack(data, axis=0)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.data[index]), int(self.labels[index])


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: Path, transform, class_to_idx: dict[str, int] | None = None):
        val_dir = root / "val"
        annotation_path = val_dir / "val_annotations.txt"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Tiny-ImageNet val annotations not found: {annotation_path}")
        if class_to_idx is None:
            wnids = _read_tiny_imagenet_wnids(root)
            class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.class_to_idx = class_to_idx
        self.samples: list[tuple[Path, int]] = []
        with annotation_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[:2]
                if wnid not in self.class_to_idx:
                    continue
                self.samples.append((val_dir / "images" / image_name, self.class_to_idx[wnid]))
        self.transform = transform
        if not self.samples:
            raise ValueError(f"No Tiny-ImageNet validation samples found under {val_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label


def _read_tiny_imagenet_wnids(root: Path) -> list[str]:
    wnids_path = root / "wnids.txt"
    if wnids_path.exists():
        return [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_dir = root / "train"
    if train_dir.exists():
        return sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    raise FileNotFoundError(f"Tiny-ImageNet class list not found under {root}")


class DatasetLoader:
    def __init__(self, data_dir: str = "./datasets"):
        self.data_dir = data_dir
        self.supported_datasets = list_dataset_names()
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_configs = self._build_dataset_configs()

    def _build_dataset_configs(self) -> dict[str, dict[str, Any]]:
        return {
            "MNIST": {
                "dataset_class": torchvision.datasets.MNIST,
                "style": "train_bool",
                "transform": _image_transform((1, 28, 28), (0.1307,), (0.3081,)),
            },
            "EMNIST": {
                "dataset_class": torchvision.datasets.EMNIST,
                "style": "emnist",
                "split": "balanced",
                "transform": _image_transform((1, 28, 28), (0.1751,), (0.3332,)),
            },
            "Fashion-MNIST": {
                "dataset_class": torchvision.datasets.FashionMNIST,
                "style": "train_bool",
                "transform": _image_transform((1, 28, 28), (0.2860,), (0.3530,)),
            },
            "CIFAR-10": {
                "dataset_class": torchvision.datasets.CIFAR10,
                "style": "train_bool",
                "transform": _image_transform((3, 32, 32), (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            },
            "CIFAR-100": {
                "dataset_class": torchvision.datasets.CIFAR100,
                "style": "train_bool",
                "transform": _image_transform((3, 32, 32), (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            },
            "Country211": {
                "dataset_class": _safe_dataset_class("Country211"),
                "style": "split",
                "train_split": "train",
                "test_split": "test",
                "transform": _image_transform((3, 64, 64)),
            },
            "Flowers102": {
                "dataset_class": _safe_dataset_class("Flowers102"),
                "style": "split",
                "train_split": "train",
                "test_split": "test",
                "transform": _image_transform((3, 64, 64)),
            },
            "GTSRB": {
                "dataset_class": _safe_dataset_class("GTSRB"),
                "style": "split",
                "train_split": "train",
                "test_split": "test",
                "transform": _image_transform((3, 64, 64)),
            },
            "Stanford Cars": {
                "dataset_class": _safe_dataset_class("StanfordCars"),
                "style": "split",
                "train_split": "train",
                "test_split": "test",
                "transform": _image_transform((3, 64, 64)),
            },
        }

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset_name = canonical_dataset_name(dataset_name)
        spec = get_dataset_spec(dataset_name)
        if spec is None:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {self.supported_datasets}")

        already_cached = self.check_dataset_exists(dataset_name)
        if already_cached:
            print(f"DATASET_READY: {dataset_name} (cached at {self.data_dir})", flush=True)
        else:
            print(f"DATASET_DOWNLOAD_START: {dataset_name} (first-time load, this may take several minutes)", flush=True)

        train_dataset, test_dataset = self._create_datasets(spec)

        if not already_cached:
            print(f"DATASET_DOWNLOAD_DONE: {dataset_name}", flush=True)
        print(f"DATASET_LOADED: {dataset_name} train={len(train_dataset)} test={len(test_dataset)}", flush=True)

        train_data, train_labels = self._dataset_to_numpy(train_dataset)
        test_data, test_labels = self._dataset_to_numpy(test_dataset)
        return train_data, train_labels, test_data, test_labels

    def _create_datasets(self, spec: DatasetSpec):
        if spec.loader == "torchvision":
            return self._create_torchvision_datasets(spec)
        if spec.loader == "tiny_imagenet":
            return self._create_tiny_imagenet_datasets(spec)
        if spec.loader == "local_image_folder":
            return self._create_local_image_folder_datasets(spec)
        if spec.loader == "leaf_femnist":
            root = Path(self.data_dir) / spec.name
            return LeafFEMNISTDataset(root, "train"), LeafFEMNISTDataset(root, "test")
        if spec.loader == "csv_text":
            root = Path(self.data_dir) / spec.name.replace(" ", "_")
            if not root.exists():
                root = Path(self.data_dir) / spec.name
            feature_dim = int(spec.metadata.get("feature_dim", spec.input_shape[0]))
            return (
                CsvTextClassificationDataset(root / "train.csv", feature_dim=feature_dim),
                CsvTextClassificationDataset(root / "test.csv", feature_dim=feature_dim),
            )
        if spec.loader == "shakespeare":
            root = Path(self.data_dir) / spec.name
            seq_len = int(spec.metadata.get("sequence_length", spec.input_shape[0]))
            vocab_size = int(spec.metadata.get("vocab_size", spec.num_classes))
            return (
                ShakespeareDataset(root / "train.txt", sequence_length=seq_len, vocab_size=vocab_size),
                ShakespeareDataset(root / "test.txt", sequence_length=seq_len, vocab_size=vocab_size),
            )
        raise ValueError(f"Unsupported dataset loader: {spec.loader}")

    def _create_torchvision_datasets(self, spec: DatasetSpec):
        config = self.dataset_configs.get(spec.name)
        if not config:
            raise ValueError(f"Missing torchvision config for dataset: {spec.name}")
        dataset_class = config.get("dataset_class")
        if dataset_class is None:
            raise ValueError(f"torchvision does not provide dataset class for {spec.name}")
        transform = config["transform"]
        style = config["style"]
        if style == "train_bool":
            return (
                dataset_class(root=self.data_dir, train=True, download=True, transform=transform),
                dataset_class(root=self.data_dir, train=False, download=True, transform=transform),
            )
        if style == "emnist":
            split = config.get("split", "balanced")
            return (
                dataset_class(root=self.data_dir, split=split, train=True, download=True, transform=transform),
                dataset_class(root=self.data_dir, split=split, train=False, download=True, transform=transform),
            )
        if style == "split":
            return (
                dataset_class(root=self.data_dir, split=config["train_split"], download=True, transform=transform),
                dataset_class(root=self.data_dir, split=config["test_split"], download=True, transform=transform),
            )
        raise ValueError(f"Unsupported torchvision loader style: {style}")

    def _create_tiny_imagenet_datasets(self, spec: DatasetSpec):
        root = Path(self.data_dir) / "Tiny-ImageNet"
        if not root.exists():
            root = Path(self.data_dir) / "tiny-imagenet-200"
        transform = _image_transform(spec.input_shape)
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet train directory not found: {train_dir}")
        train_dataset = torchvision.datasets.ImageFolder(str(train_dir), transform=transform)
        test_dataset = TinyImageNetValDataset(root, transform, train_dataset.class_to_idx)
        return train_dataset, test_dataset

    def _create_local_image_folder_datasets(self, spec: DatasetSpec):
        root = Path(self.data_dir) / spec.name
        if not root.exists():
            root = Path(self.data_dir) / spec.name.lower()
        train_dir = root / "train"
        test_dir = root / "test"
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(
                f"{spec.name} expects ImageFolder layout under {root}: train/<class>/*.jpg and test/<class>/*.jpg"
            )
        transform = _image_transform(spec.input_shape)
        return (
            torchvision.datasets.ImageFolder(str(train_dir), transform=transform),
            torchvision.datasets.ImageFolder(str(test_dir), transform=transform),
        )

    def _dataset_to_numpy(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        data_list = []
        labels_list = []
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        for batch_data, batch_labels in dataloader:
            data_list.append(batch_data.numpy())
            labels_list.append(batch_labels.numpy())
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return data, labels

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        dataset_name = canonical_dataset_name(dataset_name)
        spec = get_dataset_spec(dataset_name)
        if spec is None:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        info = {
            "name": spec.name,
            "num_classes": spec.num_classes,
            "input_shape": spec.input_shape,
            "modality": spec.modality,
            "task_type": spec.task_type,
            "default_model": spec.default_model,
            "compatible_models": list(spec.compatible_models),
            "supported": True,
        }
        info.update(spec.metadata)
        return info

    def list_supported_datasets(self) -> list:
        return self.supported_datasets.copy()

    def check_dataset_exists(self, dataset_name: str) -> bool:
        dataset_name = canonical_dataset_name(dataset_name)
        if dataset_name not in self.supported_datasets:
            return False

        dataset_dir_map = {
            "CIFAR-10": "cifar-10-batches-py",
            "CIFAR-100": "cifar-100-python",
            "MNIST": "MNIST",
            "EMNIST": "EMNIST",
            "Fashion-MNIST": "FashionMNIST",
            "Tiny-ImageNet": "Tiny-ImageNet",
            "FEMNIST": "FEMNIST",
            "AG News": "AG_News",
            "Sogou News": "Sogou_News",
            "Shakespeare": "Shakespeare",
        }
        dataset_dir = os.path.join(self.data_dir, dataset_dir_map.get(dataset_name, dataset_name))
        return os.path.exists(dataset_dir)


class DataManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config.get("data_dir", "./datasets")
        self.loader = DatasetLoader(self.data_dir)

    def load_dataset(self, dataset_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dataset_name is None:
            dataset_name = self.config.get("name", "CIFAR-10")
        return self.loader.load_dataset(dataset_name)

    def get_dataset_info(self, dataset_name: str = None) -> Dict[str, Any]:
        if dataset_name is None:
            dataset_name = self.config.get("name", "CIFAR-10")
        return self.loader.get_dataset_info(dataset_name)
