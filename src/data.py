from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST

import json
import torch

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

FashionMNIST.mirrors = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
]


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# Корневая папка проекта diplom_project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Папка, куда будет скачан Fashion-MNIST
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fashion_mnist"


def load_fashion_mnist(
    data_dir: str | Path = DEFAULT_DATA_DIR,
):
    """
    Загружает обучающую и тестовую части Fashion-MNIST.

    Параметры:
        data_dir: папка для хранения датасета.

    Возвращает:
        train_dataset: обучающая выборка из 60000 изображений.
        test_dataset: тестовая выборка из 10000 изображений.
    """

    data_dir = Path(data_dir)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset

def create_train_validation_split(
    train_dataset,
    validation_fraction: float = 0.10,
    seed: int = 42,
    output_dir: str | Path = (
        PROJECT_ROOT
        / "experiments"
        / "fashion_mnist"
        / "baseline"
    ),
    force_new_split: bool = False,
):
    """
    Делит полную обучающую выборку на train и validation.

    Полная обучающая выборка Fashion-MNIST:
        60000 изображений

    После разбиения:
        train: 54000
        validation: 6000
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir / "split_indices.npz"
    metadata_path = output_dir / "split_metadata.json"

    # Получаем настоящие метки всех 60000 изображений
    targets = train_dataset.targets.cpu().numpy()

    dataset_size = len(train_dataset)
    all_indices = np.arange(dataset_size)

    # Если разбиение уже сохранено, используем его повторно
    if (
        split_path.exists()
        and metadata_path.exists()
        and not force_new_split
    ):
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        same_parameters = (
            metadata["seed"] == seed
            and metadata["dataset_size"] == dataset_size
            and abs(
                metadata["validation_fraction"]
                - validation_fraction
            ) < 1e-12
        )

        if same_parameters:
            with np.load(split_path) as saved_split:
                train_indices = saved_split["train_indices"]
                validation_indices = saved_split["validation_indices"]

            print("Используется сохранённое разбиение.")
        else:
            raise ValueError(
                "Параметры сохранённого разбиения отличаются. "
                "Для создания нового разбиения используй "
                "force_new_split=True."
            )

    else:
        # Стратифицированное разбиение:
        # пропорции классов сохраняются в train и validation
        train_indices, validation_indices = train_test_split(
            all_indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=targets,
        )

        # Сортировка не обязательна для обучения,
        # но делает сохранённые индексы аккуратнее
        train_indices = np.sort(train_indices)
        validation_indices = np.sort(validation_indices)

        # Сохраняем индексы
        np.savez(
            split_path,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

        # Сохраняем описание разбиения
        train_class_counts = np.bincount(
            targets[train_indices],
            minlength=10,
        )

        validation_class_counts = np.bincount(
            targets[validation_indices],
            minlength=10,
        )

        metadata = {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "dataset_size": dataset_size,
            "train_size": int(len(train_indices)),
            "validation_size": int(len(validation_indices)),
            "train_class_counts": train_class_counts.tolist(),
            "validation_class_counts": validation_class_counts.tolist(),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("Создано новое разбиение.")

    # Subset не копирует изображения.
    # Он хранит только индексы элементов исходного датасета.
    train_subset = Subset(
        train_dataset,
        train_indices.tolist(),
    )

    validation_subset = Subset(
        train_dataset,
        validation_indices.tolist(),
    )

    return (
        train_subset,
        validation_subset,
        train_indices,
        validation_indices,
    )

def calculate_train_mean_std(
    full_train_dataset,
    train_indices,
    batch_size: int = 1024,
):
    """
    Вычисляет mean и std только по train-части.

    full_train_dataset:
        Полный Fashion-MNIST train из 60000 изображений.

    train_indices:
        Индексы 54000 изображений, выбранных для train.

    batch_size:
        Сколько изображений обрабатывать за один раз.
    """

    # Исходные изображения хранятся в формате uint8
    # и имеют значения от 0 до 255.
    raw_data = full_train_dataset.data

    indices = torch.as_tensor(
        train_indices,
        dtype=torch.long,
    )

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for start in range(0, len(indices), batch_size):
        end = start + batch_size

        batch_indices = indices[start:end]

        # Берём только изображения из train
        batch = raw_data[batch_indices]

        # Переводим значения 0-255 в 0-1
        batch = batch.float().div(255.0)

        pixel_sum += batch.sum().item()
        pixel_squared_sum += (batch * batch).sum().item()
        pixel_count += batch.numel()

    mean = pixel_sum / pixel_count

    variance = (
        pixel_squared_sum / pixel_count
        - mean**2
    )

    # Защита от очень маленького отрицательного
    # значения из-за погрешности вычислений.
    variance = max(variance, 0.0)

    std = variance**0.5

    return mean, std

def create_normalized_transform(
    mean: float,
    std: float,
):
    """
    Создаёт преобразование:
    изображение → Tensor → Normalize.
    """

    if std <= 0:
        raise ValueError(
            "Стандартное отклонение должно быть больше нуля."
        )

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(mean,),
            std=(std,),
        ),
    ])