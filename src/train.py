import json
from pathlib import Path

import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    Одна эпоха обучения модели.
    """

    # Режим обучения:
    # активируется Dropout и BatchNorm обучается
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in train_loader:
        # Переносим данные на CPU или GPU
        images = images.to(device)
        labels = labels.to(device)

        # Обнуляем градиенты с предыдущего шага
        optimizer.zero_grad()

        # Получаем предсказания модели
        logits = model(images)

        # Вычисляем ошибку
        loss = criterion(logits, labels)

        # Вычисляем градиенты
        loss.backward()

        # Обновляем параметры модели
        optimizer.step()

        # Размер текущей группы изображений
        batch_size = images.size(0)

        # Накапливаем ошибку
        total_loss += loss.item() * batch_size

        # Получаем предсказанный класс
        predictions = logits.argmax(dim=1)

        # Считаем правильные ответы
        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    """
    Оценка модели на validation или test.
    """

    # Режим оценки:
    # Dropout отключается, BatchNorm использует
    # уже рассчитанные статистики
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # Во время оценки градиенты не нужны
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def fit(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device,
    epochs,
    checkpoint_path=None,
    history_path=None,
    checkpoint_metadata=None,
):
    """
    Полный цикл обучения модели.

    Лучшая модель выбирается по validation accuracy.
    """

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = float("-inf")
    best_epoch = None

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    for epoch in range(1, epochs + 1):

        # Обучение на train
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Проверка на validation
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        # Сохраняем результаты текущей эпохи
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(
            validation_accuracy
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        # Проверяем, стала ли модель лучше
        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            if checkpoint_path is not None:

                checkpoint = {
                    "epoch": epoch,

                    # Веса модели
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    # Состояние оптимизатора Adam
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),

                    # Метрики лучшей модели
                    "validation_loss": validation_loss,
                    "validation_accuracy": (
                        validation_accuracy
                    ),

                    # История на момент сохранения
                    "history": history,
                }

                # Добавляем seed, нормализацию,
                # архитектуру и параметры обучения
                if checkpoint_metadata is not None:
                    checkpoint.update(
                        checkpoint_metadata
                    )

                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

    # Сохраняем полную историю после всех эпох
    if history_path is not None:
        history_path = Path(history_path)

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_path.write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("Лучшая эпоха:", best_epoch)
    print(
        "Лучшая validation accuracy:",
        best_validation_accuracy,
    )

    return history