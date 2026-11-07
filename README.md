# Fashion-MNIST CNN baseline

Проект содержит воспроизводимый baseline-эксперимент по классификации изображений Fashion-MNIST с помощью небольшой сверточной нейронной сети CNN.

## Что уже сделано

- Fashion-MNIST разделён на train, validation и test:
  - train — 54 000 изображений;
  - validation — 6 000 изображений;
  - test — 10 000 изображений.
- Разбиение выполнено стратифицированно с фиксированным `seed = 42`.
- Реализована архитектура CNN из `CNN.css`.
- Используется нормализация, рассчитанная только по обучающей части.
- Обучена модель на полной train-выборке.
- Обучена базовая модель на подвыборке F-A, содержащей 20% train-данных.
- Для F-A/F-B сохранены индексы разбиения.
- Сохранены checkpoint, история обучения, метрики по классам и confusion matrix.

## Результаты

| Эксперимент | Validation accuracy | Test accuracy | Macro-F1 |
|---|---:|---:|---:|
| Полная train-выборка, 54 000 изображений | 92.78% | 91.62% | 91.56% |
| F-A, 10 800 изображений | 90.13% | 88.85% | 88.79% |

Для полной модели наиболее сложным классом оказался `Shirt` — accuracy 72.1%. Наиболее частая ошибка — различение классов `T-shirt/top` и `Shirt`.

## Структура проекта

```text
configs/       конфигурация baseline-эксперимента
experiments/   checkpoint, метрики, графики и сохранённые индексы
notebooks/     исследовательский ноутбук с полным экспериментом
src/           загрузка данных, модель, обучение и предсказание
data/          локальный датасет; скачивается автоматически и не хранится в Git
```

## Установка

Из корневой папки репозитория:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если PowerShell запрещает активацию окружения, можно запускать команды напрямую через `.venv\\Scripts\\python.exe`.

## Запуск основного эксперимента

Откройте ноутбук:

```text
notebooks/01_fashion_mnist_exploration.ipynb
```

Выберите kernel из окружения `.venv` и выполните ячейки сверху вниз. Fashion-MNIST будет автоматически скачан в `data/fashion_mnist/` при первом запуске.

Ноутбук рассчитывает:

- разбиение train/validation/test;
- нормализацию;
- обучение полной baseline-модели;
- разбиение на F-A и F-B;
- обучение F-A baseline;
- accuracy, Macro-F1, метрики по классам и confusion matrix.

## Проверка модели на собственных изображениях

Checkpoint полной baseline-модели находится здесь:

```text
experiments/fashion_mnist/baseline/checkpoints/best_model.pt
```

Одно изображение:

```powershell
python -m src.predict --image path\to\image.png
```

Папка с изображениями:

```powershell
python -m src.predict --folder path\to\images
```

Можно указать checkpoint F-A:

```powershell
python -m src.predict `
  --checkpoint experiments/fashion_mnist/fa_baseline/checkpoints/best_model.pt `
  --folder path\to\images
```

Скрипт преобразует изображение в оттенки серого, изменяет размер до `28 x 28`, применяет сохранённую нормализацию и выводит имя класса и confidence модели.

Важно: модель обучена на изображениях, похожих на Fashion-MNIST — один предмет одежды на чёрно-белом изображении размером `28 x 28`. Для произвольных фотографий результат может быть низкого качества без дополнительной подготовки данных.

## Аппаратное окружение исходного запуска

- Python 3.10.7;
- PyTorch 2.14.1+cpu;
- torchvision 0.29.1+cpu;
- устройство: CPU;
- обучаемых параметров модели: 105 962.

Точные параметры обучения находятся в `configs/fashion_mnist_baseline.json`.
