import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Компактная CNN для классификации Fashion-MNIST.

    Вход:
        [batch_size, 1, 28, 28]

    Выход:
        [batch_size, 10]

    Выход модели — logits.
    Softmax внутри модели не используется.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # [batch_size, 1, 28, 28]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            # [batch_size, 16, 28, 28]

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # [batch_size, 16, 14, 14]

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            # [batch_size, 32, 14, 14]

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            # [batch_size, 32, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # [batch_size, 32 * 7 * 7]
            # [batch_size, 1568]

            nn.Linear(
                in_features=32 * 7 * 7,
                out_features=64,
            ),
            # [batch_size, 64]

            nn.ReLU(),

            nn.Dropout(p=dropout),

            nn.Linear(
                in_features=64,
                out_features=10,
            ),
            # [batch_size, 10]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        x:
            Тензор изображений размера
            [batch_size, 1, 28, 28].

        Возвращает:
            Десять logits для каждого изображения.
        """

        x = self.features(x)
        x = self.classifier(x)

        return x


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Считает количество обучаемых параметров модели.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )