"""Run the trained Fashion-MNIST CNN on one image or a folder of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.model import FashionCNN


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


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a checkpoint and create the CNN with its saved normalization."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    architecture = checkpoint.get("architecture", {})
    dropout = architecture.get("dropout", 0.3)

    model = FashionCNN(dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalization = checkpoint.get(
        "normalization",
        {"mean": 0.28620901279348543, "std": 0.3531597507360165},
    )

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[normalization["mean"]],
                std=[normalization["std"]],
            ),
        ]
    )

    return model, transform


@torch.no_grad()
def predict_image(
    model: FashionCNN,
    transform,
    image_path: Path,
    device: torch.device,
):
    """Return the predicted class and confidence for one image."""

    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    class_id = int(probabilities.argmax())

    return CLASS_NAMES[class_id], float(probabilities[class_id])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Fashion-MNIST-style images with FashionCNN."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to one image.")
    source.add_argument("--folder", type=Path, help="Folder with images.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/fashion_mnist/baseline/checkpoints/best_model.pt"
        ),
        help="Path to a saved model checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_checkpoint(args.checkpoint, device)

    if args.image is not None:
        image_paths = [args.image]
    else:
        supported_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = sorted(
            path
            for path in args.folder.iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )

    if not image_paths:
        raise FileNotFoundError("No supported images were found.")

    for image_path in image_paths:
        class_name, confidence = predict_image(
            model=model,
            transform=transform,
            image_path=image_path,
            device=device,
        )
        print(f"{image_path}\t{class_name}\t{confidence:.4f}")


if __name__ == "__main__":
    main()
