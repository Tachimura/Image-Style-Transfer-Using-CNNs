import glob
import os
import random

from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.utils import save_image


def create_dataloader(data_path: str) -> list[str]:
    assert os.path.exists(data_path) and os.path.isdir(data_path), f"Path: {data_path} is not a valid folder path."
    return glob.glob(os.path.join(data_path, "*"))


def select_random_image(dataset: list[str]) -> str:
    return dataset[random.randint(0, len(dataset) - 1)]


def load_image(image_path: str, max_dim: int = 1024) -> Tensor:
    img = read_image(image_path)
    factor = max_dim / max(img.shape)
    img = Resize((round(img.shape[1] * factor), round(img.shape[2] * factor)))(img)
    # Return normalized image between 0, 1
    return (img - img.min()) / (img.max() - img.min())


def save_output_image(output_image: Tensor, image_name: str, output_path: str, extension="jpg"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_image(output_image, os.path.join(output_path, image_name + f".{extension}"))
