import os

import torch
from tqdm import tqdm

from src.data.dataloader import create_dataloader, select_random_image, save_output_image, load_image
from src.data.yaml_reader import read_config_file
from src.style_transfer.engine import transfer_style
from src.utils import set_seeds
from src.utils.arguments import get_args


def main(args):
    # Read configuration settings
    config = read_config_file(args.config)
    assert config, "Error parsing config.yaml file"

    # Set seeds for reproducibility
    set_seeds(config)

    # Load paths for content and style images
    content_dataset = create_dataloader(config["data"]["content_folder"])
    style_dataset = create_dataloader(config["data"]["style_folder"])

    # Prepare the params for the style-transfer process
    style_transfer_params = {
        "model": "vgg19",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "content_weight": float(config["weights"]["alpha"]),  # alpha 1 per pytorch 1e3 per keras
        "style_weight": float(config["weights"]["beta"]),  # beta 1e6 per pytorch 1e-2 per keras
        "lr": float(config["train"]["lr"]),
        "steps": int(config["train"]["steps"]),
        "img_size": int(config["data"]["img_size"])
    }

    # For each content image, transfer the style of a random style image
    for idx, content_image_path in enumerate(tqdm(content_dataset)):
        print(f"Status: {idx} / {len(content_dataset)}")

        # Read content and a random style image
        content_image = load_image(content_image_path, max_dim=style_transfer_params["img_size"])
        style_image = load_image(select_random_image(style_dataset), max_dim=style_transfer_params["img_size"])

        # Generate the output image
        output_image = transfer_style(content_image, style_image, style_transfer_params)

        # Save the generated image
        content_image_name = content_image_path.split(os.sep)[-1].split(".")[0]
        save_output_image(output_image, f"{content_image_name}", config["data"]["output_folder"])


if __name__ == '__main__':
    main(get_args())
