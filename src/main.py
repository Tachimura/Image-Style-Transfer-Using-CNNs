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
        "model": config["model"],
        "device": torch.device("cuda" if torch.cuda.is_available() and "cuda" in config["device"] else "cpu"),
        "content_weight": float(config["weights"]["alpha"]),
        "style_weight": float(config["weights"]["beta"]),
        "lr": float(config["train"]["lr"]),
        "steps": int(config["train"]["steps"]),
        "print_every": int(config["train"]["print_every"]),
        "img_size":
            config["data"]["img_size"] if isinstance(config["data"]["img_size"], list)
            else (3, int(config["data"]["img_size"]), int(config["data"]["img_size"])),
        "layers": {
            "content": config["layers"]["content"],
            "style": config["layers"]["style"]
        },
        "weights": config["weights"]
    }

    # For each content image, transfer the style of a random style image
    for idx, content_image_path in enumerate(tqdm(content_dataset)):
        print(f"Status: {idx} / {len(content_dataset)}")

        # Read content and a random style image
        content_image = load_image(content_image_path, max_dim=max(style_transfer_params["img_size"]))
        style_image = load_image(select_random_image(style_dataset), max_dim=max(style_transfer_params["img_size"]))

        # Generate the output image
        output_image = transfer_style(content_image, style_image, style_transfer_params)

        # Save the generated image
        content_image_name = content_image_path.split(os.sep)[-1].split(".")[0]
        save_output_image(output_image, f"{content_image_name}", config["data"]["output_folder"])


if __name__ == '__main__':
    main(get_args())
