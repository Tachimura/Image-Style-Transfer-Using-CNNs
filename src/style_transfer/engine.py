import torch
from torch import Tensor, optim, nn
from tqdm import tqdm

from .utils import get_img_features, gram_matrix
from ..model import get_style_transfer_model


def transfer_style(content_image: Tensor, style_image: Tensor, config: dict):
    device = config["device"]
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    # Get style transfer model
    model = get_style_transfer_model(model_name=config["model"])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    alpha = float(config["weights"]["alpha"])
    beta = float(config["weights"]["beta"])
    return _style_transfer_train(model, device, content_image, style_image, alpha, beta, config)


def _style_transfer_train(model: nn.Module, device: torch.device, content_image: Tensor, style_image: Tensor,
                          alpha: float, beta: float, config: dict):
    # We create the output image from white noise.
    target_image = torch.rand((config["img_size"][0], config["img_size"][1], config["img_size"][2]),
                              requires_grad=True, device=device)
    optimizer = optim.Adam([target_image], lr=config["lr"])

    all_layers = dict(config["layers"]["content"].items() | config["layers"]["style"].items())

    # Get content and style features
    content_features = get_img_features(model, content_image, config["layers"]["content"])
    style_features = get_img_features(model, style_image, config["layers"]["style"])
    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for current_step in tqdm(range(config["steps"])):
        # correct the values of updated input image
        with torch.no_grad():
            target_image.clamp_(0, 1)
        optimizer.zero_grad()
        # Perform forward to calculate the target features
        target_features = get_img_features(model, target_image, all_layers)

        # Calculate the content loss as the MSE for each content layer
        content_loss = 0
        for layer in content_features:
            target_feature = target_features[layer]
            content_feature = content_features[layer]
            content_loss += torch.mean((target_feature - content_feature) ** 2)

        # Calculate the style loss
        style_loss = 0
        for layer in style_features:
            target_feature = target_features[layer]
            c, h, w = target_feature.shape

            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]

            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)

            # add to the style loss
            style_loss += layer_style_loss / (c * h * w)

        total_loss = alpha * content_loss + beta * style_loss

        # update your target image
        total_loss.backward()
        optimizer.step()
        if current_step % config["print_every"] == 0:
            print(f"Step {current_step} | Total Loss: {total_loss} -- Content Loss: {content_loss} -- Style Loss: {style_loss}")
    return target_image
