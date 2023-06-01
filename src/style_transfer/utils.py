import torch
from torch import nn, Tensor


def get_img_features(model: nn.Module, image: Tensor, layers: dict[str, str]):
    """
    Perform a forward and get the features for the set of layers we selected.
    """
    img_features = {}
    layer_features = image
    # We perform the forward for each module in the model and save the features of some layers.
    for name, layer in model._modules.items():
        layer_features = layer(layer_features)
        # We only save the features of layers we are interested in
        if name in layers:
            img_features[layers[name]] = layer_features
    return img_features


def gram_matrix(tensor: Tensor):
    """
    Calculate the Gram Matrix of a given tensor
    """
    c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gm = torch.mm(tensor, tensor.t())
    # Return 'normalized' values
    return gm.div(c * h * w)
