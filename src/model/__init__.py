from torchvision import models
from torchvision.models import VGG19_Weights, VGG16_Weights


def get_style_transfer_model(model_name="vgg19"):
    assert model_name in ["vgg16", "vgg19"]

    # Get the whole model but the classifier
    if model_name == "vgg19":
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    else:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

    # Freeze all the parameters. We will optimize the output image, not the model.
    for param in model.parameters():
        param.requires_grad_(False)
    return model
