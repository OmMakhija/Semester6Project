import functools
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

def build_model(num_classes=2, pretrained=True):
    weights    = "DEFAULT" if pretrained else None
    model      = ssdlite320_mobilenet_v3_large(weights=weights)
    num_anchors= model.anchor_generator.num_anchors_per_location()
    norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    in_channels= _get_in_channels(model)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors,
        num_classes=num_classes, norm_layer=norm_layer,
    )
    return model

def _get_in_channels(model):
    in_channels = []
    for module in model.head.regression_head.module_list:
        for layer in module.modules():
            if isinstance(layer, torch.nn.Conv2d):
                in_channels.append(layer.in_channels)
                break
    return in_channels

def load_model(weights_path, num_classes=2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model, device
