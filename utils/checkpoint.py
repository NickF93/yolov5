# YOLOv5 checkpoint compatibility helpers

import torch


def torch_load_checkpoint(path, map_location=None):
    # PyTorch 2.6 changed torch.load() to weights_only=True by default.
    # YOLOv5 v7 checkpoints store a pickled model object, so trusted .pt
    # checkpoints must be loaded with weights_only=False.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
