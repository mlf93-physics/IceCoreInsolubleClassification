__all__ = [
    'IMAGE_WIDTH',
    'IMAGE_HEIGHT',
    'NUM_CLASSES',
    'CLASSES',
    'SEED',
    'TRANSFORM_IMG'
    ]

import pathlib as pl
import numpy as np
import torchvision.transforms as tv_transforms

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

NUM_CLASSES = 6
# Classes: 'camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub'
CLASSES = np.array([1, 2, 3, 4, 5, 6])

SEED = 197

TRANSFORM_IMG = tv_transforms.Compose([
    tv_transforms.Resize([IMAGE_WIDTH, IMAGE_HEIGHT],
        interpolation=tv_transforms.InterpolationMode.BILINEAR),
    tv_transforms.Normalize(mean=[60], std=[30], inplace=True),
])