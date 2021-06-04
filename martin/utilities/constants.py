__all__ = [
    'IMAGE_WIDTH',
    'IMAGE_HEIGHT',
    'IMP_BATCH_SIZE',
    'NUM_CLASSES',
    'CLASSES',
    'NUM_WORKERS',
    'SEED',
    'TRANSFORM_IMG'
    ]

import pathlib as pl
import numpy as np
import torchvision.transforms as tv_transforms

# PATH_TO_TRAIN = pl.Path('F:/Data_IceCoreInsolubleClassification/train/')
# OUT_PATH = pl.Path('C:/Users/Martin/Documents/FysikUNI/Kandidat/AppliedMachineLearning/final_project/trained_cnns')

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

NUM_CLASSES = 6
# Classes: 'camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub'
CLASSES = np.array([1, 2, 3, 4, 5, 6])

IMP_BATCH_SIZE = 4
NUM_WORKERS = 2

SEED = 197

TRANSFORM_IMG = tv_transforms.Compose([
    tv_transforms.Resize([IMAGE_WIDTH, IMAGE_HEIGHT],
        interpolation=tv_transforms.InterpolationMode.BILINEAR),
    tv_transforms.Normalize(mean=[60], std=[30], inplace=True),
])