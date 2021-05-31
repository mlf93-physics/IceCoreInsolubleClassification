__all__ = [
    'PATH_TO_TRAIN',
    'IMAGE_WIDTH',
    'IMAGE_HEIGHT',
    'IMP_BATCH_SIZE',
    'NUM_CLASSES',
    'CLASSES',
    'NUM_WORKERS',
    'SEED'
    ]

import pathlib as pl
import numpy as np

PATH_TO_TRAIN = pl.Path('F:/Data_IceCoreInsolubleClassification/train/')
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

NUM_CLASSES = 6
# Classes: 'camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub'
CLASSES = np.array([1, 2, 3, 4, 5, 6])

IMP_BATCH_SIZE = 4
NUM_WORKERS = 0

SEED = 197