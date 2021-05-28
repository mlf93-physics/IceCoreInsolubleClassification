import matplotlib.pyplot as plt
import torch
import utilities as utils
from utilities.constants import *


def main():
    dataset = utils.ImageDataset(file_name='grim.csv',
        root_dir=PATH_TO_TRAIN, transform_enabled=True)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=IMP_BATCH_SIZE, shuffle=True)

    n_batches = 2

    images = []
    for _ in range(n_batches):
        batch = next(iter(train_dataloader))
        images.extend([batch[i, :, :, :] for i in range(IMP_BATCH_SIZE)])

    utils.plot_images(images=images)
        

if __name__ == '__main__':
    main()

    plt.show()