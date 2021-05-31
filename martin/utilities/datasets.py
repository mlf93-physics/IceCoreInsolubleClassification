import numpy as np
import torch
import torchvision
import torchvision.transforms as tv_transforms
import torch.utils.data.sampler as t_data_sampler
import utilities as utils
from utilities.constants import *

class ImageDataset(torch.utils.data.Dataset):
    """Image dataset."""

    def __init__(self, file_name=None, root_dir=None, transform_enabled=False,
            n_datapoints=None):
        """
        Args:
            file_name (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = utils.import_csv_file(file_name, nrows=n_datapoints)
        self.root_dir = root_dir
        self.transform = tv_transforms.Compose([
            tv_transforms.Resize([IMAGE_WIDTH, IMAGE_HEIGHT],
                interpolation=tv_transforms.InterpolationMode.BILINEAR),
            tv_transforms.Normalize(mean=[60], std=[30], inplace=True),
        ])
        self.transform_enabled = transform_enabled

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.root_dir / self.data_frame['imgpaths'].iloc[idx]
        image = torchvision.io.read_image(str(img_path))
        image = image.type(torch.FloatTensor)


        # Get image label
        label = self.data_frame[self.data_frame.columns[-NUM_CLASSES:]].iloc[idx]
        label = np.argwhere(np.array(label) == 1)[0][0] + 1

        if self.transform_enabled:
            image = self.transform(image)

        return image, label

def train_val_dataloader_split(size_dataset, val_frac=0.25):
    indices = range(size_dataset)
    split = int(np.floor(val_frac * size_dataset))

    np.random.seed(SEED)
    np.random.shuffle(indices)

    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = t_data_sampler.SubsetRandomSampler(train_indices)
    val_sampler = t_data_sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler