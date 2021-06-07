import numpy as np
import torch
import torchvision
import torchvision.transforms as tv_transforms
import torch.utils.data.sampler as t_data_sampler
import torch.utils.data as t_data
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

def train_val_dataloader_split_random_subset(args):
    # Get size of dataset
    data_frame = utils.import_csv_file(args, file_name='train.csv')
    size_dataset = data_frame.shape[0]

    # Prepare indices for sampler
    np.random.seed(SEED)
    indices = np.random.randint(0, size_dataset, args.n_datapoints)
    split = int(np.floor(args.val_frac * args.n_datapoints))

    # Split dataset
    train_indices = indices[split:]
    val_indices = indices[:split]

    # Define train and validation samplers
    train_sampler = t_data_sampler.SubsetRandomSampler(train_indices)
    val_sampler = t_data_sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler

def train_val_dataloader_split_weighted_subset(train_dataset, args, num_classes=6):
    class_sample_counts = torch.unique(torch.FloatTensor(train_dataset.targets),
        return_counts=True)[1]
    
    weights = 1. / class_sample_counts
    samples_weights = weights[train_dataset.targets]

    if args.n_datapoints < 0:
        num_samples = torch.min(class_sample_counts).item()*\
            class_sample_counts.size()[0]
    else:
        num_samples = args.n_datapoints
    
    sampler = t_data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=num_samples,
        replacement=False)

    indices = list(sampler)

    # Find split
    split = int(np.floor(args.val_frac * num_samples))
    # Split dataset
    train_indices = indices[split:]
    val_indices = indices[:split]

    # Define train and validation samplers
    train_sampler = t_data_sampler.SubsetRandomSampler(train_indices)
    val_sampler = t_data_sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler