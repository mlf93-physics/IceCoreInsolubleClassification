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
    indices = np.random.randint(0, size_dataset, args["n_datapoints"])
    split = int(np.floor(args["val_frac"] * args["n_datapoints"]))

    # Split dataset
    train_indices = indices[split:]
    val_indices = indices[:split]

    # Define train and validation samplers
    train_sampler = t_data_sampler.SubsetRandomSampler(train_indices)
    val_sampler = t_data_sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler

# class MapDataset(torch.utils.data.Dataset):
#     """
#     Given a dataset, creates a dataset which applies a mapping function
#     to its items (lazily, only when an item is called).

#     Note that data is not cloned/copied from the initial dataset.
#     """

#     def __init__(self, dataset, map_fn):
#         self.dataset = dataset
#         self.map = map_fn

#     def __getitem__(self, index):
#         print(self.dataset)
#         # print('index', index)
#         # print('self.dataset[index]', self.dataset[index])
#         if self.map:     
#             image = self.map(self.dataset[index][0]) 
#         else:     
#             image = self.dataset[index][0]

#         label = self.dataset[index][1]      
#         return image, label

#     def __len__(self):
#         return len(self.dataset)

def train_val_dataloader_split_weighted_subset(train_dataset, args, num_classes=6):
    class_sample_counts = torch.unique(torch.FloatTensor(train_dataset.targets),
        return_counts=True)[1]
    
    weights = 1. / class_sample_counts
    samples_weights = weights[train_dataset.targets]

    if args["n_datapoints"] < 0:
        num_samples = torch.min(class_sample_counts).item()*\
            class_sample_counts.size()[0]
    else:
        num_samples = args["n_datapoints"]
    
    sampler = t_data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=num_samples,
        replacement=False)

    indices = list(sampler)

    # Find split
    split = int(np.floor(args["val_frac"] * num_samples))
    # Split dataset
    train_indices = indices[split:]
    val_indices = indices[:split]

    # train_subdataset = t_data.Subset(train_indices, train_indices)
    # val_subdataset = t_data.Subset(train_indices, val_indices)

    return train_indices, val_indices

def train_val_test_dataloader_weighted_subset(train_dataset, args, num_classes=6):
    class_sample_counts = torch.unique(torch.FloatTensor(train_dataset.targets),
        return_counts=True)[1]
    
    print('class_sample_counts', class_sample_counts)
    
    weights = torch.Tensor([1 for _ in range(num_classes)])
    samples_weights = weights[train_dataset.targets]

    if args["n_datapoints"] < 0:
        # num_samples = torch.min(class_sample_counts).item()*\
        #     class_sample_counts.size()[0]
        num_samples = class_sample_counts.sum().item()
    else:
        num_samples = args["n_datapoints"]
    
    sampler = t_data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=num_samples,
        replacement=False)

    indices = list(sampler)

    # Find split
    split1 = int(np.floor((args["val_frac"] + args["test_frac"]) * num_samples))
    # Split dataset
    train_indices = indices[split1:]
    val_test_indices = indices[:split1]
    split2 = int(np.floor(args["val_frac"]/(args["val_frac"] + args["test_frac"])
        * len(val_test_indices)))
    val_indices = val_test_indices[:split2]
    test_indices = val_test_indices[split2:]

    # train_subdataset = t_data.Subset(train_indices, train_indices)
    # val_subdataset = t_data.Subset(train_indices, val_indices)

    return train_indices, val_indices, test_indices


def define_dataloader(args):
    print('Define dataloader')
    train_dataset = torchvision.datasets.ImageFolder(
        root=args["train_path"], transform=TRAIN_TRANSFORM,
        loader=utils.import_img)

    num_classes = len(train_dataset.classes)
    print('Train classes: ', train_dataset.classes, 'class_to_idx',
        train_dataset.class_to_idx)

    # Get sample indices
    # train_sampler, val_sampler =\
    #     utils.train_val_dataloader_split_random_subset(args)
    # train_indices, val_indices =\
    #     train_val_dataloader_split_weighted_subset(train_dataset, args)

    train_indices, val_indices, test_indices =\
        train_val_test_dataloader_weighted_subset(train_dataset, args, num_classes=num_classes)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args["batch_size"], num_workers=args["n_threads"],
        sampler=train_indices)
    
    val_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args["batch_size"], num_workers=args["n_threads"],
        sampler=val_indices)

    test_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args["batch_size"], num_workers=args["n_threads"],
        sampler=test_indices)

    return train_dataloader, val_dataloader, test_dataloader, num_classes