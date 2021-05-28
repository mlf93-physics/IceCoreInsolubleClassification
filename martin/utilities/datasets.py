import torch
import torchvision
import torchvision.transforms as tv_transforms
import utilities as utils
from utilities.constants import *

class ImageDataset(torch.utils.data.Dataset):
    """Image dataset."""

    def __init__(self, file_name=None, root_dir=None, transform_enabled=False):
        """
        Args:
            file_name (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = utils.import_csv_file(file_name)
        self.root_dir = root_dir
        self.transform = torch.nn.Sequential(
            tv_transforms.Resize([IMAGE_WIDTH, IMAGE_HEIGHT],
                interpolation=tv_transforms.InterpolationMode.BILINEAR),
            # tv_transforms.Normalize(100, 100, inplace=True),
            # tv_transforms.Normalize(
            #     mean=[100],
            #     std=[20],
            # )
        )
        self.transform_enabled = transform_enabled

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.root_dir / self.data_frame['imgpaths'].iloc[idx]
        image = torchvision.io.read_image(str(img_path))

        if self.transform_enabled:
            image = self.transform(image)

        return image