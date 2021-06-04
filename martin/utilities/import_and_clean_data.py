import pandas as pd
import torchvision
import torch
import pathlib as pl
import matplotlib.pyplot as plt
from utilities.constants import *

def get_folder_statistics(args, folder=None):
    folder_path = pl.Path(args.train_path) / folder
    file_list = list(folder_path.glob('*'))
    num_files = len(file_list)
    print(f'Number of files in folder "{folder}"', num_files)

def import_csv_file(args, file_name=None, nrows=None):
    file_path = pl.Path(args.train_path) / file_name
    data = pd.read_csv(file_path, nrows=nrows)

    return data

def write_to_csv_file(args, file_name=None, data_frame=None):
    file_path = pl.Path(args.train_path) / file_name
    data_frame.to_csv(file_path, index=False)

def clean_up_imgpaths(args):
    files = list(pl.Path(args.train_path).glob('*.csv'))
    
    for file in files:
        print('File to clean up:', file.name)
        print('Press enter to continue...')
        input()
        data = import_csv_file(file_name=file.name)
        data['imgpaths'] = data['imgpaths'].str.split('train/').str[1]
        write_to_csv_file(file_name=file.name, data_frame=data)

def import_img(path):
    image = torchvision.io.read_image(path)
    image = image.type(torch.FloatTensor)
    return image

def data_loader(args, folder_name=None, batch_size=4):
    imagenet_data = torchvision.datasets.ImageNet(pl.Path(args.train_path) / folder_name)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=args.n_threads)
    return data_loader

if __name__ == '__main__':
    print('Starting clean up function. Press enter to continue')
    input()
    clean_up_imgpaths()