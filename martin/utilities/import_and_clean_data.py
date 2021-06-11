import numpy as np
import pandas as pd
import torchvision
import torch
import pathlib as pl
import matplotlib.pyplot as plt
from utilities.constants import *

def get_folder_statistics(args, folder=None):
    folder_path = pl.Path(args["train_path"]) / folder
    file_list = list(folder_path.glob('*'))
    num_files = len(file_list)
    print(f'Number of files in folder "{folder}"', num_files)

def import_csv_file(args, file_name=None, nrows=None):
    file_path = pl.Path(args["train_path"]) / file_name
    data = pd.read_csv(file_path, nrows=nrows)

    return data

def write_to_csv_file(args, file_name=None, data_frame=None):
    file_path = pl.Path(args["train_path"]) / file_name
    data_frame.to_csv(file_path, index=False)

def clean_up_imgpaths(args):
    files = list(pl.Path(args["train_path"]).glob('*.csv'))
    
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
    imagenet_data = torchvision.datasets.ImageNet(pl.Path(args["train_path"]) / folder_name)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=args["n_threads"])
    return data_loader

def import_history(path=None):
    path = pl.Path(path)
    files = path.glob('*.txt')
    indices = []
    values = []
    headers = []

    for file in files:
        index = []
        value = []
        with open(str(file), 'r') as temp_file:
            header = temp_file.readline().strip().split(',')

            for iline in temp_file:
                line = iline.split(',')
                index.append(int(line[0]))
                value.append(float(line[1]))
        
        headers.append(header)
        indices.append(index)
        values.append(value)
        
    return headers, indices, values

def import_confusion_matrix(path=None):
    path = pl.Path(path)
    conf_matrices = []

    with open(str(path), 'r') as temp_file:
        line = temp_file.readline().split(',')
        num_classes = len(line)

    with open(str(path), 'r') as temp_file:

        matrix = np.zeros((num_classes, num_classes))
        for i, iline in enumerate(temp_file):
            if i % num_classes == 0 and i > 0:
                conf_matrices.append(matrix)
                matrix = np.zeros((num_classes, num_classes))


            line = iline.strip().split(',')
            for j in range(num_classes):
                matrix[int(i % num_classes), j] = float(line[j])


    return conf_matrices

def import_probs_and_truth(dir=None, data_set='test'):
    dir = pl.Path(dir)

    files = list(dir.glob('*.txt'))
    truth_prefix = data_set + '_truth' 
    prob_prefix = data_set + '_prob' 

    for file in files:
        if truth_prefix in file.stem:
            truth_file = file
        elif prob_prefix in file.stem:
            prob_file = file

    truth = np.genfromtxt(truth_file, delimiter=',')
    prob = np.genfromtxt(prob_file, delimiter=',')

    return truth, prob
