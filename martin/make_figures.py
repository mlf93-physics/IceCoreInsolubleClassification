import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import pathlib as pl
import sklearn.metrics as skl_metrics
import utilities as utils
import cnn_setups as cnns
from utilities.constants import *
import seaborn as sb

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default=
    'F:/Data_IceCoreInsolubleClassification/train/')
parser.add_argument('--path', type=str, default='')
parser.add_argument('--cnn_file', type=str, default=None)
parser.add_argument('--plot_type', type=str, default='history')


def history_figure(args):
    headers, indices, values = utils.import_history(
        dir=args['path'])
    
    plt.figure(figsize=(5, 3), constrained_layout=True)
    for i in range(len(headers)):
        print('headers[i]', headers[i])
        # if 'train' in headers[i][1]:
        #     values[i] = np.array(values[i])
        #     headers[i][1] += ' (x10)'
        plt.plot(indices[i], values[i], label=f'{headers[i][1]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def confusion_matrix(args):
    conf_matrix = utils.import_confusion_matrix(
        dir=args['path'], data_set='test')

    print(f'Accuracy: {np.sum(np.diagonal(conf_matrix[0]))*100}%')

    sb.heatmap(conf_matrix[0], annot=True, cmap=
        sb.color_palette("light:b", as_cmap=True))

def confusion_matrix_vs_time(args):
    val_conf_matrices = utils.import_confusion_matrix(
        dir=args['path'], data_set='val')
    
    train_conf_matrices = utils.import_confusion_matrix(
        dir=args['path'], data_set='train')


    num_epochs = len(val_conf_matrices)
    num_classes = val_conf_matrices[0].shape[0]
    val_diags = np.zeros((num_epochs, num_classes))
    val_off_diags = np.zeros((num_epochs, 1))
    train_diags = np.zeros((num_epochs, num_classes))
    train_off_diags = np.zeros((num_epochs, 1))

    for i, matrix in enumerate(val_conf_matrices):
        val_diags[i, :] = np.diagonal(matrix)
        val_off_diags[i] = np.sum(np.sum(matrix)) - np.sum(np.diagonal(matrix))

    for i, matrix in enumerate(train_conf_matrices):
        train_diags[i, :] = np.diagonal(matrix)
        train_off_diags[i] = np.sum(np.sum(matrix)) - np.sum(np.diagonal(matrix))
    
    epoch_array = np.arange(1, num_epochs + 1)

    print(f'Accuracy val: {np.sum(val_diags, axis=1)*100}%')
    print(f'Accuracy train: {np.sum(train_diags, axis=1)*100}%')
    
    plt.plot(epoch_array, np.sum(train_diags, axis=1), label='Train acc.')
    plt.plot(epoch_array, np.sum(val_diags, axis=1), label='Val. acc.')
    # plt.plot(epoch_array, val_off_diags, 'k')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.legend()
    plt.ylim(0, 1)

def plot_roc_curves(args):
    truth, prob = utils.import_probs_and_truth(args['path'], data_set='test')
    
    num_points, num_classes = truth.shape[0], truth.shape[1]

    print('num_classes', num_classes)

    classes3 = ['ash', 'dust', 'pollen']
    classes_pollen = ['corylus', 'qrobur', 'qsuber']
    classes6 = ['camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub']

    classes = classes6

    plt.figure(figsize=(5, 3), constrained_layout=True)

    for i in range(num_classes):
        temp_fpr, temp_tpr, _ = skl_metrics.roc_curve(truth[:, i],
            prob[:, i], drop_intermediate=True)

        temp_auc = skl_metrics.auc(temp_fpr, temp_tpr)

        plt.plot(temp_fpr, temp_tpr, label=classes[i] + f'; AUC = {temp_auc:.4f}')
    
    fpr_micro, tpr_micro, _ = skl_metrics.roc_curve(truth.ravel(), prob.ravel())
    temp_auc = skl_metrics.auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, 'k', label=f'Micro-average; AUC = {temp_auc:.4f}')
    plt.legend()

def visualise_trained_cnn_filters(args):
    t_cnn = cnns.TorchNeuralNetwork1(num_classes=3).to(DEVICE)
    # print('args["cnn_file"]', args["cnn_file"])
    t_cnn.load_state_dict(torch.load(args["cnn_file"], 
        map_location=torch.device(DEVICE)))

    t_cnn.eval()
    with torch.no_grad():
        utils.plot_images(images=t_cnn.conv2.weight)

def visualise_trained_cnn_featuremaps(args):
    train_dataset = torchvision.datasets.ImageFolder(
        root=args["train_path"], transform=VISUALIZE_TRANSFORM,
        loader=utils.import_img)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
        shuffle=True)

    
    t_cnn = cnns.TorchNeuralNetwork2(num_classes=6).to(DEVICE)
    # print('args["cnn_file"]', args["cnn_file"])
    t_cnn.load_state_dict(torch.load(args["cnn_file"],
        map_location=torch.device(DEVICE)))

    print('t_cnn', dir(t_cnn))
    input()


    t_cnn.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, start=0):
            # Extract labels and data
            input_batch, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            featuremaps1 = t_cnn.conv1(input_batch)
            featuremaps2 = t_cnn.conv2(featuremaps1)
            featuremaps1 = featuremaps1.permute(1, 0, 2, 3)
            featuremaps2 = featuremaps2.permute(1, 0, 2, 3)

            utils.plot_images(images=featuremaps1)
            utils.plot_images(images=input_batch)

            utils.plot_images(images=featuremaps2)


            plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    if args['path'] is not None:
        path = pl.Path(args['path'])

        files = list(path.glob('*.txt'))
        network_prefix = 'saved_network'

        relevant_file = []
        for file in files:
            if network_prefix in file.stem:
                relevant_file = file

        args['cnn_file'] = str(relevant_file)

    if args['plot_type'] == 'history':
        history_figure(args)
    elif args['plot_type'] == 'conf':
        confusion_matrix(args)
    elif args['plot_type'] == 'conf_vs_time':
        confusion_matrix_vs_time(args)
    elif args['plot_type'] == 'roc':
        plot_roc_curves(args)
    elif args['plot_type'] == 'filters':
        visualise_trained_cnn_filters(args)
    elif args['plot_type'] == 'featuremaps':
        visualise_trained_cnn_featuremaps(args)

    plt.show()