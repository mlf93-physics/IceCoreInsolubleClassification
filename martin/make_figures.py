import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import sklearn.metrics as skl_metrics
import utilities as utils
import cnn_setups as cnns
from utilities.constants import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default=
    'F:/Data_IceCoreInsolubleClassification/train/')
parser.add_argument('--path', type=str, default='')
parser.add_argument('--cnn_file', type=str, default=None)
parser.add_argument('--plot_type', type=str, default='history')


def history_figure(args):
    headers, indices, values = utils.import_history(
        dir=args['path'])
    
    for i in range(len(headers)):
        plt.plot(indices[i], values[i], label=f'{headers[i][1]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def confusion_matrix_vs_time(args):
    conf_matrices = utils.import_confusion_matrix(
        dir=args['path'], data_set='val')

    num_epochs = len(conf_matrices)
    num_classes = conf_matrices[0].shape[0]
    diags = np.zeros((num_epochs, num_classes))
    off_diags = np.zeros((num_epochs, 1))

    for i, matrix in enumerate(conf_matrices):
        diags[i, :] = np.diagonal(matrix)
        off_diags[i] = np.sum(np.sum(matrix)) - np.sum(np.diagonal(matrix))
    
    epoch_array = np.arange(1, num_epochs + 1)
    
    plt.plot(epoch_array, diags)
    plt.plot(epoch_array, off_diags, 'k')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

def plot_roc_curves(args):
    truth, prob = utils.import_probs_and_truth(args['path'], data_set='test')
    
    num_points, num_classes = truth.shape[0], truth.shape[1]

    print('num_classes', num_classes)

    classes3 = ['ash', 'dust', 'pollen']
    classes6 = ['camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub']

    classes = classes3

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

    
    t_cnn = cnns.TorchNeuralNetwork1(num_classes=3).to(DEVICE)
    # print('args["cnn_file"]', args["cnn_file"])
    t_cnn.load_state_dict(torch.load(args["cnn_file"],
        map_location=torch.device(DEVICE)))


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

    if args['plot_type'] == 'history':
        history_figure(args)
    elif args['plot_type'] == 'conf_vs_time':
        confusion_matrix_vs_time(args)
    elif args['plot_type'] == 'roc':
        plot_roc_curves(args)
    elif args['plot_type'] == 'filters':
        visualise_trained_cnn_filters(args)
    elif args['plot_type'] == 'featuremaps':
        visualise_trained_cnn_featuremaps(args)

    plt.show()