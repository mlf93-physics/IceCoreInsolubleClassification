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
parser.add_argument('--path', type=str, default='')
parser.add_argument('--cnn_file', type=str, default=None)


def history_figure():
    headers, indices, values = utils.import_history(
        path='../../trained_cnns/test1_run_2021-06-11_14-45-33/temp/')
    
    for i in range(len(headers)):
        plt.plot(indices[i], values[i], label=f'{headers[i][1]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def confusion_matrix_vs_time():
    conf_matrices = utils.import_confusion_matrix(
        path='../../trained_cnns/test1_run_2021-06-11_14-45-33/val_conf_matrix_2021-06-11_14-45-33.txt')

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
    truth, prob = utils.import_probs_and_truth(args['path'])
    
    num_points, num_classes = truth.shape[0], truth.shape[1]

    print('num_classes', num_classes)

    classes3 = ['ash', 'dust', 'pollen']
    classes6 = ['camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub']

    classes = classes3

    for i in range(num_classes):
        temp_fpr, temp_tpr, _ = skl_metrics.roc_curve(truth[:, i],
            prob[:, i], drop_intermediate=True)

        temp_auc = skl_metrics.auc(temp_fpr, temp_tpr)

        plt.plot(temp_fpr, temp_tpr, label=classes[i] + f'; AUC = {temp_auc:.2f}')
    
    fpr_micro, tpr_micro, _ = skl_metrics.roc_curve(truth.ravel(), prob.ravel())
    temp_auc = skl_metrics.auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, 'k', label=f'Micro-average; AUC = {temp_auc:.2f}')
    plt.legend()

def visualise_trained_cnn_filters(args):
    t_cnn = cnns.TorchNeuralNetwork1(num_classes=6).to(DEVICE)
    # print('args["cnn_file"]', args["cnn_file"])
    t_cnn.load_state_dict(torch.load(args["cnn_file"]))

    t_cnn.eval()
    with torch.no_grad():
        utils.plot_images(images=t_cnn.conv1.weight)

def visualise_trained_cnn_featuremaps(args):
    train_dataset = torchvision.datasets.ImageFolder(
        root=args["train_path"], transform=VISUALIZE_TRANSFORM,
        loader=utils.import_img)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
        shuffle=True)

    
    t_cnn = cnns.TorchNeuralNetwork1(num_classes=6).to(DEVICE)
    # print('args["cnn_file"]', args["cnn_file"])
    t_cnn.load_state_dict(torch.load(args["cnn_file"]))

    t_cnn.eval()
    with torch.no_grad():
        image, label = dataloader()
        utils.plot_images(images=t_cnn.conv1(image))

if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)

    # history_figure()
    # confusion_matrix_vs_time()
    plot_roc_curves(args)
    # visualise_trained_cnn(args)

    plt.show()