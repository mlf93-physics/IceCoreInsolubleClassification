import numpy as np
import matplotlib.pyplot as plt
import utilities as utils

def history_figure():
    headers, indices, values = utils.import_history(path='../../trained_cnns/test/')
    
    for i in range(len(headers)):
        plt.plot(indices[i], values[i], label=f'{headers[i][1]}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def confusion_matrix_vs_time():
    conf_matrices = utils.import_confusion_matrix(
        path='../../trained_cnns/conf_matrix_2021-06-10_15-33-14.txt')

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

# history_figure()
confusion_matrix_vs_time()

plt.show()