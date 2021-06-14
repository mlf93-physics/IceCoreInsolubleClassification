import torch
import numpy as np
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt
import cnn_setups as cnns
import utilities as utils
from utilities.constants import *

def save_train_accuracy(args, outputs, labels):
    # Get probabilities
    prob = torch.exp(outputs).cpu().detach().numpy()
    # Normalise
    prob = prob / np.reshape(np.sum(prob, axis=1), (-1, 1))

    # Save prediction from max probability
    index_of_max_prob = np.argmax(prob, axis=1)
    # Get predictions
    predictions = list(index_of_max_prob)

    # Get confusion matrix
    conf_matrix = skl_metrics.confusion_matrix(labels, predictions)
    conf_matrix = conf_matrix/np.sum(np.sum(conf_matrix))
    utils.save_conf_matrix(args, conf_matrix, data_set='train')


def test_cnn(cnn, args, dataloader=None, get_proba=False, data_set='val'):
    print(f'Get predictions on {data_set} data')
    
    predictions = []
    truth = []
    outputs = torch.Tensor().to(DEVICE)
    probs = None

    for _, data in enumerate(dataloader, start=0):
        # Extract labels and data
        input_batch, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        # Save validation labels
        truth.extend(labels.tolist())
        # Predict validation batches
        output = cnn(input_batch).to(DEVICE)
        outputs = torch.cat((outputs, output), 0).to(DEVICE)

        if get_proba:
            # Get probabilities
            prob = torch.exp(output).cpu().detach().numpy()
            # Normalise
            prob = prob / np.reshape(np.sum(prob, axis=1), (-1, 1))
            # Save prediction from max probability
            index_of_max_prob = np.argmax(prob, axis=1)
            # Get predictions
            predictions.extend(list(index_of_max_prob))
            # Get max probability
            # prob = prob[np.arange(prob.shape[0]), index_of_max_prob]
            if probs is None:
                probs = prob
            else:
                probs = np.concatenate((probs, prob), axis=0)

    n_datapoints = probs.shape[0]
    if get_proba:
        truth_2d_array = np.zeros((n_datapoints, args["num_classes"]))
        for i, truth_value in enumerate(truth):
            truth_2d_array[i, truth_value] = 1
        # Save probs and truth
        utils.save_probs_and_truth(args, probs, truth_2d_array,
            data_set=data_set)

        # Get confusion matrix
        conf_matrix = skl_metrics.confusion_matrix(truth, predictions)
        conf_matrix = conf_matrix/np.sum(np.sum(conf_matrix))
        print(f'Confusion matrix:\n {conf_matrix}')
        utils.save_conf_matrix(args, conf_matrix, data_set=data_set)

        if args["dev_plot"]:
            plt.imshow(conf_matrix)

    truth = torch.LongTensor(truth).to(DEVICE)
    
    return outputs, truth
        

def test_validation_on_saved_model(args):
    print('Testing validation set on saved model')
    # Make cnn from file
    t_cnn = cnns.TorchNeuralNetwork()
    t_cnn.load_state_dict(torch.load(args["cnn_file"]))

    _, val_dataloader = utils.define_dataloader(args)

    test_validation(t_cnn, dataloader=val_dataloader)