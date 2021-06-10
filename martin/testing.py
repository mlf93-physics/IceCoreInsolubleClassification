import torch
import numpy as np
import sklearn.metrics as skl_metrics
import matplotlib.pyplot as plt
import cnn_setups as cnns
import utilities as utils
from utilities.constants import *

def test_cnn(cnn, args, dataloader=None, get_proba=False):
    print('Get predictions on data')

    probs = []
    predictions = []
    truth = []
    outputs = torch.Tensor().to(DEVICE)

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
            prob = prob[np.arange(prob.shape[0]), index_of_max_prob]
            probs.extend(prob)

    if get_proba:
        # Get accuracy score
        conf_matrix = skl_metrics.confusion_matrix(truth, predictions)
        conf_matrix = conf_matrix/np.sum(np.sum(conf_matrix))
        print(f'Confusion matrix:\n {conf_matrix}')
        utils.save_conf_matrix(args, conf_matrix)

        if args.dev_plot:
            plt.imshow(conf_matrix)

    truth = torch.LongTensor(truth).to(DEVICE)
    
    return outputs, truth
        

def test_validation_on_saved_model(args):
    print('Testing validation set on saved model')
    # Make cnn from file
    t_cnn = cnns.TorchNeuralNetwork()
    t_cnn.load_state_dict(torch.load(args.cnn_file))

    _, val_dataloader = utils.define_dataloader(args)

    test_validation(t_cnn, dataloader=val_dataloader)