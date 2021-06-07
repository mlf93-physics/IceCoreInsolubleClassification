import torch
import numpy as np
import sklearn.metrics as skl_metrics
from cnn_setups import TorchNeuralNetwork
import utilities as utils

def test_validation(cnn, dataloader=None):
    print('Get predictions on data')

    probs = []
    predictions = []
    val_true = []

    for _, data in enumerate(dataloader, start=0):
        # Extract labels and data
        input_batch, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        # Save validation labels
        val_true.extend(labels.tolist())
        # Predict validation batches
        output = cnn(input_batch).to(DEVICE)

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

    # Get accuracy score
    acc = skl_metrics.balanced_accuracy_score(val_true, predictions)
    print(f'Accuracy score: {acc*100:.2f}%')
        

def test_validation_on_saved_model(args):
    print('Testing validation set on saved model')
    # Make cnn from file
    t_cnn = TorchNeuralNetwork()
    t_cnn.load_state_dict(torch.load(args.cnn_file))

    _, val_dataloader = utils.define_dataloader(args)

    test_validation(t_cnn, dataloader=val_dataloader)