import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as t_nn
import torch.optim as t_optim
import sklearn.metrics as skl_metrics
import numpy as np
import utilities as utils
from utilities.constants import *
from cnn_setups import TorchNeuralNetwork
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_file', type=str, default=None)
parser.add_argument('--n_datapoints', type=int, default=None)

def define_dataloader(args):
    print('Define dataloader')
    dataset = utils.ImageDataset(file_name='train.csv',
        root_dir=PATH_TO_TRAIN, transform_enabled=True,
        n_datapoints=args.n_datapoints)

    # Get sample indices
    train_sampler, val_sampler =\
        utils.train_val_dataloader_split(dataset.data_frame.shape[0],
            val_frac=0.25)

    train_dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=IMP_BATCH_SIZE, num_workers=NUM_WORKERS,
        sampler=train_sampler)
    
    val_dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=IMP_BATCH_SIZE, num_workers=NUM_WORKERS,
        sampler=val_sampler)

    return train_dataloader, val_dataloader
    

def run_torch_CNN(train_dataloader=None):
    print('Initialising torch CNN')
    t_cnn = TorchNeuralNetwork()
    criterion = t_nn.CrossEntropyLoss()
    optimizer = t_optim.SGD(t_cnn.parameters(), lr=0.001, momentum=0.9)

    print('Running torch CNN')
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, start=0):
            # Extract labels and data
            input_batch, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = t_cnn(input_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 8 == 7:    # print every 2000 mini-batches
            print('Epoch: %d, Batch: %5d, Running_loss: %.2e' %
                (epoch + 1, i + 1, running_loss / 8))
            running_loss = 0.0

    print('Finished Training. Saving trained network')

    torch.save(t_cnn.state_dict(), str(OUT_PATH /
        f'saved_network_{time_stamp}.txt'))

    return t_cnn

def test_validation(cnn, dataloader=None):
    print('Get predictions on data')

    probs = []
    predictions = []
    val_true = []

    for _, data in enumerate(dataloader, start=0):
        # Extract labels and data
        input_batch, labels = data
        # Save validation labels
        val_true.extend(labels.tolist())
        # Predict validation batches
        output = cnn(input_batch)

        # Get probabilities
        prob = torch.exp(output).detach().numpy()
        prob = prob / np.reshape(np.max(prob, axis=1), (IMP_BATCH_SIZE, 1))
        prob = prob[np.arange(IMP_BATCH_SIZE), np.argmax(prob, axis=1)]
        probs.extend(prob)
        # Get predictions
        predictions.extend(list(np.round(prob, 0).astype(np.int)))

    acc = skl_metrics.accuracy_score(val_true, prob)
    print(f'Accuracy score: {acc*100:.2f}%')
        

def test_validation_on_saved_model(args):
    print('Testing validation set on saved model')
    # Make cnn from file
    t_cnn = TorchNeuralNetwork()
    t_cnn.load_state_dict(torch.load(args.cnn_file))

    _, val_dataloader = define_dataloader(args)

    test_validation(t_cnn, dataloader=val_dataloader)

def main(args):
    # Get timestamp to save files to unique names
    global time_stamp
    time_stamp = time.gmtime()
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time_stamp)

    print('Running main script')

    train_dataloader, val_dataloader = define_dataloader(args)

    cnn = run_torch_CNN(train_dataloader=train_dataloader)

    test_validation(cnn, dataloader=val_dataloader)

    # n_batches = 1

    # images = []
    # labels = []
    # for _ in range(n_batches):
    #     batch, label = next(iter(train_dataloader))
    #     labels.append(label)
    #     images.extend([batch[i, :, :, :] for i in range(IMP_BATCH_SIZE)])

    # utils.plot_images(images=images)
        

if __name__ == '__main__':
    args = parser.parse_args()

    if args.cnn_file is not None:
        test_validation_on_saved_model(args)
    else:
        main(args)

    plt.show()