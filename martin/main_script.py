import pathlib as pl
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as t_nn
import torch.optim as t_optim
import sklearn.metrics as skl_metrics
import numpy as np
import utilities as utils
from utilities.constants import *
from cnn_setups import TorchNeuralNetwork
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_file', type=str, default=None)
parser.add_argument('--train_path', type=str, default=
    'F:/Data_IceCoreInsolubleClassification/train/')
parser.add_argument('--out_path', type=str, default=
    'C:/Users/Martin/Documents/FysikUNI/Kandidat/AppliedMachineLearning/final_project/trained_cnns')
parser.add_argument('--n_datapoints', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=2)
parser.add_argument('--val_frac', type=float, default=0.25)
parser.add_argument('--n_folds', type=int, default=4)
parser.add_argument('--save_cnn', action='store_true')


def define_dataloader(args):
    print('Define dataloader')
    train_dataset = torchvision.datasets.ImageFolder(
        root=args.train_path, transform=TRANSFORM_IMG,
        loader=utils.import_img)
    
    print('Train classes: ', train_dataset.classes, 'class_to_idx', train_dataset.class_to_idx)

    # train_dataset = utils.ImageDataset(file_name='train.csv',
    #     root_dir=PATH_TO_TRAIN, transform_enabled=True,
    #     n_datapoints=args.n_datapoints)

    # Get sample indices
    train_sampler, val_sampler =\
        utils.train_val_dataloader_split(args)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.n_threads,
        sampler=train_sampler)
    
    val_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, num_workers=args.n_threads,
        sampler=val_sampler)

    return train_dataloader, val_dataloader
    

def run_torch_CNN(args, train_dataloader=None):
    print('Initialising torch CNN')
    t_cnn = TorchNeuralNetwork().to(device)
    criterion = t_nn.CrossEntropyLoss()
    optimizer = t_optim.SGD(t_cnn.parameters(), lr=0.001, momentum=0.9)

    print('Running torch CNN')
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, start=0):
            # Extract labels and data
            input_batch, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = t_cnn(input_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 8 == 7:    # print every 2000 mini-batches
                print('Epoch: %d, Batch: %5d, Running_loss: %.2e' %
                    (epoch + 1, i + 1, running_loss / 8))
            running_loss = 0.0

    print('Finished Training')

    if args.save_cnn:
        print('Saving trained network')
        torch.save(t_cnn.state_dict(), pl.Path(args.out_path) /
            f'saved_network_{time_stamp}.txt')

    return t_cnn

def test_validation(cnn, dataloader=None):
    print('Get predictions on data')

    probs = []
    predictions = []
    val_true = []

    for _, data in enumerate(dataloader, start=0):
        # Extract labels and data
        input_batch, labels = data[0].to(device), data[1].to(device)
        # Save validation labels
        val_true.extend(labels.tolist())
        # Predict validation batches
        output = cnn(input_batch).to(device)

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

    _, val_dataloader = define_dataloader(args)

    test_validation(t_cnn, dataloader=val_dataloader)

def main(args):
    global time_stamp
    

    # Get timestamp to save files to unique names
    time_stamp = time.gmtime()
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time_stamp)

    print('Using {} device'.format(device))

    print('Running main script')

    train_dataloader, val_dataloader = define_dataloader(args)

    cnn = run_torch_CNN(args, train_dataloader=train_dataloader)

    test_validation(cnn, dataloader=val_dataloader)

    # n_batches = 1

    # images = []
    # labels = []
    # for _ in range(n_batches):
    #     batch, label = next(iter(train_dataloader))
    #     labels.append(label)
    #     images.extend([batch[i, :, :, :] for i in range(args.batch_size)])

    # utils.plot_images(images=images)
        

if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments: ', args)


    if args.cnn_file is not None:
        test_validation_on_saved_model(args)
    else:
        main(args)

    plt.show()