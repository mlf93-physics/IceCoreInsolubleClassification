import math
import pathlib as pl
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as t_nn
import torch.optim as t_optim
import utilities as utils
from utilities.constants import *
import cnn_setups as cnns
import testing as test

# Set seed
torch.manual_seed(SEED)

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

def run_torch_CNN(args, train_dataloader=None, val_dataloader=None):
    print('Initialising torch CNN')
    t_cnn = cnns.TorchNeuralNetwork().to(DEVICE)
    criterion = t_nn.CrossEntropyLoss()
    optimizer = t_optim.SGD(t_cnn.parameters(), lr=0.001, momentum=0.9)

    print('Running torch CNN')
    train_loss_list = []
    train_loss_index_list = []
    val_loss_list = []
    val_loss_index_list = []
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times

        # Set model in train mode
        t_cnn.train()

        for i, data in enumerate(train_dataloader, start=0):
            running_train_loss = 0.0
            # Extract labels and data
            input_batch, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = t_cnn(input_batch)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            # Print statistics
            # Convert to float to avoid accumulating history (memory)
            step = 8
            running_train_loss += float(train_loss.item())
            if i % step == step - 1:    # print every 2000 mini-batches
                print('Epoch: %d, Batch: %5d, Running_train_loss: %.2e' %
                    (epoch + 1, i + 1, running_train_loss / step))
            
        train_loss_list.append(running_train_loss)
        train_loss_index_list.append(epoch)
            

        # Disable gradient computations and set model into evaluation mode
        t_cnn.eval()
        with torch.no_grad():
            # Predict on validation set
            output, truth = test.test_cnn(t_cnn, dataloader=val_dataloader)

        val_loss = criterion(output, truth)
        val_loss_list.append(val_loss)
        val_loss_index_list.append(epoch)

    print('Finished Training')

    if args.save_cnn:
        print('Saving trained network')
        torch.save(t_cnn.state_dict(), pl.Path(args.out_path) /
            f'saved_network_{TIME_STAMP}.txt')
    
    utils.save_history_array(args, train_loss_index_list, train_loss_list, history_name='train_loss')
    utils.save_history_array(args, val_loss_index_list, val_loss_list, history_name='val_loss')
    utils.plot_history_array(train_loss_index_list, train_loss_list)
    utils.plot_history_array(val_loss_index_list, val_loss_list)
    

def main(args):
    print('Using {} device'.format(DEVICE))

    print('Running main script')

    train_dataloader, val_dataloader = utils.define_dataloader(args)

    run_torch_CNN(args, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader)

if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments: ', args)


    if args.cnn_file is not None:
        test.test_validation_on_saved_model(args)
    else:
        main(args)

    plt.tight_layout()
    plt.show()