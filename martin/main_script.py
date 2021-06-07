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
    loss_list = []
    loss_index_list = []
    for epoch in range(args.n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, start=0):
            # Extract labels and data
            input_batch, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = t_cnn(input_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            # Convert to float to avoid accumulating history (memory)
            step = 8
            running_loss += float(loss.item())
            if i % step == step - 1:    # print every 2000 mini-batches
                print('Epoch: %d, Batch: %5d, Running_loss: %.2e' %
                    (epoch + 1, i + 1, running_loss / step))
            
            loss_list.append(running_loss)
            loss_index_list.append((epoch + 1)*(i + 1))
            running_loss = 0.0


    print('Finished Training')

    if args.save_cnn:
        print('Saving trained network')
        torch.save(t_cnn.state_dict(), pl.Path(args.out_path) /
            f'saved_network_{TIME_STAMP}.txt')
    
    utils.save_history_array(args, loss_index_list, loss_list, history_name='loss')
    utils.plot_history_array(loss_index_list, loss_list)
    
    test.test_validation(t_cnn, dataloader=val_dataloader)

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