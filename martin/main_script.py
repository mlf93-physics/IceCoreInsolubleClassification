import os
import pathlib as pl
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as t_nn
import torch.optim as t_optim
import torchsummary
from codecarbon import EmissionsTracker
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
parser.add_argument('--val_frac', type=float, default=0.15)
parser.add_argument('--test_frac', type=float, default=0.15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_folds', type=int, default=4)
parser.add_argument('--save_cnn', action='store_true')
parser.add_argument('--save_history', action='store_true')
parser.add_argument('--dev_plot', action='store_true')
parser.add_argument('-a', '--architecture', type=int, default=1)
parser.add_argument('--run_label', type=str, default='')

def run_torch_CNN(args, train_dataloader=None, val_dataloader=None,
        test_dataloader=None):
    print('Initialising torch CNN')
    if args["architecture"] == 1:
        t_cnn = cnns.TorchNeuralNetwork1(num_classes=args['num_classes']).to(DEVICE)
    elif args["architecture"] == 2:
        t_cnn = cnns.TorchNeuralNetwork2(num_classes=args['num_classes']).to(DEVICE)
    elif args["architecture"] == 3:
        t_cnn = cnns.TorchNeuralNetwork3(num_classes=args['num_classes']).to(DEVICE)
    elif args["architecture"] == 4:
        t_cnn = cnns.TorchNeuralNetwork4(num_classes=args['num_classes']).to(DEVICE)

    print(torchsummary.summary(t_cnn, (1, IMAGE_HEIGHT, IMAGE_WIDTH)))

    criterion = t_nn.CrossEntropyLoss()
    # optimizer = t_optim.SGD(t_cnn.parameters(), lr=0.001, momentum=0.9)
    optimizer = t_optim.Adam(t_cnn.parameters(), lr=args["lr"])

    print('Running torch CNN')
    train_loss_list = []
    train_loss_index_list = []
    val_loss_list = []
    val_loss_index_list = []
    for epoch in range(args["n_epochs"]):  # loop over the dataset multiple times
        running_train_loss = 0.0

        # Set model in train mode
        t_cnn.train()

        for i, data in enumerate(train_dataloader, start=0):
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
                    (epoch + 1, i + 1, running_train_loss / ((i + 1)*args["batch_size"])))

        train_loss_list.append(running_train_loss / ((i + 1)*args["batch_size"]))
        train_loss_index_list.append(epoch)
            

        # Disable gradient computations and set model into evaluation mode
        t_cnn.eval()
        with torch.no_grad():
            # Predict on validation set
            output, truth = test.test_cnn(t_cnn, args, dataloader=val_dataloader,
                get_proba=True, data_set='val')

        val_loss = criterion(output, truth)
        val_loss_list.append(val_loss)
        val_loss_index_list.append(epoch)

    print('Finished Training')
    t_cnn.eval()
    with torch.no_grad():
        # Predict on test set
        test_output, test_truth = test.test_cnn(t_cnn, args, dataloader=test_dataloader,
            get_proba=True, data_set='test')

    if args["save_cnn"]:
        print('Saving trained network')
        torch.save(t_cnn.state_dict(), pl.Path(args["out_path"]) /
            f'saved_network_{TIME_STAMP}.txt')
    
    if args["save_history"]:
        utils.save_history_array(args, train_loss_index_list, train_loss_list, history_name='train_loss')
        utils.save_history_array(args, val_loss_index_list, val_loss_list, history_name='val_loss')

    if args["dev_plot"]:
        utils.plot_history_array(train_loss_index_list, train_loss_list)
        utils.plot_history_array(val_loss_index_list, val_loss_list)
    

def main(args):
    print('Using {} device'.format(DEVICE))

    # tracker = EmissionsTracker(output_dir=args["out_path"])
    # tracker.start()

    print('Running main script')

    train_dataloader, val_dataloader, test_dataloader, num_classes\
        = utils.define_dataloader(args)

    args["num_classes"] = num_classes

    run_torch_CNN(args, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, test_dataloader=test_dataloader)

    # tracker.stop()

if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments: ', args)
    args = vars(args)

    # Append run_label to out path
    args['out_path'] += '/' + args['run_label'] + f'_run_{TIME_STAMP}/'
    os.mkdir(args['out_path'])

    if args["cnn_file"] is not None:
        test.test_validation_on_saved_model(args)
    else:
        main(args)

    if args["dev_plot"]:
        plt.tight_layout()
        plt.show()