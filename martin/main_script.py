import matplotlib.pyplot as plt
import torch
import torch.nn as t_nn
import torch.optim as t_optim
import utilities as utils
from utilities.constants import *
from cnn_setups import TorchNeuralNetwork

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

def define_dataloader():
    print('Define dataloader')
    dataset = utils.ImageDataset(file_name='train.csv',
        root_dir=PATH_TO_TRAIN, transform_enabled=True, n_datapoints=1000)

    # Get sample indices
    train_sampler, val_sampler =\
        utils.train_val_dataloader_split(dataset.shape[0], val_frac=0.25)

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
            if i % 8 == 7:    # print every 2000 mini-batches
                print('Epoch: %d, Batch: %5d, Running_loss: %.2e' %
                    (epoch + 1, i + 1, running_loss / 8))
                running_loss = 0.0

    print('Finished Training')

def main():
    print('Running main script')

    train_dataloader, val_dataloader = define_dataloader()

    run_torch_CNN(train_dataloader=train_dataloader)

    # n_batches = 1

    # images = []
    # labels = []
    # for _ in range(n_batches):
    #     batch, label = next(iter(train_dataloader))
    #     labels.append(label)
    #     images.extend([batch[i, :, :, :] for i in range(IMP_BATCH_SIZE)])

    # utils.plot_images(images=images)
        

if __name__ == '__main__':
    main()

    plt.show()