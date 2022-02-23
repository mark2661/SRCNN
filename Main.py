import copy
import torch
import time
import Train
import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from Dataset import TrainingDataset, ValidationDataset
from Model import SRCNN
from utils import plot_training_results


def main(training_data_path, validation_data_path, learning_rate, batch_size, number_of_epochs):
    # set the training device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create the training and validation dataset objects for the dataloader
    training_dataset = TrainingDataset(training_data_path)
    validation_dataset = ValidationDataset(validation_data_path)

    # define the dataloaders
    training_loader = DataLoader(dataset=training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)

    # create SRCNN model instance and pass to gpu for training
    model = SRCNN()
    model.to(DEVICE)

    # define the SRCNN model parameters
    optimiser = torch.optim.Adam([
        {'params': model.l1.parameters()},
        {'params': model.l2.parameters()},
        {'params': model.l3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)
    criterion = nn.MSELoss()

    # arrays to store statistics from each training loop
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    best_psnr = 0
    best_weights = copy.deepcopy(model.state_dict())
    # note the start time for use calculating the final running time of the model training loop
    start = time.time()

    # main training and validation loop
    for epoch in range(number_of_epochs):
        print(f"Epoch {epoch + 1} of {number_of_epochs}")
        train_epoch_loss, train_epoch_psnr = Train.train(model,
                                                         training_loader,
                                                         device=DEVICE,
                                                         criterion=criterion,
                                                         optimiser=optimiser,
                                                         iterations_per_epoch=int(
                                                             len(training_dataset) // training_loader.batch_size))
        val_epoch_loss, val_epoch_psnr = Train.validate(model,
                                                        validation_loader,
                                                        device=DEVICE,
                                                        criterion=criterion,
                                                        iterations_per_epoch=int(
                                                            len(validation_dataset) // validation_loader.batch_size))

        print(f"\nTrain PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")

        # store the epoch training and validation average loss and average PSNR for data plotting
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        # save the best state_dict
        if val_epoch_psnr > best_psnr:
            best_psnr = val_epoch_psnr
            best_weights = copy.deepcopy(model.state_dict())

    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")

    # save the best model state dict to disk
    print('Saving model...')
    if os.path.isfile(os.path.join(os.curdir, "outputs", 'model.pth')):
        # if file exists delete it so we can save a new state dict with the same name
        os.remove(os.path.join(os.curdir, "outputs", 'model.pth'))
    torch.save(best_weights, os.path.join(os.curdir, "outputs", 'model.pth'))

    # display Results
    plot_training_results(model, train_loss, train_psnr, val_loss, val_psnr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str, required=True)
    parser.add_argument('--validation-data', type=str, required=True)
    parser.add_argument('--lr', type=float, default=pow(10, -4))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    main(training_data_path=args.training_data,
         validation_data_path=args.validation_data,
         learning_rate=args.lr,
         batch_size=args.batch_size,
         number_of_epochs=args.epochs)
