import copy
import torch
import time
import Train
import argparse
import os
import pickle
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
from Dataset import TrainingDataset, ValidationDataset
from Model import SRCNN
from utils import plot_training_results, save_results_plot
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss
from pathlib import Path
import sys


def main(training_data_path, validation_data_path, learning_rate,
         batch_size, number_of_epochs, filter_num, output_dir, model_num=1):
    # create an output dir for this model
    Path(os.path.join(output_dir, 'model{}'.format(model_num))).mkdir(parents=True, exist_ok=True)

    # set the training device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define a custom transform for the training dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90)
    ])

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
    model = SRCNN(filter_num)
    model.to(DEVICE)

    # define the SRCNN model parameters
    optimiser = torch.optim.Adam([
        {'params': model.l1.parameters()},
        {'params': model.l2.parameters()},
        {'params': model.l3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = SSIMLoss(data_range=1.)
    # criterion = MultiScaleSSIMLoss(kernel_size=3)

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

        if val_epoch_psnr > best_psnr:
            best_weights = copy.deepcopy(model.state_dict())

        # terminate training if over fitting is detected (moving average is increasing)
        moving_averages = pd.Series(val_loss).rolling(window=50, min_periods=10).mean().fillna(sys.maxsize)
        if epoch > 50 and moving_averages.iloc[-50] < moving_averages.iloc[-1]:
            break

        # save weights, validation_loss_history, and validation_psnr_history every 250 Epochs
        if epoch > 0 and epoch % 250 == 0:
            df = pd.DataFrame({'Training Loss': train_loss, 'Validation Loss': val_loss, 'Training_PSNR': train_psnr,
                               'Validation PSNR': val_psnr})
            # create checkpoint directory if one does not exist
            Path(os.path.join(output_dir, 'model{}'.format(model_num), 'checkpoints')).mkdir(parents=True,
                                                                                             exist_ok=True)
            save_current_training_state(weights=best_weights, data_frame=df,
                                        output_dir=os.path.join(output_dir, 'model{}'.format(model_num), 'checkpoints'),
                                        model_num=model_num, epoch_num=df.shape[0])

    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")

    # should maybe wrap in try catch statement ?
    print('Saving model...')
    df = pd.DataFrame({'Training Loss': train_loss, 'Validation Loss': val_loss, 'Training_PSNR': train_psnr,
                       'Validation PSNR': val_psnr})
    save_current_training_state(weights=best_weights, data_frame=df,
                                output_dir=os.path.join(output_dir, 'model{}'.format(model_num)),
                                model_num=model_num, epoch_num=df.shape[0])

    # save the plot of validation loss and PSNR (x-axis equals number of epochs, every 50 epochs)
    save_results_plot(val_loss=val_loss, val_psnr=val_psnr, num_epochs=df.shape[0], tick_spacing=50,
                      output_dir=os.path.join(output_dir, 'model{}'.format(model_num)),
                      file_name='model{}'.format(model_num))


def save_current_training_state(weights, data_frame, output_dir, model_num, epoch_num):
    # save model state dict
    torch.save(weights, os.path.join(output_dir, 'model{}_{}epochs.pth'.format(model_num, epoch_num)))

    # save the pandas data frame containing the training history (using pickle to serialise)
    data_frame.to_pickle(os.path.join(output_dir, 'model{}_{}epochs.pickle'.format(model_num, epoch_num)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str, required=True)
    parser.add_argument('--validation-data', type=str, required=True)
    parser.add_argument('--lr', type=float, default=pow(10, -4))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--filter-num', type=int, default=128)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--model-num', type=int, default=1)
    args = parser.parse_args()
    main(training_data_path=args.training_data,
         validation_data_path=args.validation_data,
         learning_rate=args.lr,
         batch_size=args.batch_size,
         number_of_epochs=args.epochs,
         filter_num=args.filter_num,
         output_dir=args.output_dir,
         model_num=args.model_num)
