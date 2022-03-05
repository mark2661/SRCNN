import copy
import torch
import time
import Train
import argparse
import os
import pickle
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Dataset import TrainingDataset, ValidationDataset
from Model import SRCNN
from utils import plot_training_results, save_results_plot
from piq import ssim, SSIMLoss, MultiScaleSSIMLoss
from pathlib import Path


def main(training_data_path, validation_data_path, learning_rate,
         batch_size, number_of_epochs, output_dir, model_num=1):
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
    training_dataset = TrainingDataset(training_data_path, transform)
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

        # # save the best state_dict
        # if val_epoch_psnr > best_psnr:
        #     best_psnr = val_epoch_psnr
        #     best_weights = copy.deepcopy(model.state_dict())

        # save weights, validation_loss_history, and validation_psnr_history every 250 Epochs
        if epoch > 0 and epoch % 250 == 0:
            save_current_training_state(model=model, val_psnr=val_psnr, val_loss=val_loss, output_dir=output_dir,
                                        model_num=model_num, epoch=epoch)

    end = time.time()
    print(f"Finished training in: {((end - start) / 60):.3f} minutes")

    # # save the best model state dict to disk
    # print('Saving model...')
    # if os.path.isfile(os.path.join(os.curdir, "outputs", 'model.pth')):
    #     # if file exists delete it so we can save a new state dict with the same name
    #     os.remove(os.path.join(os.curdir, "outputs", 'model.pth'))
    # #torch.save(best_weights, os.path.join(os.curdir, "outputs", 'model.pth'))
    # torch.save(model.state_dict(), os.path.join(os.curdir, "outputs", 'model.pth'))

    # should maybe wrap in try catch statement ?
    print('Saving model...')
    save_current_training_state(model=model, val_psnr=val_psnr, val_loss=val_loss, output_dir=output_dir,
                                model_num=model_num, epoch_num=number_of_epochs)

    # save the plot of validation loss and PSNR (x-axis equals number of epochs, every 50 epochs)
    save_results_plot(val_loss=val_loss, val_psnr=val_psnr, tick_spacing=49,
                      output_dir=os.path.join(output_dir, 'model{}'.format(model_num)),
                      file_name='model{}.png'.format(model_num))

    # # display Results
    # plot_training_results(model, train_loss, train_psnr, val_loss, val_psnr)


def save_current_training_state(model, val_psnr, val_loss, output_dir, model_num, epoch_num):
    # save model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, 'model{}'.format(model_num),
                                                'model{}_{}epochs.pth'.format(model_num, epoch_num)))
    # save list of epoch validation PSNR (using pickle to serialise list)
    with open(os.path.join(output_dir, 'model{}'.format(model_num), 'val_psnr_{}.pickle'.format(epoch_num)), 'wb') as f:
        pickle.dump(val_psnr, f)
    # save list of epoch validation loss ((using pickle to serialise list)
    with open(os.path.join(output_dir, 'model{}'.format(model_num), 'val_loss_{}.pickle'.format(epoch_num)), 'wb') as f:
        pickle.dump(val_loss, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str, required=True)
    parser.add_argument('--validation-data', type=str, required=True)
    parser.add_argument('--lr', type=float, default=pow(10, -4))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--model-num', type=int, default=1)
    args = parser.parse_args()
    main(training_data_path=args.training_data,
         validation_data_path=args.validation_data,
         learning_rate=args.lr,
         batch_size=args.batch_size,
         number_of_epochs=args.epochs,
         output_dir=args.output_dir,
         model_num=args.model_num)
