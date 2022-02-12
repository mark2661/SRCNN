import h5py
import torch
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import SRCNNDataset
from Model import SRCNN
import Train

ROOT_DIRECTORY = "/content/drive/MyDrive/FYP"


def load_data(path):
    with h5py.File(path)as data_file:
        # read in training data and training labels and cast data to 32 bit floats
        training_data = data_file['data'][:].astype('float32')
        training_label = data_file['label'][:].astype('float32')

    x_train, x_val, y_train, y_val = train_test_split(training_data, training_label,
                                                      test_size=0.25)  # test_size=0.25 means 25% of the samples will be used for the validation set
    return x_train, x_val, y_train, y_val


def plot_training_results(model, train_loss, train_psnr, val_loss, val_psnr):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../outputs/loss.png')
    #plt.savefig(os.path.join(ROOT_DIRECTORY, "outputs", "loss.png"))
    plt.show()
    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    #plt.savefig('../outputs/psnr.png')
    #plt.savefig(os.path.join(ROOT_DIRECTORY, "outputs", "psnr.png"))
    plt.show()
    # save the model to disk
    print('Saving model...')
    #torch.save(model.state_dict(), '../outputs/model.pth')
    #os.path.join(ROOT_DIRECTORY, "outputs", "loss.png")
    torch.save(model.state_dict(), os.path.join(os.curdir, "outputs",'model.pth'))


def main():
    # Model parameters
    batch_size = 64
    epochs = 100
    lr = pow(10, -4)
    DATA_PATH = os.path.join(os.curdir,'TrainingData', 'crop_train.h5') #need to enter the folder containing the .h5 file

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train, x_val, y_train, y_val = load_data(DATA_PATH)

    training_dataset = SRCNNDataset(images=x_train, labels=y_train)
    validation_dataset = SRCNNDataset(images=x_val, labels=y_val)

    training_loader = DataLoader(training_dataset, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    model = SRCNN()
    model.to(DEVICE)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr = Train.train(model,
                                                         training_loader,
                                                         device=DEVICE,
                                                         criterion=criterion,
                                                         optimiser=optimiser,
                                                         iterations_per_epoch=int(len(training_dataset)/training_loader.batch_size))
        val_epoch_loss, val_epoch_psnr = Train.validate(model,
                                                        validation_loader,
                                                        device=DEVICE,
                                                        criterion=criterion,
                                                        iterations_per_epoch=int(len(validation_dataset)/validation_loader.batch_size))

        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")
    plot_training_results(model, train_loss, train_psnr, val_loss, val_psnr)

if __name__ == "__main__":
    main()
