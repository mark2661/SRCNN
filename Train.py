import torch
from tqdm import tqdm
from utils import calculate_psnr


def train(model, dataloader, device, criterion, optimiser, iterations_per_epoch):
    # set model to training mode
    model.train()

    running_loss = 0
    running_psnr = 0

    for data in tqdm(dataloader, total=iterations_per_epoch):
        # send training data and label to gpu
        image_data = data[0].to(device)
        label = data[1].to(device)

        # forward pass
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # backpropagate and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # add batch loss and psnr to running totals
        running_loss += loss.item()
        batch_psnr = calculate_psnr(label, outputs)
        running_psnr += batch_psnr

    # calculate average training loss and psnr for the epoch
    average_loss = running_loss / iterations_per_epoch
    average_psnr = running_psnr / iterations_per_epoch
    return average_loss, average_psnr


def validate(model, dataloader, device, criterion, iterations_per_epoch):
    # set model to evaluation mode
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0

    # use torch.no_grad() to disable gradient calculations
    with torch.no_grad():
        for data in dataloader:
            # send training data and labels to gpu
            image_data = data[0].to(device)
            label = data[1].to(device)

            # forward pass
            outputs = model(image_data).clamp(0.0, 1.0)
            loss = criterion(outputs, label)

            # add batch loss and psnr to running totals
            running_loss += loss.item()
            batch_psnr = calculate_psnr(label, outputs)
            running_psnr += batch_psnr

    # calculate average training loss and psnr for the epoch
    average_loss = running_loss / iterations_per_epoch
    average_psnr = running_psnr / iterations_per_epoch

    return average_loss, average_psnr
