import torch
from tqdm import tqdm
import math as maths
import numpy as np
import os
from torchvision.utils import save_image


def calculate_psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy() if type(label) == torch.Tensor else label
    outputs = outputs.cpu().detach().numpy() if type(outputs) == torch.Tensor else outputs
    img_diff = outputs - label
    rmse = maths.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        psnr = 20 * maths.log10(max_val / rmse)
        return psnr


def train(model, dataloader, device, criterion, optimiser, iterations_per_epoch):
    model.train()
    running_loss = 0
    running_psnr = 0
    for data in tqdm(dataloader, total= iterations_per_epoch):
        # send training data and labels to gpu
        image_data = data[0].to(device)
        label = data[1].to(device)

        # forward pass
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # backprop and optimise
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        batch_psnr = calculate_psnr(label, outputs)
        """
        original psnr function possibly didn't work because tensor contains only the y channel, 
        it might be defined to work with for rgb images. 
        also the need to check output dtype because torch dot tensor is defined as C X H X W 
        whereas most numpy images are H X W X C
        """
        running_psnr += batch_psnr

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / iterations_per_epoch

    return final_loss, final_psnr


def validate(model, dataloader, device, criterion, iterations_per_epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for data in tqdm(dataloader, total=iterations_per_epoch):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)

            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = calculate_psnr(label, outputs)
            running_psnr += batch_psnr

        #outputs = outputs.cpu()
        # save_image(outputs, f"../outputs/val_sr{epoch}.png")
        #save_image(outputs, os.path.join(ROOT_DIRECTORY, "outputs", "val_sr{}.png".format(epoch)))
    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / iterations_per_epoch

    return final_loss, final_psnr


