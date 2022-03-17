import math as maths
import numpy as np
import torch
import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd


def calculate_psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy() if type(label) == torch.Tensor else label
    outputs = outputs.cpu().detach().numpy() if type(outputs) == torch.Tensor else outputs
    img_diff = outputs - label
    rmse = maths.sqrt(np.mean(pow(img_diff, 2)))
    if rmse == 0:
        return 100
    else:
        psnr = 20 * maths.log10(max_val / rmse)
        return psnr


def calculate_mse(label, outputs):
    label = label.cpu().detach().numpy() if type(label) == torch.Tensor else label
    outputs = outputs.cpu().detach().numpy() if type(outputs) == torch.Tensor else outputs
    img_diff = np.mean(pow(outputs - label, 2))
    return abs(img_diff)


def centre_crop_image(image, cropped_height, cropped_width):
    """
    Perform a centre crop on image producing a new image of dimensions cropped_height X cropped_width
    """
    image_centre = tuple(map(lambda x: x / 2, image.shape))
    x = image_centre[1] - cropped_width / 2
    y = image_centre[0] - cropped_height / 2

    return image[int(y):int(y + cropped_height), int(x):int(x + cropped_width)]  # might need to call .astype('float')


def artificially_degrade_image(image, factor):
    """
    Artificially degrade an image by downscaling by a factor of "factor"
    and upscaliing to the original size using Bi-linear interpolation
    """
    # calculate old and new image dimensions
    h, w, _ = image.shape
    new_height = h // factor
    new_width = w // factor

    # downscale the image
    #image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # upscale the image
    #image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    return image


def modcrop(img, scale):
    """
    crops the image to a size where the length and width
    divide evenly into the scaling factor
    """
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    """
    Removes the border from an image
    """
    img = image[border: -border, border: -border]
    return img


def plot_training_results(model, train_loss, train_psnr, val_loss, val_psnr):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('../outputs/loss.png')
    # plt.savefig(os.path.join(ROOT_DIRECTORY, "outputs", "loss.png"))
    plt.show()
    # psnr plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    # plt.savefig('../outputs/psnr.png')
    # plt.savefig(os.path.join(ROOT_DIRECTORY, "outputs", "psnr.png"))
    plt.show()


def save_results_plot(val_loss, val_psnr, num_epochs, tick_spacing, output_dir, file_name):
    # loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(list(range(0, num_epochs+1, tick_spacing)))
    plt.savefig(os.path.join(output_dir, '{}_loss.png'.format(file_name)))

    # psnr plot
    plt.figure(figsize=(10, 7))
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.xticks(list(range(0, num_epochs+1, tick_spacing)))
    plt.savefig(os.path.join(output_dir, '{}_psnr.png'.format(file_name)))
