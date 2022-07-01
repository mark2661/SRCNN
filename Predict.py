import cv2
import torch
import Model
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import calculate_psnr, artificially_degrade_image, modcrop, calculate_mse
from definitions import ROOT_DIR


def display_predicted_results(gt, deg, pre):
    """
    Displays ground truth version, degraded version, and SRCNN predicted version of an image side by side
    with MSE and SSIM metric scores
    """
    # display image subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    # ground truth image subplot
    axs[0].imshow(gt, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Original Image\nPSNR: {:.2f}\nMSE: {:.2f}'.format(calculate_psnr(gt, gt, 255.),
                                                                        calculate_mse(gt, gt), ))
    # degraded image (low res) subplot
    axs[1].imshow(deg, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('Degraded Image\nPSNR: {:.2f}\nMSE: {:.2f}'.format(calculate_psnr(gt, deg, 255.),
                                                                        calculate_mse(gt, deg), ))
    # SRCNN predicted image subplot
    axs[2].imshow(pre, cmap='gray', vmin=0, vmax=255)
    axs[2].set_title(
        "SRCNN Predicted Image\nPSNR: {:.2f}\nMSE: {:.2f}".format(calculate_psnr(gt, pre, 255.),
                                                                  calculate_mse(gt, pre), ))

    # remove axis ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def predict_srcnn(REFERENCE_IMAGE_PATH, scale=3, greyscale=False):
    """
    This function predicts a high resolution version of a ground truth image using an artificially
    degraded version of the ground truth image.
    :param REFERENCE_IMAGE_PATH: file path to the ground truth image
    :param scale: up-scaling factor of the low res image
    :return: ground truth image (RBG format), Bi-linearly interpolated low res image (RBG format),
             SRCNN predicted hi-res image (RBG format)
    """
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Create a model instance and load in pre-trained weights
    model = Model.SRCNN()
    model_weights_path = os.path.join(ROOT_DIR, "model_weights.pth")
    state_dict = torch.load(model_weights_path)
    model.load_state_dict(state_dict)

    # pass the model to the device
    model.to(device)

    # switch model to evaluation mode
    model.eval()

    # load the reference and degraded image
    ref = cv2.imread(REFERENCE_IMAGE_PATH)
    deg = artificially_degrade_image(ref, scale)

    # resize the images so they divide wholly with the scale value
    ref = modcrop(ref, scale)
    deg = modcrop(deg, scale)

    """"
    Create an input image for the SRCNN from the degraded image,
    convert to YCrCb (cv2.imread reads in as BGR by default)
    Extract the y-channel data (The SRCNN was trained on the Y channel only and only takes inputs with 1 colour channel)
    and normalise the pixel values.
    """
    # covert degraded image to YCrCb colour space
    deg_y_cr_cb_image = cv2.cvtColor(deg, cv2.COLOR_BGR2YCrCb)
    deg_y_cr_cb_image_height, deg_y_cr_cb_image_width, _ = deg_y_cr_cb_image.shape

    # create a zeros matrix to store the y channel data
    y_channel = np.zeros((deg_y_cr_cb_image_height, deg_y_cr_cb_image_width, 1), dtype=float)
    y_channel[:, :, 0] = deg_y_cr_cb_image[:, :, 0].astype(np.float32) / 255  # typecast and normalise pixel intensities

    # Pass image to SRCNN to predict high-res version
    with torch.no_grad():
        # reshape the matrix from h X w X c format to c X h X w format
        y_channel = np.transpose(y_channel, (2, 0, 1))
        # covert np matrix to torch.float tensor and pass to device
        y_channel = torch.tensor(y_channel, dtype=torch.float).to(device)
        # add a fourth dimension which represents a batch size. b X c X h X w format
        y_channel = y_channel.unsqueeze(0)
        # .clamp will cap all pixel outputs outside of the 0 - 1 range
        predicted = model(y_channel).clamp(0.0, 1.0)

    predicted = predicted.cpu().detach().numpy()
    # reshape to h X w X c format
    predicted = predicted.reshape(predicted.shape[2], predicted.shape[3], predicted.shape[1])
    # re-map pixel intensities to 0-255 range
    predicted = np.clip(predicted * 255., 0.0, 255.0).astype(np.uint8)

    # merge predicted y channel with cr and cb channels and covert to RGB
    deg_y_cr_cb_image[:, :, 0] = predicted[:, :, 0]
    predicted_image = cv2.cvtColor(deg_y_cr_cb_image, cv2.COLOR_YCrCb2RGB)

    if greyscale:
        return cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), cv2.cvtColor(deg, cv2.COLOR_BGR2RGB), predicted[:, :, 0]
    else:
        return cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), cv2.cvtColor(deg, cv2.COLOR_BGR2RGB), predicted_image


def main(image_path):
    display_predicted_results(*predict_srcnn(image_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True)
    args = parser.parse_args()
    main(image_path=args.image_path)
