import cv2
import torch
import Model
import os
import numpy as np
import argparse
import glob
import pandas as pd
import openpyxl
import pickle
from definitions import ROOT_DIR
from matplotlib import pyplot as plt
from utils import calculate_psnr, artificially_degrade_image, modcrop, calculate_mse, shave
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from collections import defaultdict


# def display_predicted_results(gt, deg, pre):
#     # display image subplots
#     fig, axs = plt.subplots(1, 3, figsize=(20, 8))
#     axs[0].imshow(gt)
#     axs[0].set_title('Original Image')
#
#     axs[1].imshow(deg)
#     axs[1].set_title('Degraded Image\nPSNR: {:.2f}\nMSE: {:.2f}'.format(calculate_psnr(gt, deg, 255.),
#                                                                         calculate_mse(gt, deg)))
#
#     axs[2].imshow(pre)
#     axs[2].set_title("SRCNN Predicted Image\nPSNR: {:.2f}\nMSE: {:.2f}".format(calculate_psnr(gt, pre, 255.),
#                                                                                calculate_mse(gt, pre)))
#
#     # remove axis ticks
#     for ax in axs:
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     plt.show()


def test_srcnn(REFERENCE_IMAGE_PATH,
               PRE_TRAINED_MODEL_WEIGHTS_PATH, filter_num, scale=3):
    """
    This function predicts a high resolution version of a ground truth image using an artificially
    degraded version of the ground truth image.
    :param REFERENCE_IMAGE_PATH: file path to the ground truth image
    :param PRE_TRAINED_MODEL_WEIGHTS_PATH: file path to the pre-trained SRCNN weights
    :param scale: up-scaling factor of the low res image
    :return: PSNR of the degraded image, PSNR of the SRCNN predicted image
    """
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Create a model instance and load in pre-trained weights
    model = Model.SRCNN(filter_num)
    state_dict = torch.load(PRE_TRAINED_MODEL_WEIGHTS_PATH)
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
    # predicted_image = cv2.cvtColor(deg_y_cr_cb_image, cv2.COLOR_YCrCb2RGB)
    predicted_image = predicted[:, :, 0]

    r = shave(cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)[:, :, 0], 3)
    d = shave(cv2.cvtColor(deg, cv2.COLOR_BGR2YCrCb)[:, :, 0], 3)
    p = shave(predicted_image, 3)
    return calculate_psnr(r, p, 255.), ssim(r, p), calculate_mse(r, p)


def main(test_set_path, model_weights_path, filter_num, median=False):
    if median:
        srcnn_psnr = []
        srcnn_ssim = []
        srcnn_mse = []
        for image in os.listdir(test_set_path):
                test_image_path = os.path.join(test_set_path, image)
                srcnn_psnr_score, srcnn_ssim_score, srcnn_mse_score = test_srcnn(test_image_path, model_weights_path, filter_num)
                srcnn_psnr.append(srcnn_psnr_score)
                srcnn_mse.append(srcnn_mse_score)
                srcnn_ssim.append(srcnn_ssim_score)

        return np.median(srcnn_psnr), np.median(srcnn_ssim), np.median(srcnn_mse)

    else:
        running_bi_cubic_psnr = 0
        running_srcnn_psnr = 0
        running_srcnn_ssim = 0
        running_srcnn_mse = 0
        for image in os.listdir(test_set_path):
            test_image_path = os.path.join(test_set_path, image)
            srcnn_psnr, srcnn_ssim, srcnn_mse = test_srcnn(test_image_path, model_weights_path, filter_num)
            # running_bi_cubic_psnr += bi_cubic_psnr
            running_srcnn_psnr += srcnn_psnr
            running_srcnn_ssim += srcnn_ssim
            running_srcnn_mse += srcnn_mse

        # average_bi_cubic_psnr = running_bi_cubic_psnr / len(os.listdir(test_set_path))
        average_srcnn_psnr = running_srcnn_psnr / len(os.listdir(test_set_path))
        average_srcnn_ssim = running_srcnn_ssim / len(os.listdir(test_set_path))
        average_srcnn_mse = running_srcnn_mse / len(os.listdir(test_set_path))

        """
        print(
            "Test set: {}\nAverage Bi-Cubic PSNR: {:.2f}\nAverage SRCNN PSNR: {:.2f}\n".format(test_set_path.split("/")[-1],
                                                                                               average_bi_cubic_psnr,
                                                                                              average_srcnn_psnr))
        """
        return average_srcnn_psnr, average_srcnn_ssim, average_srcnn_mse



def calculate_averages(l):
    averages = []
    for i in range(3):
        running_total = 0
        for t in l:
            running_total += t[i]
        averages.append(running_total / len(l))
    return tuple(averages)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--output-dir', type=str, required=True)
    # args = parser.parse_args()
    # output_dir = args.output_dir
    test_results = dict()

    for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
        network_filter_number = int(network.split('\\')[-1].split('_')[0])
        print(network_filter_number)
        for test_set_path in glob.glob(os.path.join(ROOT_DIR, 'testSets', '*')):
            print(test_set_path)
            results = []
            for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
                psnr_score, ssim_score, mse_score = main(test_set_path, model_state_dict, network_filter_number)
                results.append((psnr_score, ssim_score, mse_score))
            if network_filter_number in test_results.keys():
                test_results[network_filter_number].append(calculate_averages(results))
            else:
                test_results[network_filter_number] = [calculate_averages(results)]



    with open('test_results.pickle', 'wb') as f:
        pickle.dump(test_results, f)

    df = pd.DataFrame.from_dict(test_results, orient='index', columns=['BSDS100', 'Set5', 'Set14', 'Urban100'])
    df.to_csv(os.path.join(ROOT_DIR, 'Data', 'test_set_results.csv'), encoding='utf-8')
    print(df)
