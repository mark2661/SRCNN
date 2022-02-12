import cv2
import torch
import Model
import os
import numpy as np
from matplotlib import pyplot as plt
from Train import calculate_psnr


def centre_crop_image(image, cropped_height, cropped_width):
  """Perform a centre crop on image producing a new image of dimensions cropped_height X cropped_width"""
  image_centre = tuple(map(lambda x: x/2, image.shape))
  x = image_centre[1] - cropped_width/2
  y = image_centre[0] - cropped_height/2

  return image[int(y):int(y+cropped_height),int(x):int(x + cropped_width)] #might need to call .astype('float')

def artifically_degrade_image(image, factor):
  """
  Artificially degrade an image by downscaling by a factor of "factor"
  and upscaliing to the original size using Bi-linear interpolation
  """
  #calculate old and new image dimensions
  h, w, _ = image.shape
  new_height = h // factor
  new_width = w // factor

  #downscale the image
  image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

  #upscale the image
  image = cv2.resize(image, (w,h), interpolation = cv2.INTER_LINEAR)

  return image #might need to call .astype('float')

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
  """Removes the border from an image"""
  img = image[border: -border, border: -border]
  return img


PRE_TRAINED_MODEL_WEIGHTS_PATH = os.path.join(os.curdir, 'outputs', 'model.pth')

def diaplay_predicted_results(gt, deg, pre):
    # display image subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(gt)
    axs[0].set_title('Original Image')
    axs[1].imshow(deg)
    axs[1].set_title('Degraded Image\nPSNR: {:.2f}'.format(calculate_psnr(gt, deg)))
    #axs[1].set_title('Degraded Image')
    axs[2].imshow(pre)
    axs[2].set_title("SRCNN Predicted Image\nPSNR: {:.2f}".format(calculate_psnr(gt, pre)))
    #axs[2].set_title("SRCNN Predicted Image")

    # remove axis ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def predict_srcnn(reference_image_path):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    # Create a model instance and load in pre-trained weights
    model = Model.SRCNN()
    state_dict = torch.load(PRE_TRAINED_MODEL_WEIGHTS_PATH)
    model.load_state_dict(state_dict)
    # pass the model to the device
    model.to(device)
    # switch model to evaluation mode
    model.eval()

    # load the reference and degraded image
    ref = cv2.imread(reference_image_path)
    deg = artifically_degrade_image(ref, 3)
    #deg = artifically_degrade_image(ref, 2)

    # perform image pre-processing
    ref = modcrop(ref, 3)
    deg = modcrop(deg, 3)

    """" 
    Create an input image for the SRCNN from the degraded image,
    convert to YCrCb (cv2.imread reads in as BGR by default) 
    Extract the y-channel data (The SRCNN was trained on the Y channel only and only takes inputs with 1 colour channel) and normalise the pixel values.
    """
    # covert degraded image to YCrCb colour space
    temp = cv2.cvtColor(deg, cv2.COLOR_BGR2YCrCb)
    temp_h, temp_w, _ = temp.shape

    Y = np.zeros((temp_h, temp_w, 1), dtype=float)  # create a zeros matrix to store the y channel data
    Y[:, :, 0] = temp[:, :, 0].astype(np.float32) / 255  # typecast and normaise pixel intensities
    # print(Y.shape)

    # Pass image to SRCNN to predict high-res version
    with torch.no_grad():
        Y = np.transpose(Y, (2, 0, 1))  # reshape the matric to c X h X w format
        Y = torch.tensor(Y, dtype=torch.float).to(device)  # covert np matrix to torch tensor and pass to device
        Y = Y.unsqueeze(0)  # adds a fourth dimension which represents a batch size
        predicted = model(Y)

    predicted = predicted.cpu().detach().numpy()
    predicted = predicted.reshape(predicted.shape[2], predicted.shape[3],
                                  predicted.shape[1])  # reshape to h X w X c format
    # re-map pixel intensities to 0-255 range
    predicted *= 255
    # cap any pixels exceeding the thresshold
    predicted[predicted[:, :, :] > 255] = 255
    predicted[predicted[:, :, :] < 0] = 0
    predicted = predicted.astype(np.uint8)

    # merge predicted y channel with other channels
    temp[:, :, 0] = predicted[:, :, 0]
    predicted_image = cv2.cvtColor(temp, cv2.COLOR_YCrCb2RGB)

    return cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), cv2.cvtColor(deg, cv2.COLOR_BGR2RGB), predicted_image

def main():
    test_image_path = os.path.join(os.curdir, 'testSets', 'Set5', 'bird_GT.bmp')
    diaplay_predicted_results(*predict_srcnn(test_image_path))

if __name__ == '__main__':
    main()