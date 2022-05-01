import pickle
import os
from definitions import ROOT_DIR
import cv2
from Predict import predict_srcnn
from utils import artificially_degrade_image, calculate_psnr
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

with open(os.path.join(ROOT_DIR, 'Data', 'best_models.pickle'), 'rb') as f:
    best_models = pickle.load(f)

SET5_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set5')
SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
IMAGE_PATHS = [os.path.join(SET14_PATH, 'baboon.{}'.format('bmp')) ,os.path.join(SET5_PATH, 'butterfly.{}'.format('bmp')), os.path.join(SET5_PATH, 'bird.{}'.format('bmp')),
               os.path.join(SET14_PATH, 'monarch.{}'.format('bmp')), os.path.join(SET14_PATH, 'zebra.{}'.format('bmp')),
               os.path.join(SET14_PATH, 'baboon.{}'.format('bmp'))]
for image_path in IMAGE_PATHS:
    gt = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)
    predicted_images = [cv2.cvtColor(gt, cv2.COLOR_YCrCb2RGB), cv2.cvtColor(artificially_degrade_image(gt, 3), cv2.COLOR_YCrCb2RGB)]
    #predicted_images = [gt[:, :, 0],
                        #artificially_degrade_image(gt, 3)[:, :, 0]]
    for model in best_models:
        network_filter_num = int(model)
        predicted_images.append(predict_srcnn(image_path, best_models[model], network_filter_num)[2])
    break

f, ax = plt.subplots(2, 4, figsize=(19.2, 10.8))
ax[0, 0].imshow(predicted_images[0])
ax[0, 0].set_title('Original / PSNR / SSIM')
ax[0, 1].imshow(predicted_images[1])
#ax[0, 1].set_title('Bicubic / {:.3f} / {:.3f}'.format(calculate_psnr(predicted_images[0], predicted_images[1], 255.),
                                              #ssim(predicted_images[0][:, :, 0], predicted_images[1], full=True)[0]))

ax[0, 2].imshow(predicted_images[2])
ax[0, 3].imshow(predicted_images[3])
ax[1, 0].imshow(predicted_images[4])
ax[1, 1].imshow(predicted_images[5])
ax[1, 2].imshow(predicted_images[6])
[axi.set_axis_off() for axi in ax.ravel()]

#f.tight_layout()
plt.axis('off')
plt.show()

