import os
import glob
import utils
import cv2
from definitions import ROOT_DIR
from pathlib import Path


SET5_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set5')
SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
IMAGE_FOLDER_PATH = os.path.join(ROOT_DIR, 'DATA', 'example_t91_images2')
output_dir = os.path.join(ROOT_DIR, 'Data')


#image_path = os.path.join(SET5_PATH, '{}.bmp'.format('baby'))
image_path = os.path.join(SET14_PATH, '{}.bmp'.format('flowers'))
#image_path = os.path.join(IMAGE_FOLDER_PATH, '{}.png'.format('tt8'))
image = cv2.imread(image_path)

#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
degraded_image = utils.artificially_degrade_image(image, 3)
# create output directory if it does not exist

cv2.imwrite(os.path.join(output_dir, 'bi-cubic-flowers.bmp'), degraded_image)

# h, w, _ = image.shape
# new_height = h // 3
# new_width = w // 3
#
# image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
# image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
# image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
# #cv2.imwrite(os.path.join(output_dir, 'lr-monarch.bmp'), image)
# cv2.imwrite(os.path.join(output_dir, 'ilr-tt8.png'), image)