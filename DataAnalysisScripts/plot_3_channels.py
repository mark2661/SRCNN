import os
import cv2
from definitions import ROOT_DIR
import numpy as np

#IMAGE_FOLDER_PATH = os.path.join(ROOT_DIR, 'DATA')
IMAGE_FOLDER_PATH = os.path.join(ROOT_DIR, 'DATA', 'example_t91_images2')
SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
output_dir = os.path.join(ROOT_DIR, 'Data', 'presentation_images')

image = cv2.imread(os.path.join(IMAGE_FOLDER_PATH, 'tt8.png'))
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(image)
#cv2.imwrite(os.path.join(output_dir, 't91_y.png'), y)
cv2.imshow('y', cv2.cvtColor(y, cv2.COLOR_GRAY2RGB))
cv2.imshow('cr', cr)
cv2.imshow('cb', cb)

# cv2.imwrite(os.path.join(output_dir, 'ilr-tt8-y.png'), y)
# cv2.imwrite(os.path.join(output_dir, 'ilr-tt8-cr.png'), cr)
# cv2.imwrite(os.path.join(output_dir, 'ilr-tt8-cb.png'), cb)

# h, w = y.shape
# y_patch = y[h//2-h//4:h//2+h//4, w//2-w//4:w//2+w//4]
# cv2.imshow('y-patch', y_patch)
cv2.waitKey(0)
#cv2.imwrite(os.path.join(output_dir, 'hr-tt8-y-patch.png'), y_patch)



# def train(args):
#
#
#
# lr = cv2.resize(lr, (hr_width, hr_height), interpolation=cv2.INTER_LINEAR)
#
# lr = np.array(lr).astype(np.float32)
# #hr = convert_rgb_to_y(hr)
# hr = hr[:, :, 0]
# #lr = convert_rgb_to_y(lr)
# lr = lr[:, :, 0]
#
# for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
#     for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
#         lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
#
#
# lr_patches = np.array(lr_patches)


