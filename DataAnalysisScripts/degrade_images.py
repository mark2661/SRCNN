import os
import glob
import utils
import cv2
from definitions import ROOT_DIR
from pathlib import Path


SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
output_dir = os.path.join(ROOT_DIR, 'Data', 'degraded_set14')

for image_path in glob.glob(os.path.join(SET14_PATH, '*')):
    image_name = image_path.split('\\')[-1]
    image = cv2.imread(image_path)

    image = utils.modcrop(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb), 3)
    degraded_image = utils.artificially_degrade_image(image, 3)
    # create output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, image_name), degraded_image)