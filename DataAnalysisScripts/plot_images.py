import pickle
import os
from definitions import ROOT_DIR
import cv2
from Predict import predict_srcnn
from utils import artificially_degrade_image, calculate_psnr
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

with open(os.path.join(ROOT_DIR, 'Data', 'best_models.pickle'), 'rb') as f:
    best_models = pickle.load(f)

df_14_psnr = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_psnr_data.pkl'))
df_14_ssim = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_ssim_data.pkl'))
df_5_psnr = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_psnr_data.pkl'))
df_5_ssim = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_ssim_data.pkl'))


SET5_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set5')
SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
IMAGE_PATHS = [os.path.join(SET5_PATH, 'butterfly.{}'.format('bmp')), os.path.join(SET14_PATH, 'ppt3.{}'.format('bmp')),
               os.path.join(SET14_PATH, 'baboon.{}'.format('bmp'))]

for image_path in IMAGE_PATHS:
    image = image_path.split('\\')[-1].split('.')[0]
    set_num = image_path.split('\\')[-2]
    #print(set_num)
    gt = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)
    predicted_images = [cv2.cvtColor(gt, cv2.COLOR_YCrCb2RGB), cv2.cvtColor(artificially_degrade_image(gt, 3), cv2.COLOR_YCrCb2RGB)]
    #predicted_images = [gt[:, :, 0],
                        #artificially_degrade_image(gt, 3)[:, :, 0]]
    for model in sorted(best_models.keys()):
        network_filter_num = int(model)
        predicted_images.append(predict_srcnn(image_path, best_models[model], network_filter_num)[2])


    f, ax = plt.subplots(2, 4, figsize=(19.2, 10.8))
    f.suptitle("{} {}".format(set_num, image), fontsize=16)
    ax[0, 0].imshow(predicted_images[0])
    ax[0, 0].set_title('Original / PSNR (dB) / SSIM')
    ax[0, 1].imshow(predicted_images[1])
    ax[0, 1].set_title('Bi-linear / {:.3f} / {:.3f}'.format(df_5_psnr['0'].loc[image] if set_num=='Set5' else df_14_psnr['0'].loc[image],
                                                            df_5_ssim['0'].loc[image] if set_num=='Set5' else df_14_ssim['0'].loc[image]))

    ax[0, 2].imshow(predicted_images[2])
    ax[0, 2].set_title('16-8-1 network / {:.3f} / {:.3f}'.format(df_5_psnr['16'].loc[image] if set_num=='Set5' else df_14_psnr['16'].loc[image],
                                                            df_5_ssim['16'].loc[image] if set_num=='Set5' else df_14_ssim['16'].loc[image]))
    ax[0, 3].imshow(predicted_images[3])
    ax[0, 3].set_title('32-16-1 network / {:.3f} / {:.3f}'.format(df_5_psnr['32'].loc[image] if set_num=='Set5' else df_14_psnr['32'].loc[image],
                                                            df_5_ssim['32'].loc[image] if set_num=='Set5' else df_14_ssim['32'].loc[image]))

    ax[1, 0].imshow(predicted_images[4])
    ax[1, 0].set_title('64-32-1 network / {:.3f} / {:.3f}'.format(df_5_psnr['64'].loc[image] if set_num=='Set5' else df_14_psnr['64'].loc[image],
                                                            df_5_ssim['64'].loc[image] if set_num=='Set5' else df_14_ssim['64'].loc[image]))

    ax[1, 1].imshow(predicted_images[5])
    ax[1, 1].set_title('128-64-1 network / {:.3f} / {:.3f}'.format(df_5_psnr['128'].loc[image] if set_num=='Set5' else df_14_psnr['128'].loc[image],
                                                            df_5_ssim['128'].loc[image] if set_num=='Set5' else df_14_ssim['128'].loc[image]))

    ax[1, 2].imshow(predicted_images[6])
    ax[1, 2].set_title('256-128-1 network / {:.3f} / {:.3f}'.format(df_5_psnr['256'].loc[image] if set_num=='Set5' else df_14_psnr['256'].loc[image],
                                                            df_5_ssim['256'].loc[image] if set_num=='Set5' else df_14_ssim['256'].loc[image]))
    [axi.set_axis_off() for axi in ax.ravel()]

    #f.tight_layout()
    plt.axis('off')
   # plt.savefig(os.path.join(ROOT_DIR, 'Data', 'plots', '{}_subjective.svg'.format(image)), format='svg', dpi=1200)
    plt.show()

