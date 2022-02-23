import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import cv2
from utils import modcrop


def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        #hr = pil_image.open(image_path).convert('RGB')
        hr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)

        #hr_width = (hr.width // args.scale) * args.scale
        #hr_height = (hr.height // args.scale) * args.scale
        hr = modcrop(hr, args.scale)
        hr_height, hr_width, _ = hr.shape
        #hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = cv2.resize(hr, (hr_width//args.scale, hr_height//args.scale), interpolation=cv2.INTER_LINEAR)
        #lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = cv2.resize(lr, (hr_width, hr_height), interpolation=cv2.INTER_LINEAR)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        #hr = convert_rgb_to_y(hr)
        hr = hr[:, :, 0]
        #lr = convert_rgb_to_y(lr)
        lr = lr[:, :, 0]

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
       # hr = pil_image.open(image_path).convert('RGB')
        hr = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)
        #hr_width = (hr.width // args.scale) * args.scale
        #hr_height = (hr.height // args.scale) * args.scale
        hr = modcrop(hr, args.scale)
        hr_height, hr_width, _ = hr.shape
        #hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        #lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = cv2.resize(hr, (hr_width // args.scale, hr_height // args.scale), interpolation=cv2.INTER_LINEAR)
        #lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        lr = cv2.resize(lr, (hr_width, hr_height), interpolation=cv2.INTER_LINEAR)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
       # hr = convert_rgb_to_y(hr)
        hr = hr[:, :, 0]
       # lr = convert_rgb_to_y(lr)
        lr = lr[:, :, 0]

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)