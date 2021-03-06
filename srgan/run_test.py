import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolderPierre, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Datasets')
parser.add_argument('--upscale_factor', default=4, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_80.pth',
                    type=str, help='generator model epoch name')
parser.add_argument('--folder', default='data/test',
                    type=str, help='define folder with test images')
parser.add_argument('--outfolder', default='results/test/SRF_',
                    type=str, help='define folder for result images')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
FOLDERNAME = opt.folder
OUT_PATH = opt.outfolder + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

results = {'psnr': [], 'ssim': []}

model = Generator(UPSCALE_FACTOR).eval()
# if torch.cuda.is_available():
#    model = model.cuda()
model.load_state_dict(torch.load('logs/epochs/' + MODEL_NAME, map_location=lambda storage, loc:storage))

test_set = TestDatasetFromFolderPierre(
     str(FOLDERNAME), upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4,
                         batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing datasets]')

for image_name, lr_image, hr_restore_img, hr_image in test_bar:
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
    hr_image = Variable(hr_image, volatile=True)
    # if torch.cuda.is_available():
     #   lr_image = lr_image.cuda()
     #   hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).data[0]

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    utils.save_image(image, OUT_PATH + image_name, padding=5)

    # save psnr\ssim
    results['psnr'].append(psnr)
    results['ssim'].append(ssim)

out_path = 'logs/statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) +
                  '_test_results.csv', index_label='DataSet')
