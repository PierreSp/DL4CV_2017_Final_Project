import argparse
import os
import logging
import datetime
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

# special functionality for plots
# WARNING NEEDS HUGE AMOUNT OF RESSOURCES AND MEMORY
detailedlog = 0
if detailedlog:
    niter = 100
    # we will do tensorboard...


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Super Resolution Models')
    parser.add_argument(
        '--crop_size', default=88, type=int, help='training images crop size')
    parser.add_argument(
        '--upscale_factor', default=4, type=int, choices=[2, 4, 8],
        help='super resolution upscale factor')
    parser.add_argument(
        '--g_trigger_threshold', default=0.2, type=float,
        choices=[0.1, 0.2, 0.3, 0.4, 0.5],
        help='generator update trigger threshold')
    parser.add_argument(
        '--g_update_number', default=2, type=int, choices=[1, 2, 3, 4, 5],
        help='generator update number')
    parser.add_argument(
        '--num_epochs', default=100, type=int, help='train epoch number')
    parser.add_argument(
        '--batch_size', default=64, type=int, help='batch size for training')
    parser.add_argument(
        '--verbose', action='store_true', help='create verbose logging file')
    parser.add_argument(
        '--no-cuda', action='store_true',
        help='override cuda and use cpu, even if cuda is available')
    return parser.parse_args()


# Define Constants
STATISTICS_PATH = 'logs/statistics/'

opt = parse_args()
CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
G_TRIGGER_THRESHOLD = opt.g_trigger_threshold
G_UPDATE_NUMBER = opt.g_update_number
BATCH_SIZE_TRAIN = opt.batch_size
VERBOSE = opt.verbose
USE_CUDA = True if not opt.no_cuda and torch.cuda.is_available() else False
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%h%M%s')

####################
###    Logger    ###
####################
logger = logging.getLogger('SRNET_logger')
logger.setLevel(logging.INFO)
# Create file handler which logs even debug messages
fh = logging.FileHandler('logs/training_{}.log'.format(TIMESTAMP))
if VERBOSE:
    print("Net is verbose. Not intended for long training")
    fh.setLevel(logging.DEBUG)
else:
    fh.setLevel(logging.INFO)
# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


# DataLoaders
train_set = TrainDatasetFromFolder(
    'data/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder(
    'data/val', upscale_factor=UPSCALE_FACTOR)
train_loader = DataLoader(dataset=train_set, num_workers=4,
                          batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4,
                        batch_size=1, shuffle=False)


# Networks and Loss
netG = Generator(UPSCALE_FACTOR)
print('# generator parameters:',
      sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:',
      sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if USE_CUDA:
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()


# Optimizer
optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [],
           'g_score': [], 'psnr': [], 'ssim': []}


# Actual training loop
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0,
                       'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)                       # CURRENT batch size
        running_results['batch_sizes'] += batch_size    # TOTAL visited size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        z = Variable(data)
        if USE_CUDA:
            real_img = real_img.cuda()
            z = z.cuda()
        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss
        ###########################
        index = 1
        while ((real_out.data[0] - fake_out.data[0] > G_TRIGGER_THRESHOLD) or g_update_first) and (
                index <= G_UPDATE_NUMBER):
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()  # backprop
            optimizerG.step()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            logger.debug("Fake-output_ mean: " + str(fake_out))
            logger.debug("Data 0 from gen: " + str(real_out.data[0]))
            logger.debug("Data" + str(real_out.data))
            g_update_first = False
            index += 1

        g_loss = generator_criterion(fake_out, fake_img, real_img)
        running_results['g_loss'] += g_loss.data[0] * batch_size
        d_loss = 1 - real_out + fake_out
        running_results['d_loss'] += d_loss.data[0] * batch_size
        running_results['d_score'] += real_out.data[0] * batch_size
        running_results['g_score'] += fake_out.data[0] * batch_size

        total_images_seen = running_results['batch_sizes']
        train_bar.set_description(
            desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                 .format(epoch, NUM_EPOCHS,
                         running_results['d_loss'] / total_images_seen,
                         running_results['g_loss'] / total_images_seen,
                         running_results['d_score'] / total_images_seen,
                         running_results['g_score'] / total_images_seen))

    netG.eval()
    out_path = 'results/val/SRF_{}_{}/'.format(UPSCALE_FACTOR, TIMESTAMP)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0,
                      'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        # Use volatile for more eff (no backward possible)
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if USE_CUDA:
            lr = lr.cuda()
            hr = hr.cuda()
        sr = netG(lr)

        batch_mse = ((sr - hr) ** 2).data.mean()
        valing_results['mse'] += batch_mse * batch_size
        batch_ssim = pytorch_ssim.ssim(sr, hr).data[0]
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['psnr'] = 10 * log10(
            1 / (valing_results['mse'] / valing_results['batch_sizes']))
        valing_results['ssim'] = (valing_results['ssims']
                                  / valing_results['batch_sizes'])
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

        val_images.extend(
            [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
    val_images = torch.stack(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)
    val_save_bar = tqdm(val_images, desc='[saving training results]')
    index = 1
    for image in val_save_bar:
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'epoch_%d_index_%d.png' %
                         (epoch, index), padding=5)
        index += 1

    # save model parameters
    torch.save(netG.state_dict(), 'logs/epochs/netG_epoch_%d_%d.pth' %
               (UPSCALE_FACTOR, epoch))
    torch.save(netD.state_dict(), 'logs/epochs/netD_epoch_%d_%d.pth' %
               (UPSCALE_FACTOR, epoch))
    # save loss\scores\psnr\ssim
    results['d_loss'].append(
        running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(
        running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(
        running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(
        running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'],
                  'Loss_G': results['g_loss'],
                  'Score_D': results['d_score'],
                  'Score_G': results['g_score'],
                  'PSNR': results['psnr'],
                  'SSIM': results['ssim']},
            index=range(1, epoch + 1))
        filename = '{}_srf_{}_train_results.csv'.format(
            TIMESTAMP, UPSCALE_FACTOR)
        filepath = os.path.join(STATISTICS_PATH, filename)
        data_frame.to_csv(filepath, index_label='Epoch')
