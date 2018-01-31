import cv2
import os
import torch
import torchvision.utils as utils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


from data_utils import TestDatasetFromFolderPierre, display_transform
from model import Generator

PATH_INPUT = "data/live/input/"
PATH_OUTPUT = "data/live/output/"
UPSCALE_FACTOR = 4
MODEL_NAME = "netG_epoch_80.pth"
FOLDERNAME = PATH_INPUT
OUT_PATH = PATH_OUTPUT + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

model = Generator(UPSCALE_FACTOR).eval()
# if torch.cuda.is_available():
#    model = model.cuda()
model.load_state_dict(torch.load('logs/epochs/' + MODEL_NAME,
                                 map_location=lambda storage, loc: storage))

if not os.path.exists(PATH_INPUT):
    os.makedirs(PATH_INPUT)
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

# Prepare cv2
cam = cv2.VideoCapture(0)
cv2.namedWindow("SRGAN_LIVE")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("SRGAN_LIVE", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "live_srgan_{}.png".format(img_counter)
        img_in_path = os.path.join(PATH_INPUT, img_name)
        cv2.imwrite(img_in_path, frame)
        print("{} written!".format(img_name))
        test_set = TestDatasetFromFolderPierre(
            str(FOLDERNAME), upscale_factor=UPSCALE_FACTOR)
        test_loader = DataLoader(dataset=test_set, num_workers=5,
                                 batch_size=1, shuffle=False)
        test_bar = tqdm(test_loader, desc='[testing datasets]')

        for image_name, lr_image, hr_restore_img, hr_restore_img_bi, hr_image in test_bar:
            image_name = image_name[0]
            lr_image = Variable(lr_image, volatile=True)
            hr_image = Variable(hr_image, volatile=True)
            sr_image = model(lr_image)
            test_images = torch.stack(
                [display_transform()(sr_image.data.cpu().squeeze(0)),
                 display_transform()(hr_restore_img.squeeze(0)),
                 display_transform()(hr_image.data.cpu().squeeze(0)),
                 display_transform()(hr_restore_img_bi.cpu().squeeze(0))])
            image = utils.make_grid(test_images, nrow=2, padding=5)
            utils.save_image(image, OUT_PATH + image_name, padding=5)
        os.remove(img_in_path)
        print("{} deleted!".format(img_name))
        img_out_path = os.path.join(PATH_OUTPUT + "4/", img_name)
        img_out = cv2.imread(img_out_path)
        cv2.imshow('image', img_out)
        while True:
            k = cv2.waitKey(1)
            if k % 256 == 32:
                cv2.destroyWindow('image')
                break

        #cv2.waitKey(0)
        img_counter += 1


cam.release()

cv2.destroyAllWindows()
