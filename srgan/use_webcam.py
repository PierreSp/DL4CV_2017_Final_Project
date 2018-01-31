import cv2
import os
import time
from matplotlib import pyplot as plt

PATH_INPUT = "data/live/input/"
PATH_OUTPUT = "data/live/output/"

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
        os.system("python srgan/run_test.py --folder " + str(PATH_INPUT) + " --outfolder " + str(PATH_OUTPUT))
        time.sleep(2)
        os.remove(img_in_path)
        print("{} deleted!".format(img_name))
        img_out_path = os.path.join(PATH_OUTPUT + "4/", img_name)
        img_out = cv2.imread(img_out_path)
        cv2.imshow('image', img_out)
        cv2.waitKey(0)
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
