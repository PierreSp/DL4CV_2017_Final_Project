import cv2
import os

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
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
