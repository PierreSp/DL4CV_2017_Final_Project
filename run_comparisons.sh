#!/usr/bin/bash

# 1. Test:
# python srgan/train.py --num_epochs 1 --batch_size 360

# 2. ONLY specific loss
# 2.1. only perception
python srgan/train.py --batch_size 360 --no-discriminator --weight_image 0
# 2.2. only pixel-wise
python srgan/train.py --batch_size 360 --no-discriminator --weight_perception 0
# 2.3. only adversarial
python srgan/train.py --batch_size 360 --weight_image 0 --weight_perception 0

# 2. 2/3 losses
# 2.1. no pixel-wise
python srgan/train.py --batch_size 360 --no-discriminator --weight_image 0
# 2.2. no discriminator
python srgan/train.py --batch_size 360 --no-discriminator
# 2.3. no perception
python srgan/train.py --batch_size 360 --weight_perception 0

# 3. VGG16 vs VGG19 vs VGG16&VGG19
python srgan/train.py --batch_size 360 --network vgg16
python srgan/train.py --batch_size 360 --network vgg19
python srgan/train.py --batch_size 360 --network vgg16vgg19

# 4. G_UPDATE_NUMBER = 1
python srgan/train.py --batch_size 360 --g_update_number 1
