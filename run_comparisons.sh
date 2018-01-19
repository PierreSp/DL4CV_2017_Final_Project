#!/usr/bin/bash

# 1. Test:
python -m srgan.train --num_epochs 1 --batch_size 512

# 2. ONLY specific loss
# 2.1. only perception
python -m srgan.train --batch_size 512 --no-discriminator --weight_image 0
# 2.2. only pixel-wise
python -m srgan.train --batch_size 512 --no-discriminator --weight_perception 0
# 2.3. only adversarial
python -m srgan.train --batch_size 512 --weight_image 0 --weight_perception 0

# 2. 2/3 losses
# 2.1. no pixel-wise
python -m srgan.train --batch_size 512 --no-discriminator --weight_image 0
# 2.2. no discriminator
python -m srgan.train --batch_size 512 --no-discriminator
# 2.3. no perception
python -m srgan.train --batch_size 512 --weight_perception 0

# 3. VGG16 vs VGG19 vs VGG16&VGG19
python -m srgan.train --batch_size 512 --network vgg16
python -m srgan.train --batch_size 512 --network vgg19
python -m srgan.train --batch_size 512 --network vgg16vgg19

# 4. G_UPDATE_NUMBER = 1
python -m srgan.train --batch_size 512 --g_update_number 1
