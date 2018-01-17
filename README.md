# Super-Resolution with GANs

Based on the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf).  
Related work: https://github.com/leftthomas/SRGAN.  

## Datasets
- [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
<!-- Script to download -->

## Setup
1. Install pytorch as described on their [website](pytorch.org)
2. `pip install -r requirements.txt`
3. (Not done yet) Execute the setup script to download the VOC2012 data and partition them into train test and validation sets

## Usage
- Load images to *data/val* and *data/train* folder
- Run ```python srgan/train.py --upscale_factor 4 ``` All options can be found with *train --help*
- Find outcome pictures in *results/val* folder
- Find statics (PSNR, SSIM,...) in *logs/statistics*
- The weights will be saved in *log/epochs/*
- For testing load files to the *data/test* folder and run '''test.py''' with the trained weights as parameter