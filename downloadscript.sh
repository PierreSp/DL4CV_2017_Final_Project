#!/bin/bash

# get VOC 2012
echo "Downloading VOC 2012 dataset"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

rm VOCtrainval_11-May-2012.tar

# Move data from VOC jpeg folder with all images to train
mv VOCdevkit/VOC2012/JPEGImages/* data/train/.
# Move small set from training to validation
mv data/train/2010_0049* data/val/.

# Delete downloaded data
rm -rf VOCdevkit

echo "Downloaded VOC 2012 dataset"

# get 
echo "Downloading DIV2k dataset"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

rm DIV2K_train_HR.zip

# Move data from DIV2k image folder to train
mv DIV2K_train_HR/* data/train/.

# Delete downloaded data
rm -rf DIV2K

echo "Downloaded DIV2k dataset"

