#!/bin/bash

echo "Cleaning val images"
find results/val/. -name "*.png" | grep -e '.*epoch_.*[24638]_' | xargs rm -f

echo "Cleaning epochs"
find logs/epochs/. -name "*.pth" | grep -e '.*epoch_.*[2468]\.pth$' | xargs rm -f
