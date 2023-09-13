#!/bin/bash

FILENAME="models/backbones/pretrained/3x3resnet101-imagenet.pth"

mkdir -p models/backbones/pretrained
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O $FILENAME
