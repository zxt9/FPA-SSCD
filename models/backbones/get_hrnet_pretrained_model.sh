#!/bin/bash

FILENAME="models/backbones/pretrained/hrnetv2_w48_imagenet_pretrained.pth"

mkdir -p models/backbones/pretrained
wget https://objects.githubusercontent.com/github-production-release-asset-2e65be/299848440/7e0d0200-0689-11eb-9294-6812fcacef39?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220612%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220612T225545Z&X-Amz-Expires=300&X-Amz-Signature=ba15e7c3cc90a3b69bf7abc45bfd6e5a777d651a46783b61cd8865dea9698e72&X-Amz-SignedHeaders=host&actor_id=58558095&key_id=0&repo_id=299848440&response-content-disposition=attachment%3B%20filename%3Dhrnet_cs_8090_torch11.pth&response-content-type=application%2Foctet-stream -O $FILENAME
