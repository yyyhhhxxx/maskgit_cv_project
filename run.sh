#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
python vqgan_main0123.py --config configs/vqgan0123.yaml

# python vqgan_main.py --config configs/vqgan.yaml
# python -m pytorch_fid path/to/dataset1 path/to/dataset2