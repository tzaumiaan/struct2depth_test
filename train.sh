#!/bin/sh
tf_model_root=~/workspace/models/research/struct2depth

ckpt_dir=model_ckpt
data_dir=kitti_processed
#imagenet_ckpt=resnet_imagenet_v2_fp32_20181001/model.ckpt-225207
pretrained_ckpt=model_kitti/model-199160

python $tf_model_root/train.py \
  --logtostderr \
  --checkpoint_dir=$ckpt_dir \
  --data_dir=$data_dir \
#  --imagenet_ckpt=$imagenet_ckpt \
#  --imagenet_norm=True \
  --pretrained_ckpt=$pretrained_ckpt \
  --joint_encoder=False
