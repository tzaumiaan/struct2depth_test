#!/bin/sh
model_dir=mask_rcnn_inception_v2_coco_2018_01_28
data_dir=../kitti/img
output_dir=output

python mask_rcnn_inference.py \
--model_dir $model_dir \
--data_dir $data_dir \
--output_dir $output_dir \
--color gray

