#!/bin/sh
tf_model_root=~/workspace/models/research/struct2depth

input_dir=kitti/2011_09_26/2011_09_26_drive_0005_sync/image_02/data/
output_dir=output/
model_checkpoint="./model_kitti/model-199160"
#model_checkpoint="./model_trained_0414/model-28500"

python $tf_model_root/inference.py \
    --logtostderr \
    --file_extension=png \
    --depth \
    --egomotion=True \
    --input_dir=$input_dir \
    --output_dir=$output_dir \
    --model_ckpt=$model_checkpoint


