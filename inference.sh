#!/bin/sh
tf_model_root=~/workspace/models/research/struct2depth

input_dir=kitti/img/
output_dir=output
#output_dir=output_0422_2_24700
model_checkpoint="./model_kitti/model-199160"
#model_checkpoint="./model_ckpt_bak/2/model-24700"

python $tf_model_root/inference.py \
    --logtostderr \
    --file_extension=png \
    --depth \
    --egomotion=True \
    --input_dir=$input_dir \
    --output_dir=$output_dir \
    --model_ckpt=$model_checkpoint


