#!/bin/sh
tf_model_root=~/workspace/models/research/struct2depth

input_dir=racecar/img/041
output_dir=output_racecar_0508_30080
model_checkpoint=model_ckpt_bak/0508/model-30080
#model_checkpoint="./model_kitti/model-199160"

python $tf_model_root/inference.py \
    --logtostderr \
    --file_extension=jpg \
    --depth \
    --egomotion=True \
    --input_dir=$input_dir \
    --output_dir=$output_dir \
    --model_ckpt=$model_checkpoint


