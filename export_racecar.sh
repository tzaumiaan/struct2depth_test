#!/bin/sh
export_dir=model_racecar_0508_30080
model_checkpoint=model_ckpt_bak/0508/model-30080

python export.py \
    --logtostderr \
    --export_dir=$export_dir \
    --model_ckpt=$model_checkpoint


