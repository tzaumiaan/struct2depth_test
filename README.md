# struct2depth_test
Apply `struct2depth` to self recorded dataset

Credits to https://github.com/tensorflow/models/tree/master/research/struct2depth

This is by far just an exploration

## Environment setup
1. Make sure the `$PYTHONPATH` includes 
[the path](https://github.com/tensorflow/models/tree/master/research/struct2depth)
for `struct2depth`.
Edit `env.sh` first then run it
```
source env.sh
```

2. Create a local link to `KITTI` dataset under the structure like
```
./kitti/
├── 2011_09_26
│   ├── 2011_09_26_drive_0005_sync
│   │   └─── image_02
│   │       └─── data
│   │           ├── *.png
│   │           └── ...
│   └── calib_cam_to_cam.txt
...
```

3. Inference on this dataset.
Edit `inference.sh` first for choosing the model checkpoints,
dataset path, and output path (default `./output`),
then run it
```
source inference.sh
```
The results will be in `./output` folder

4. Using Mask R-CNN model to generate instance segmentation mask for training.
Edit `mask_rcnn/run.sh` for choosing the model name,
dataset path, and output path (default `mask_rcnn/output`),
then follow these steps to run it
```
cd mask_rcnn
source run.sh
```
*TODO* Integrate with the next step

5. Run preparation script to generate dataset for training
```
mkdir kitti_processed
python gen_data_kitti.py
```
Then the `train.txt` containing the input list will be under `kitti_processed` folder, specifying triplets used for training

6. Run training
```
source train.sh
```
