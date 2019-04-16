# struct2depth_test
Apply `struct2depth` to self recorded dataset

Credits to https://github.com/tensorflow/models/tree/master/research/struct2depth

This is by far just an exploration

## Environment setup
1. Make sure the `$PYTHONPATH` includes the path of [Tensorflow models github](https://github.com/tensorflow/models/tree/master/)
and path for `struct2depth`
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
3. Inference on this dataset
```
source inference.sh
```
The results will be in `./output` folder
4. Run preparation script to generate dataset for training
```
mkdir kitti_processed
python gen_data_kitti.py
```
Then the `train.txt` containing the input list will be under `kitti_processed` folder, specifying triplets used for training
5. Run training
```
source train.sh
```
