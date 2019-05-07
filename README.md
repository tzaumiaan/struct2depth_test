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
There are 3 additional changes in original `util.py`. The first regarding `load_image()`:
```
-  im_data = np.fromstring(gfile.Open(img_file).read(), np.uint8)
+  im_data = np.fromstring(gfile.Open(img_file, 'rb').read(), np.uint8)
```
The second regarding `get_vars_to_save_and_restore()`:
```
-          not_loaded.remove(v.op.name)
+          logging.info('removing {} ...'.format(v.op.name))
+          if v.op.name in not_loaded:
+            not_loaded.remove(v.op.name)
```
And the third regarding `format_number()`:
```
-  locale.setlocale(locale.LC_ALL, 'en_US')
+  locale.setlocale(locale.LC_ALL, 'en_US.utf8')
```

2. Create a local link to `KITTI` dataset under the structure like
```
./kitti/
├── img -> 2011_09_26/2011_09_26_drive_0005_sync/image_02/data
│   ├── *.png
│   └── ...
├── calib_cam_to_cam.txt -> 2011_09_26/calib_cam_to_cam.txt
...
```

3. Inference on this dataset.
Edit `inference.sh` first for choosing the model checkpoints,
dataset path, and output path (default `./output`),
then run it
```
source inference_kitti.sh
```
The results will be in `./output` folder

4. Using Mask R-CNN model to generate instance segmentation mask for training.
Edit `mask_rcnn/run.sh` for choosing the model name,
dataset path, and output path (default `mask_rcnn/output`),
then follow these steps to run it
```
cd mask_rcnn
source run.sh
cd ..
```
Now make the soft link of segmentation results to the dataset
```
cd kitti
ln -s ../mask_rcnn/output segimg
cd ..
```
So the dataset looks like this now
```
./kitti/
├── img -> 2011_09_26/2011_09_26_drive_0005_sync/image_02/data
│   ├── *.png
│   └── ...
├── segimg -> ../mask_rcnn/output
│   ├── *-seg.png
│   └── ...
├── calib_cam_to_cam.txt -> 2011_09_26/calib_cam_to_cam.txt
...
```

5. Run preparation script to generate dataset for training
```
python gen_data_kitti.py --input_dir=kitti --output_dir=kitti_processed
```
Then the `train.txt` containing the input list will be under `kitti_processed` folder, specifying triplets used for training

6. Run training
```
source train_kitti.sh
```
