# data generation for self recorded dataset, adapted from 
# https://github.com/tensorflow/models/blob/master/research/struct2depth/gen_data_kitti.py
# ====
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====
# suitable for the following data structure
# input_dir
# |--- img
# |    |--- [clip_name]
# |    |    |--- *.jpg
# |    |--- [clip_name].xml
# |    | ...
# |    |--- train.txt
# |--- calib.txt
#
# note:
# 1. the annotation xml is of CVAT format
# 2. the calibration file is of the following format
#    fx,0,cx,0,fy,cy,0,0,1
#    k1,k2,p1,p2,k3

import os
from absl import app, flags, logging
import numpy as np
import cv2
from alignment import align
from cvat_anno_parser import cvat_anno_parser

# arguments
flags.DEFINE_string('input_dir', None, 'Dataset root path.')
flags.DEFINE_string('output_dir', None, 'Output path.')
flags.mark_flag_as_required('input_dir')
flags.mark_flag_as_required('output_dir')
FL = flags.FLAGS

# constants
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1


def get_lines(filename):
  with open(filename, 'r') as f:
    lines = f.read().splitlines()
  return lines

def get_camera_intrinsic(filename):
  lines = get_lines(filename)
  cam_intr = np.array(lines[0].split(',')).astype(np.float)
  cam_intr = np.reshape(cam_intr, (3,3))
  return cam_intr

def img_scale(img, segimg=None, cam_intr=None, target_h=HEIGHT, target_w=WIDTH):
  old_h, old_w, _ = img.shape
  new_img = cv2.resize(img, (target_w, target_h))
  if segimg is not None:
    assert segimg.shape[0:1]==img.shape[0:1], 'image size mismatch'
    # resize using INTER_NEAREST to avoid confusion for alignment
    new_segimg = cv2.resize(segimg, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
  else:
    new_segimg = None
  if cam_intr is not None:
    zoom_x = float(target_w)/old_w
    zoom_y = float(target_h)/old_h
    new_cam_intr = cam_intr.copy()
    new_cam_intr[0, 0] *= zoom_x
    new_cam_intr[0, 2] *= zoom_x
    new_cam_intr[1, 1] *= zoom_y
    new_cam_intr[1, 2] *= zoom_y
  else:
    new_cam_intr = None
  return new_img, new_segimg, new_cam_intr

def main(_):
  train_dir = os.path.join(FL.output_dir, 'train')
  if not os.path.exists(FL.output_dir):
    os.mkdir(FL.output_dir)
    if not os.path.exists(train_dir):
      os.mkdir(train_dir)
  
  f_tr = open(os.path.join(FL.output_dir, 'train.txt'), 'w') 
  file_calibration = os.path.join(FL.input_dir, 'calib.txt')
  calib_camera = get_camera_intrinsic(file_calibration)
  file_train_list = os.path.join(FL.input_dir, 'img', 'train_list.txt')
  train_list = get_lines(file_train_list)
  
  for clip in train_list:
    imgs = sorted(os.listdir(os.path.join(FL.input_dir, 'img', clip)))
    logging.info('Total {} images in clip {}'.format(len(imgs), clip))
    clip_output_dir = os.path.join(train_dir, clip)
    if not os.path.exists(clip_output_dir):
      os.mkdir(clip_output_dir)
    
    # initialize CVAT annotation parser for bounding boxes
    xml_file = os.path.join(FL.input_dir, 'img', clip+'.xml')
    anno_parser = cvat_anno_parser(xml_file)

    ct = 1
    triplet, seg_triplet = [], []
    for i in range(0, len(imgs), STEPSIZE):
      img_file = os.path.join(FL.input_dir, 'img', clip, imgs[i])
      logging.info('Processing {} ...'.format(img_file))
      img = cv2.imread(img_file)
      segimg = anno_parser.get_seg_map(imgs[i], (img.shape[0], img.shape[1]), color='gray')
      img, segimg, cam_intr = img_scale(img, segimg, calib_camera)
      calib_representation = ','.join([str(c) for c in cam_intr.flatten()])
      triplet.append(img)
      seg_triplet.append(segimg)
      # if there are enough frames for a triplet
      if len(triplet)==3:
        output_name = str(ct).zfill(10)
        cmb = np.hstack(triplet)
        align1, align2, align3 = align(seg_triplet[0], seg_triplet[1], seg_triplet[2])
        cmb_seg = np.hstack([align1, align2, align3])
        cv2.imwrite(os.path.join(clip_output_dir, output_name + '.png'), cmb)
        cv2.imwrite(os.path.join(clip_output_dir, output_name + '-fseg.png'), cmb_seg)
        f = open(os.path.join(clip_output_dir, output_name + '_cam.txt'), 'w')
        f.write(calib_representation)
        f.close()
        f_tr.write('{} {}\n'.format(os.path.join('train', clip), output_name))
        del triplet[0]
        del seg_triplet[0]
        ct+=1
  f_tr.close() 

if __name__ == '__main__':
  app.run(main)
