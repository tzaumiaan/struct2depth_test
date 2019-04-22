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
# |    |--- *.png
# |--- segimg
# |    |--- *-seg.png
# |--- calib_cam_to_cam.txt

import os
from absl import app, flags, logging
import numpy as np
import cv2
from alignment import align

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

# parser for calibration file
def get_line(file, start):
  file = open(file, 'r')
  lines = file.readlines()
  lines = [line.rstrip() for line in lines]
  ret = None
  for line in lines:
    nline = line.split(': ')
    if nline[0]==start:
      ret = nline[1].split(' ')
      ret = np.array([float(r) for r in ret], dtype=float)
      ret = ret.reshape((3,4))[0:3, 0:3]
      break
  file.close()
  return ret

def img_scale(img, segimg=None, cam_intr=None, target_h=HEIGHT, target_w=WIDTH):
  old_h, old_w, _ = img.shape
  new_img = cv2.resize(img, (target_w, target_h))
  if segimg is not None:
    assert segimg.shape[0:1]==img.shape[0:1], 'image size mismatch'
    # resize using INTER_NEAREST to avoid confusion for alignment
    new_segimg = cv2.resize(segimg, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
  if cam_intr is not None:
    zoom_x = float(target_w)/old_w
    zoom_y = float(target_h)/old_h
    new_cam_intr = cam_intr.copy()
    new_cam_intr[0, 0] *= zoom_x
    new_cam_intr[0, 2] *= zoom_x
    new_cam_intr[1, 1] *= zoom_y
    new_cam_intr[1, 2] *= zoom_y
  return new_img, new_segimg, new_cam_intr

def main(_):
  train_dir = os.path.join(FL.output_dir, 'train')
  if not os.path.exists(FL.output_dir):
    os.mkdir(FL.output_dir)
    if not os.path.exists(train_dir):
      os.mkdir(train_dir)
  
  f_tr = open(os.path.join(FL.output_dir, 'train.txt'), 'w') 
  file_calibration = os.path.join(FL.input_dir, 'calib_cam_to_cam.txt')
  calib_camera = get_line(file_calibration, 'P_rect_02')
  imgs = sorted(os.listdir(os.path.join(FL.input_dir, 'img')))
  logging.info('Total {} images in {}'.format(len(imgs), FL.input_dir))
  
  ct = 1
  triplet, seg_triplet = [], []
  for i in range(0, len(imgs), STEPSIZE):
    img_file = os.path.join(FL.input_dir, 'img', imgs[i])
    segimg_file = os.path.join(FL.input_dir, 'segimg', imgs[i].replace('.png', '-seg.png'))
    logging.info('Processing {} ...'.format(img_file))
    img = cv2.imread(img_file)
    if os.path.exists(segimg_file):
      segimg = cv2.imread(segimg_file, 0) # read as grayscale
    else:
      segimg = np.zeros(shape=(img.shape[0], img.shape[1])) # all black
    img, segimg, cam_intr = img_scale(img, segimg, calib_camera)
    calib_representation = ','.join([str(c) for c in cam_intr.flatten()])
    triplet.append(img)
    seg_triplet.append(segimg)
    # if there are enough frames for a triplet
    if len(triplet)==3:
      output_name = str(ct).zfill(10)
      cmb = np.hstack(triplet)
      #align1, align2, align3 = seg_triplet[0], seg_triplet[1], seg_triplet[2]
      align1, align2, align3 = align(seg_triplet[0], seg_triplet[1], seg_triplet[2])
      cmb_seg = np.hstack([align1, align2, align3])
      cv2.imwrite(os.path.join(train_dir, output_name + '.png'), cmb)
      cv2.imwrite(os.path.join(train_dir, output_name + '-fseg.png'), cmb_seg)
      f = open(os.path.join(train_dir, output_name + '_cam.txt'), 'w')
      f.write(calib_representation)
      f.close()
      f_tr.write('{} {}\n'.format('train', output_name))
      del triplet[0]
      del seg_triplet[0]
      ct+=1
  f_tr.close() 

if __name__ == '__main__':
  app.run(main)
