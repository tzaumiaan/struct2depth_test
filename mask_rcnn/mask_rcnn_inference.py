import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from datetime import datetime
import os
import cv2
from matplotlib.colors import hsv_to_rgb

# arguments
flags.DEFINE_string('model_dir', 'mask_rcnn_inception_v2_coco_2018_01_28', 'Model path for inference.')
flags.DEFINE_string('model_ckpt', 'model.ckpt', 'Model checkpoint name.')
flags.DEFINE_string('model_pb', 'frozen_inference_graph.pb', 'Model PB name.')
flags.DEFINE_string('data_dir', None, 'Data path for inference.')
flags.DEFINE_string('output_dir', None, 'Output path.')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('output_dir')
FL = flags.FLAGS

# model wrapper for mask rcnn
class model_wrapper(object):
  def __init__(self, using_frozen_pb=True):
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      if using_frozen_pb:
        # Load a frozen Tensorflow model into memory as graph
        t0 = datetime.now()
        graph_def_ = tf.GraphDef()
        frozen_pb = os.path.join(FL.model_dir, FL.model_pb)
        with tf.gfile.GFile(frozen_pb, 'rb') as fid:
          graph_def_.ParseFromString(fid.read())
          tf.import_graph_def(graph_def_, name='')
        t1 = datetime.now()
        logging.info('Tensorflow model loaded')
        logging.info('Elapsed time = {}sec'.format((t1-t0).total_seconds()))
      else:
        # Load model from checkpoint
        t0 = datetime.now()
        init_op = tf.global_variables_initializer()
        model_ckpt = os.path.join(FL.model_dir, FL.model_ckpt)
        model_meta = model_ckpt+'.meta'
        saver = tf.train.import_meta_graph(model_meta)
        self.sess.run(init_op)
        saver.restore(self.sess, model_ckpt)
        t1 = datetime.now()
        logging.info('Meta graph imported from {}'.format(model_meta))
        logging.info('Weights restored from'.format(model_ckpt))
        logging.info('Elapsed time = {}sec'.format((t1-t0).total_seconds()))
  
  def inference(self, image):
    h, w, c = image.shape
    image_batch = np.expand_dims(image, axis=0)
    # get operators from graph
    image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
    detection_masks = self.graph.get_tensor_by_name('detection_masks:0')
    detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.graph.get_tensor_by_name('num_detections:0')
    # graph extension
    detection_masks = reframe_masks(
        detection_boxes,
        detection_masks,
        num_detections,
        h, w)
    # run inference
    with self.graph.as_default():
      t0 = datetime.now()
      (boxes, masks, scores, classes, num) = self.sess.run(
          [detection_boxes, detection_masks, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_batch})
      t1 = datetime.now()
      num = int(num)
      logging.info('Inference time = {}sec'.format((t1-t0).total_seconds()))
      logging.info('{} objects found'.format(num))
    # post processing ...
    # purge useless dimension 
    boxes, masks, scores, classes = np.squeeze(boxes), np.squeeze(masks), np.squeeze(scores), np.squeeze(classes)
    # take only valid results
    boxes, scores, classes = boxes[:num,:], scores[:num], classes[:num]
    # x-y reorder
    boxes = boxes[:,np.array([1,0,3,2])]
    # transform from 0-1 to 0-w and 0-h
    boxes = np.multiply(boxes, np.array([w,h,w,h])).astype(np.int32)
    # merge masks to one
    seg_map = np.zeros(shape=image.shape, dtype=np.uint8)
    for n_ in range(num)[::-1]:
      seg_map[masks[n_]>0,:] = colorize(n_)
    return seg_map, boxes, scores, classes

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)

def reframe_masks(boxes_tensor, masks_tensor, num_tensor, h, w):
  # The following processing is only for single image
  b_ = tf.squeeze(boxes_tensor, [0])
  m_ = tf.squeeze(masks_tensor, [0])
  n_ = tf.cast(num_tensor[0], tf.int32)
  b_ = tf.slice(b_, [0, 0], [n_, -1])
  m_ = tf.slice(m_, [0, 0, 0], [n_, -1, -1])
  masks_reframed = reframe_box_masks_to_image_masks(m_, b_, h, w)
  masks_reframed = tf.cast(tf.greater(masks_reframed, 0.5), tf.uint8)
  # Follow the convention by adding back the batch dimension
  return tf.expand_dims(masks_reframed, 0)

def vis_frame(outfile, image, bboxes, labels, scores):
  out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  text_font = cv2.FONT_HERSHEY_SIMPLEX
  text_size = 0.7
  for i, bbox in enumerate(bboxes):
    pos_min, pos_max = (bbox[0], bbox[1]), (bbox[2], bbox[3])
    cv2.rectangle(out, pos_min, pos_max, (255,0,0),3)
    text = '{:d}({:1.3f})'.format(int(labels[i]), scores[i])
    cv2.putText(out,  text, pos_min, 
                text_font, text_size, (255,0,0), 2, cv2.LINE_AA)
  cv2.imwrite(outfile, out)

def colorize(obj_id):
  hue = int(obj_id) % 256 # mod 256 to fit 0-255
  hue = int('{:08b}'.format(hue)[::-1], 2) # scramble
  hue = int(float(hue)*180/256) # scale to fit 0-179
  sat, val = 200, 255
  hsv = np.array([[[hue,sat,val]]], dtype=np.uint8)
  rgb = np.squeeze(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
  return rgb

def main(_):
  model = model_wrapper(using_frozen_pb=False)
  if not os.path.exists(FL.output_dir):
    os.mkdir(FL.output_dir)
  
  # count numer of images
  for i_ in os.listdir(FL.data_dir):
    [image_name, image_format] = i_.split('.', 2)
    logging.info('Reading {}.{} ...'.format(image_name, image_format))
    if image_format != 'png' and image_format != 'jpg':
      continue
    t0 = datetime.now()
    image_file = os.path.join(FL.data_dir, i_)
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    seg_map, bbox_list, score_list, label_list = model.inference(image)
    outfile = '{}-fseg.png'.format(image_name)
    outfile = os.path.join(FL.output_dir, outfile)
    cv2.imwrite(outfile, seg_map)
    exec_time = (datetime.now() - t0).total_seconds()
    logging.info('Image {} finished with exec time {} sec'.format(image_name, exec_time))
  
if __name__=='__main__':
  app.run(main)
