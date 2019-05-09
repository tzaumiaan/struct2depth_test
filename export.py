# this script will read in the trained model checkpoint
# and export the model as the frozen PB and meta/data set 

import model, nets, util # from struct2depth

import os
import tensorflow as tf
from absl import app, flags, logging

flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to evaluate.')
flags.DEFINE_string('export_dir', None, 'Directory to export the model for inference.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_enum('architecture', nets.RESNET, nets.ARCHITECTURES,
                  'Defines the architecture to use for the depth prediction '
                  'network. Defaults to ResNet-based encoder and accompanying '
                  'decoder.')
flags.DEFINE_boolean('imagenet_norm', True, 'Whether to normalize the input '
                     'images channel-wise so that they match the distribution '
                     'most ImageNet-models were trained on.')
flags.DEFINE_bool('use_skip', True, 'Whether to use skip connections in the '
                  'encoder-decoder architecture.')
flags.DEFINE_bool('joint_encoder', False, 'Whether to share parameters '
                  'between the depth and egomotion networks by using a joint '
                  'encoder architecture. The egomotion network is then '
                  'operating only on the hidden representation provided by the '
                  'joint encoder.')
flags.mark_flag_as_required('model_ckpt')
flags.mark_flag_as_required('export_dir')
FL = flags.FLAGS

def export(_):
  # create the model graph
  inference_model = model.Model(
      is_training=False,
      batch_size=FL.batch_size,
      img_height=FL.img_height,
      img_width=FL.img_width,
      seq_length=FL.seq_length,
      architecture=FL.architecture,
      imagenet_norm=FL.imagenet_norm,
      use_skip=FL.use_skip,
      joint_encoder=FL.joint_encoder)
  # restore from check point
  vars_to_restore = util.get_vars_to_save_and_restore(FL.model_ckpt)
  saver = tf.train.Saver(vars_to_restore)
  with tf.Session() as sess:
    saver.restore(sess, FL.model_ckpt)
    if not tf.gfile.Exists(FL.export_dir):
      tf.gfile.MakeDirs(FL.export_dir)
    logging.info('Model will be exported to {}'.format(FL.export_dir))
    exported_ckpt_path = os.path.join(FL.export_dir, 'model.ckpt')
    saver.save(sess, exported_ckpt_path)
    logging.info('Checkpoint for inference written out: {}'.format(exported_ckpt_path))
    # define I/O for inference
    output_node_names = [ 
        'depth_prediction/raw_input', # input image
        'depth_prediction/add_3'] # output disparity
    #output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(output_node_names)
    # freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)
    # Save the frozen graph
    exported_pb_path = os.path.join(FL.export_dir, 'frozen_inference_graph.pb')
    with tf.gfile.GFile(exported_pb_path, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
    logging.info('PB graph for inference written out: {}'.format(exported_pb_path))

if __name__ == '__main__':
  app.run(export)
