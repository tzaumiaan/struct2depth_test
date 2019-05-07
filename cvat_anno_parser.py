from lxml import etree
from collections import defaultdict
import os
import numpy as np
import cv2

MIN_OBJ_AREA = 100

def etree_to_dict(t):
  d = {t.tag: {} if t.attrib else None}
  # traverse children
  children = list(t)
  if children:
    dd = defaultdict(list)
    for dc in map(etree_to_dict, children):
      for k, v in dc.items():
        dd[k].append(v)
    d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
  # append attributes to dictionary
  if t.attrib:
    d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
  # append text part to dictionary 
  if t.text:
    text = t.text.strip()
    if children or t.attrib:
      if text:
        d[t.tag]['#text'] = text
    else:
      d[t.tag] = text
  return d

def parse_cvat_annotation(anno_dict):
  # generate object
  object_dict = {}
  for i_ in anno_dict['annotations']['image']:
    if type(i_['box']) != list:
      i_['box'] = [i_['box']]
    object_dict[str(i_['@name'])] = {
        'width': int(i_['@width']),
        'height': int(i_['@height']),
        'box': i_['box']}
  return object_dict

def colorize(obj_id, color='rgb'):
  colorcode = int(obj_id) % 128 # mod 128 to fit 0-127
  colorcode = int('{:07b}'.format(colorcode)[::-1], 2) # scramble
  if color=='rgb':
    hue = int(float(colorcode)*180/128) # scale to fit 0-179
    sat, val = 200, 255
    hsv = np.array([[[hue,sat,val]]], dtype=np.uint8)
    rgb = np.squeeze(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
  elif color=='gray':
    gray = int(float(colorcode)*192/128 + 63) # scale fit 0-255 and avoid dark shades
    rgb = np.array([gray,gray,gray], dtype=np.uint8)
  return rgb

def draw_seg_map(object_dict, img_name, img_size, color='rgb'):
  seg_img = np.zeros(shape=(img_size[0], img_size[1], 3)) # all black as background
  if img_name in object_dict:
    img_dict = object_dict[img_name]
    assert img_size == (img_dict['height'], img_dict['width'])
    for obj_id, obj in enumerate(img_dict['box']):
      xmin = int(float(obj['@xtl']))
      ymin = int(float(obj['@ytl']))
      xmax = int(float(obj['@xbr'])+1)
      ymax = int(float(obj['@ybr'])+1)
      if (xmax - xmin)*(ymax - ymin) < MIN_OBJ_AREA:
        continue
      seg_img[ymin:ymax,xmin:xmax,:] = colorize(obj_id, color)
  return seg_img

class cvat_anno_parser():
  def __init__(self, xml_file):
    assert os.path.exists(xml_file), 'annotation file {} does not exist'.format(xml_file)
    anno_xml_root = etree.parse(xml_file).getroot()
    anno_dict = etree_to_dict(anno_xml_root)
    self.object_dict = parse_cvat_annotation(anno_dict)
  def get_seg_map(self, img_name, img_size, color='rgb'):
    return draw_seg_map(self.object_dict, img_name, img_size, color)
