#! /usr/bin/env python
from __future__ import absolute_import, print_function
import argparse
import cv2
import json
import os

import numpy as np
from tqdm import tqdm

from frontend import YOLO
from utils import draw_boxes

# try:
#     from mean_average_precision.detection_map import DetectionMAP
#     calc_map = True
# except ImportError:
#     calc_map = False

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    required=True,
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    required=True,
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    required=True,
    help='path to an image, a video (mp4 format), or a directory of images or videos')

argparser.add_argument(
    '-s',
    '--stats',
    action='store_true',
    help='show coordinates and scores for each result')


def predict_file(image_path, yolo, config, show_stats):
    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       50.0,
                                       (frame_w, frame_h))

        for _ in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()
    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    image_path = args.input
    show_stats = args.stats

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=config['model']['input_size'],
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    if os.path.isdir(image_path):
        for img in os.listdir(image_path):
            print('PREDICT', img)
            predict_file(os.path.join(image_path, img), yolo, config, show_stats)
    else:
        predict_file(image_path, yolo, config, show_stats)

    # TODO
    # yolo = YOLO(...)  # Create model
    # yolo.load_weights(weights_path)  # Load weights
    # mAP = DetectionMAP(num_classes)  # Initialise metric
    # for image in images:
    #     boxes = yolo.predict(image)
    #     # prepare objects pred_bb, pred_classes, pred_conf, gt_bb and gt_classes
    #     mAP.evaluate(pred_bb, pred_classes, pred_conf, gt_bb, gt_classes)  # Update the metric
    #
    # mAP.plot()  # Get the value of the metric and precision-recall plot for each class


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
