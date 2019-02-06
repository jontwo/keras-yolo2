"""
Show images in matplotlib with annotation drawn on top

Created on 24th December 2018

@author: jonmorris
"""
# -*- coding:utf-8 -*-
from __future__ import print_function, absolute_import

# Python Stdlib Imports
import argparse
import json
import os

# Third-party app imports
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Project imports

# CONSTANTS
IMAGE_TYPES = ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tif', '.tiff']
PARSER_HELP_STR = "Show annotation for training images"


def is_image(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_TYPES


def add_rect_to_ax(ax, x, y, w, h):
    rectr = patches.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2)
    rectk = patches.Rectangle((x, y), w, h, fill=False)
    ax.add_patch(rectr)
    ax.add_patch(rectk)


def plot_image(filename, bbox, separate=False):
    img = plt.imread(filename)
    cols = img.shape[-1] if separate else 1
    xsize, ysize = plt.rcParams.get('figure.figsize')
    xsize *= cols * 0.6
    fig, ax = plt.subplots(1, cols, figsize=[xsize, ysize])
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax'] - bbox['ymin']
    print('\nShowing', filename)
    print('Bbox', bbox)
    print('Image max: {} min: {} mean: {} ({}) sd: {} ({})'.format(img.max(), img.min(), img.mean(),
          img.mean() / img.max(), img.std(), img.std() / img.max()))
    if cols == 1:
        ax.set_title(filename)
        ax.imshow(img)
        add_rect_to_ax(ax, bbox['xmin'], bbox['ymin'], width, height)
    else:
        fig.suptitle(filename)
        for i in range(cols):
            band = img[..., i]
            print('Band {} max: {} min: {} mean: {} ({}) sd: {} ({})'.format(
                i, band.max(), band.min(), band.mean(), band.mean() / band.max(),
                band.std(), band.std() / band.max()))
            ax[i].imshow(band, cmap='gray')
            add_rect_to_ax(ax[i], bbox['xmin'], bbox['ymin'], width, height)
    plt.show()


def main():
    argparser = argparse.ArgumentParser(description=PARSER_HELP_STR)

    argparser.add_argument('-c', '--conf', default='config.json',
                           help='path to configuration file')
    argparser.add_argument('-d', '--imgdir', default='.',
                           help='path to image directory')
    argparser.add_argument('-s', '--suffix', default='',
                           help='image file suffix')
    argparser.add_argument('-m', '--multi_image', action='store_true',
                           help='show multi-band images in separate plots')

    args = argparser.parse_args()
    conf_type = os.path.splitext(args.conf)[1]
    if conf_type == '.json':
        from preprocessing import parse_annotation
        with open(args.conf) as cfb:
            config = json.loads(cfb.read())
        train_imgs, _ = parse_annotation(config['train']['train_annot_folder'],
                                         config['train']['train_image_folder'],
                                         config['model']['labels'])
    elif conf_type == '.txt':
        filenames = [x for x in os.listdir(args.imgdir) if is_image(x)]
        if not filenames:
            print('No images found in {}'.format(args.imgdir))
            return
        train_imgs = []
        with open(args.conf) as txt:
            for line in txt.readlines():
                item = {}
                data = line.strip().split(' ')
                poss_imgs = [f for f in filenames if data[0] in f]
                print(len(poss_imgs), 'found for', data[0])
                if args.suffix:
                    poss_imgs = [f for f in poss_imgs if args.suffix in f]
                    if poss_imgs:
                        item['filename'] = os.path.join(args.imgdir, poss_imgs[0])
                    else:
                        print('No images found with suffix {}'.format(args.suffix))
                else:
                    item['filename'] = os.path.join(args.imgdir, poss_imgs[0])
                item['object'] = [zip(
                    ['xmin', 'ymax', 'xmax', 'ymin'], [float(x) for x in data[5:]]
                )]
                train_imgs.append(item)
        print(train_imgs)
    else:
        print('Invalid config file {}'.format(args.conf))
        return

    for imobj in train_imgs:
        plot_image(imobj['filename'], imobj['object'][0], separate=args.multi_image)


if __name__ == '__main__':
    main()
