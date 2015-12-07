#!/usr/bin/env python

# Based on detect.py from caffe.  Heavily modified to read a directory CSV files
# that have image names bounding box info, classify each BB, then send the file name,
# BB coords, and top 10 predictions to STD out.  Redirect this to create a log.


import numpy as np
import pandas as pd
import os
import argparse
import time

import caffe

CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir, "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir, "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--crop_mode",
        default="list",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    parser.add_argument(
        "--input_dir",
        default='/home/xorlog/caffe/python/myCSVinput/',
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_dir",
        default='/home/xorlog/caffe/python/myDetectOutput/',
        help="Ourput Directory"
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")



# Modified code start

    # Make detector.
    detector = caffe.Detector(args.model_def, args.pretrained_model, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)

    # Read our labels
    with open('../data/ilsvrc12/det_synset_words.txt') as f:
    #with open('../data/ilsvrc12/synset_words.txt') as f:
        labels_df = pd.DataFrame([
        {
            'synset_id': l.strip().split(' ')[0],
            'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
        ])

    labels_df.sort('synset_id')


    for inputFile in os.listdir(args.input_dir):
        outputFile = args.output_dir + inputFile[:-3]+'h5'
        inputFile = args.input_dir + inputFile

        print'\n'
            # Load input.
        t = time.time()
        if inputFile.lower().endswith('csv'):
            inputs = pd.read_csv(inputFile, sep=',', dtype={'filename': str})
            inputs.set_index('filename', inplace=True)
        else:
            raise Exception("Unknown input file type: not in txt or csv.")

            # Detect.
        if args.crop_mode == 'list':
            # Unpack sequence of (image filename, windows).
            images_windows = [(ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
                    for ix in inputs.index.unique()
                ]
            detections = detector.detect_windows(images_windows)
        else:
            detections = detector.detect_selective_search(inputs)



        # Collect into dataframe with labeled fields.
        df = pd.DataFrame(detections)
        df.set_index('filename', inplace=True)
        df[COORD_COLS] = pd.DataFrame(data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)


        myOut = str(df['window'])
        #COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']
        myOut = myOut[47:]
        scores = myOut.split()
        print scores[0]
        print 'xmin, ymin, xmax, ymax'
        print scores[2],
        print scores[1][1:],
        print scores[4][:-1] +',',
        print scores[3][:-1]

        predictions_df = pd.DataFrame(np.vstack(df.prediction.values), columns=labels_df['name'])
        max_s = predictions_df.max(0)
        max_s.sort(ascending=False)
        print(max_s[:10])


        del(df['window'])

# Modified code end


if __name__ == "__main__":
    import sys
    main(sys.argv)

