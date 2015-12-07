#-------------------------------------------------------------------------------
# Name:        predict.py
#
# Purpose:     Try a predict-based approach to disambiguation with context
#
# Author:      Willie Boag
#-------------------------------------------------------------------------------


############################################
#  Standard python libraries
############################################

import cPickle as pickle
import argparse



############################################
#  Code from this directory
############################################

from model import Model



############################################
#  Code from another directory
############################################

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import load_predictions




############################################
#  This code
############################################



def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",
        help = "Predictions file",
        dest = "pred",
        default = '../data/pred/log/test.caffe'
    )
    parser.add_argument("--model",
        help = "Where to load model",
        dest = "model",
    )
    parser.add_argument("--out",
        help = "Output prediction file",
        dest = "out",
    )
    args = parser.parse_args()

    # Error check args
    if not args.pred or not os.path.exists(args.pred):
        print >>sys.stderr, '\n\tERROR: must provide predictions file\n'
        parser.print_help()
        exit(1)
    if not args.model or not os.path.exists(os.path.dirname(args.model)):
        print >>sys.stderr, '\n\tERROR: must provide where to store model\n'
        parser.print_help()
        exit(1)
    if not args.out or not os.path.exists(os.path.dirname(args.out)):
        print >>sys.stderr, '\n\tERROR: must provide where to new predictions\n'
        parser.print_help()
        exit(1)


    # load predictions
    all_pred = load_predictions.Predictions(args.pred, verbose=0)
    test_pred = all_pred.images()
    pred_imgs = {img.getName():img for img in test_pred}


    print 'predictions: ', len(pred_imgs)

    # TODO - arg for where to load model
    # Load the model
    print 'loading ', args.model
    with open(args.model, 'rb') as f:
        m = pickle.load(f)

    # Predict improved labels
    updated_pred_imgs = m.predict(pred_imgs)

    # TODO - arg to dump to file
    print 'dumping predictions to ', args.out
    with open(args.out, 'w') as f:
        for img in sorted(updated_pred_imgs, key=lambda img:img.getName()):
            print >>f, img.dump()



if __name__ == '__main__':
    main()
