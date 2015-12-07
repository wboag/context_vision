#-------------------------------------------------------------------------------
# Name:        train.py
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
import load_annotations




############################################
#  This code
############################################


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",
        help = "Predictions file",
        dest = "pred",
    )
    parser.add_argument("--model",
        help = "Where to store model",
        dest = "model",
    )
    parser.add_argument("--no-context",
        help = "Should context features be disabled",
        dest = "no_context",
        action = 'store_true',
    )
    parser.add_argument("--overfit",
        help = "Should the model do an overfitting sanity check?",
        dest = "overfit",
        action = 'store_true'
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

    # load predictions
    all_pred = load_predictions.Predictions(args.pred, verbose=0)
    train_pred = all_pred.images()

    # load annotations
    all_gold = load_annotations.Annotations(train_pred)
    train_gold = all_gold.images()


    # Ensure you're looking at the same set of images
    predicted_names = set([ img.getName() for img in train_pred ])
    annotated_names = set([ img.getName() for img in train_gold ])
    common_names = predicted_names & annotated_names

    print 'predictions: ', len(predicted_names)
    print 'annotations: ', len(annotated_names)
    print 'common:      ', len(common_names)

    pred_imgs= {i.getName():i for i in train_pred if i.getName() in common_names}
    gold_imgs= {i.getName():i for i in train_gold if i.getName() in common_names}

    print 'predictions: ', len(pred_imgs)
    print 'annotations: ', len(gold_imgs)

    # Build classifier that generates context features
    m = Model(context=not args.no_context, overfit=args.overfit)
    m.train(pred_imgs, gold_imgs)

    # TODO - arg for where to serialize model
    # Serialize the model
    print 'serializing ', args.model
    with open(args.model, 'wb') as f:
        pickle.dump(m, f)




if __name__ == '__main__':
    main()
