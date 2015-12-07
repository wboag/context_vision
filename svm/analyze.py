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
    parser.add_argument("--model",
        help = "Where to load model",
        dest = "model",
    )
    args = parser.parse_args()

    # Error check args
    if not args.model or not os.path.exists(os.path.dirname(args.model)):
        print >>sys.stderr, '\n\tERROR: must provide where model is stored\n'
        parser.print_help()
        exit(1)

    # Load the model
    print 'loading ', args.model
    with open(args.model, 'rb') as f:
        m = pickle.load(f)

    # Predict improved labels
    m.analyze()




if __name__ == '__main__':
    main()
