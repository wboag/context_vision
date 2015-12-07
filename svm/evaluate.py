#-------------------------------------------------------------------------------
# Name:        evaluate.py
#
# Purpose:     Evaluate predictions based on precision, recall, and specificity
#
# Author:      Willie Boag
#-------------------------------------------------------------------------------



import os
import sys
import argparse



# Add include paths to my files to load data
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
includedir = os.path.join(basedir, 'include')
if includedir not in sys.path:
    sys.path.append(includedir)


import load_predictions
import load_annotations


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",
        help = "Predictions file",
        dest = "pred",
    )
    parser.add_argument("--frac",
        help = "Whether displayed P and R should be fractions or decimals",
        dest = "frac",
        action = 'store_true'
    )
    args = parser.parse_args()

    # Error check
    if not args.pred:
        print >>sys.stderr, '\n\tERROR: must provide predictions file\n'
        parser.print_help()
        exit()

    # load predictions
    all_pred = load_predictions.Predictions(args.pred, verbose=0)
    test_pred = all_pred.images()

    # load annotations
    all_gold = load_annotations.Annotations(test_pred)
    test_gold = all_gold.images()


    # Ensure you're looking at the same set of images
    predicted_names = set([ img.getName() for img in test_pred ])
    annotated_names = set([ img.getName() for img in test_gold ])
    common_names = predicted_names & annotated_names

    #print 'predictions: ', len(predicted_names)
    #print 'annotations: ', len(annotated_names)
    #print 'common:      ', len(common_names)

    pred_imgs = {i.getName():i for i in test_pred if i.getName() in common_names}
    gold_imgs = {i.getName():i for i in test_gold if i.getName() in common_names}

    #print 'predictions: ', len(pred_imgs)
    #print 'annotations: ', len(gold_imgs)

    # Get all possible labels (should be 200 total)
    mapfile = os.path.join(basedir, 'data/mapping/small_to_english.txt')
    ind = {}
    with open(mapfile, 'r') as f:
        for line in f.readlines()[1:]:
            toks = line.strip().split('\t')
            ind[toks[2]] = len(ind)

    # Aggregate TP, FP, FN, and TN across every image
    confusion = create_confusion(pred_imgs, gold_imgs, common_names, ind)

    # attempt to visualize
    display_confusion(confusion, ind, args.frac)


def create_confusion(pred_imgs, gold_imgs, common_names, ind):
    confusion = [ [ 0 for _ in ind ] for __ in ind ]

    for name in common_names:
        #print name
        #print
        #print pred_imgs[name]
        #for obj in pred_imgs[name]._objs:
        #    print obj
        #print
        #print gold_imgs[name]

        # For each object, evaluate performance
        for pred in pred_imgs[name]._objs:
            # Find the corresponding gold object instance (based on bbox)
            pred_bounding = pred._bounding.bounding()
            r = None
            for cand_gold_obj in gold_imgs[name]._objs:
                if pred_bounding == cand_gold_obj.bounding():
                    r = cand_gold_obj._name
            assert r, 'Couldnt find gold: %s -- %s' % (name, str(pred_bounding))

            predicted_label = pred._preds[0][0]
            reference_label = r
            confusion[ ind[predicted_label] ][ ind[reference_label] ] += 1

        #print '\n\n\n'
        #exit()

    return confusion




def display_confusion(confusion, ind, print_frac):

    labels = ind   # hash tabble: label -> index

    pad = max(len(l) for l in labels) + 6

    '''
    # Display the confusion matrix
    print ""
    print ""
    print ""
    print "================"
    print "PREDICTION RESULTS"
    print "================"
    print ""
    print "Confusion Matrix"
    print "%10s %10s" % (' ' * pad, "\t".join(labels.keys()))
    for act, act_v in labels.items():
        print "%10s %10s" % (act.rjust(pad), "\t\t\t".join([str(confusion[act_v][pre_v]) for pre, pre_v in labels.items()]))
    print ""
    '''


    print "Analysis"
    print " " * (pad-3), "Precision   Recall       F1     Accuracy"


    # Compute the analysis stuff
    correct = 0
    total   = 0
    precision = []
    recall    = []
    f1        = []
    accuracy  = []
    for lab, lab_v in sorted(labels.items()):
        tp = confusion[lab_v][lab_v]
        fp = sum(confusion[lab_v][v] for k, v in labels.items() if v != lab_v)
        fn = sum(confusion[v][lab_v] for k, v in labels.items() if v != lab_v)

        tn = sum(confusion[v1][v2] for k1, v1 in labels.items()
          for k2, v2 in labels.items() if v1 != lab_v and v2 != lab_v)

        correct = tp + tn
        total   = tp + tn + fp + fn

        p = float(tp)    / (tp + fp           + 1e-100)
        r = float(tp)    / (tp + fn           + 1e-100)
        f = float(2*p*r) / (p  + r            + 1e-100)
        a = float(tp+tn) / (tp + fp + fn + tn + 1e-100)

        number_predicted = tp + fp
        actual_positive  = tp + fn

        prec = '%d/%d' % (tp, number_predicted)
        rec  = '%d/%d' % (tp, actual_positive)

        # Don't count the 0-annotation, 0-prediction rows against yourself
        #if number_predicted or actual_positive:
        if actual_positive:
        #if True:
            precision += [p]
            recall    += [r]
            f1        += [f]
            accuracy  += [a]

            if print_frac:
                print "%s %-10s%-10s%-10.4f%-10.4f"%(lab.ljust(pad),prec,rec,f,a)
            else:
                print "%s %-10.4f%-10.4f%-10.4f%-10.4f" %(lab.ljust(pad),p,r,f,a)

    print "--------\n"


    print 'Macro-averaged precision: ', sum(precision) / len(precision)
    print 'Macro-averaged recall:    ', sum(recall)    / len(recall)
    print 'Macro-averaged F-score:   ', sum(f1)        / len(f1)
    print 'Accuracy:                 ', float(correct) / total
    print




if __name__ == '__main__':
    main()
