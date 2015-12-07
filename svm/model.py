#-------------------------------------------------------------------------------
# Name:        model.py
#
# Purpose:     Try a predict-based approach to disambiguation with context
#
# Author:      Willie Boag
#-------------------------------------------------------------------------------


############################################
#  Standard python libraries
############################################

import numpy as np
import scipy.sparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from collections import defaultdict




############################################
#  Code from another directory
############################################

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIR = os.path.join(BASE_DIR, 'include')
if INCLUDE_DIR not in sys.path:
    sys.path.append(INCLUDE_DIR)

from load_predictions import ObjectPred, ImagePred
from load_annotations import small_to_english




############################################
#  This code
############################################



class Model:

    def __init__(self, context=True, overfit=False):
        self._USE_CONTEXT = context
        self._OVERFIT     = overfit
        #self._architecture = 'skip-gram'
        self._architecture = 'cbow'
        if self._OVERFIT:
            self._architecture = 'nope'

        # Which labels to look at during analysis
        self._interest_labels = ['snowmobile', 'ski', 'dumbbell', 'computer mouse', 'laptop', 'chair', 'pencil box', 'pencil sharpener', 'rubber eraser', 'pizza']



    def train(self, pred_imgs, gold_imgs):
        # JUST IN CASE - make sure deterministic traversal through array
        pred_imgs = pred_imgs.items()

        # Get all possible labels (should be 200 total)
        labels = set(small_to_english.values())
        self._ind = { label:i for i,label in enumerate(labels) }
        self._rev_ind = { v:k for k,v in self._ind.items() }

        if self._OVERFIT:
            # Make a mapping to uniquely identify each object instance
            self._instance_vocab = {}
            for img_name,img in pred_imgs:
                for obj in img._objs:
                    key = (img_name, obj._bounding.bounding())
                    self._instance_vocab[key] = len(self._instance_vocab)

        # Build 400-dimensional (context-enhanced) feature vectors
        X,Y = self.feature_extraction(pred_imgs, gold_imgs)

        '''
        print 'X: '
        print  X
        print 'Y: ', Y
        '''

        # This will be useful when interpreting the sklearn's learned weights
        self._labels_set = sorted(set(Y.tolist()))
        self._labels_set_ind = {ind:i for i,ind in enumerate(self._labels_set)}

        # Send data to ML library
        print '\tbegin training Logistic Regression'
        self._model = LinearSVC()
        self._model.fit(X, Y)
        print '\tend   training Logistic Regression'

        # build frequency table of labels
        self._freq_labels = defaultdict(int)
        for img in gold_imgs.values():
            labels = [ obj._name for obj in img._objs]
            for label in labels:
                self._freq_labels[label] += 1

        # build explicit count co-occurence matrix
        if self._USE_CONTEXT:
            # Count co-occurences
            counts = defaultdict(lambda:defaultdict(lambda:1e-9))
            for img in gold_imgs.values():
                collection = [obj._name for obj in img._objs]
                no_dups = list(set(collection))
                for i in range(len(no_dups)):
                    # this label
                    lab1 = no_dups[i]

                    # all remaining labels after removing one of these labels
                    rest = list(collection)
                    rest.remove(no_dups[i])
                    rest = list(set(rest))
                    for j in range(len(rest)):
                        lab2 = rest[j]
                        counts[lab1][lab2] += 1

            # Build submatrix of counts
            m = len(self._labels_set)
            n = len(self._ind)
            self._cooccur = np.zeros( (m,n) )
            for i,ind1 in enumerate(self._labels_set):
                for lab2,j in self._ind.items():
                    lab1 = self._rev_ind[ind1]
                    self._cooccur[i,j] = counts[lab1][lab2]

            '''
            # normalize the cooccurrence counts (to be weights)
            cond = self._cooccur / np.sum(self._cooccur, axis=0)
            avg = np.sum(cond) / (cond.shape[0] * cond.shape[1])
            normed_cooccur = cond - avg

            print 'setting context weights to explicit counts'
            #self._model.coef_[:,200:] = normed_cooccur * 0.5
            self._model.coef_[:,200:] = np.zeros( (196,200) )

            params  = self._model.coef_
            this    = params[:,:200]
            context = params[:,200:]

            print np.sum(this)
            print np.std(this)
            print
            print np.sum(context)
            print np.std(context)
            print

            #exit()
            '''




    def predict(self, pred_imgs):
        # JUST IN CASE - make sure deterministic traversal through array
        pred_imgs = pred_imgs.items()

        # Build 400-dimensional (context-enhanced) feature vectors
        X,_ = self.feature_extraction(pred_imgs, gold_imgs=None)

        # Predict labels
        #Y = self._model.predict(X)
        #'''
        all_confs = self._model.decision_function(X)
        top10_predictions = []
        for datapoint in all_confs.tolist():
            confs = []
            for i,ind in enumerate(self._labels_set):
                conf = datapoint[i]
                label = self._rev_ind[ind]
                confs.append( (label,conf) )
            top10 = sorted(confs, key=lambda t:t[1], reverse=True)[:10]
            top10_predictions.append( top10 )
        #'''

        #print 'X: '
        #print X
        #Y = self._model.predict(X)
        #print 'Y: ', Y

        # Package these predictions back into a structured format
        #updated_pred_imgs = self.format_predictions(pred_imgs, Y)
        updated_pred_imgs = self.format_predictions(pred_imgs, top10_predictions)
        return updated_pred_imgs



    def feature_extraction(self, pred_imgs, gold_imgs=None):

        # list to store each 400-dimensional datapoint
        design_matrix = []
        data_labels   = []
        for img_name,img in pred_imgs:
            vecs = [ self.pred2vec(dict(obj._preds)) for obj in img._objs ]
            N = len(vecs)
            for i,obj in enumerate(img._objs):

                # NOTE: this doesn't work :(   -- not sure why; it should
                if self._architecture == 'skip-gram':
                    # Grab the caffe prediction vector (200 dim)
                    this    = vecs[i]

                    # For each image in the context, create a 400-dim datapoint
                    for j in range(N):
                        if i != j:
                            context = vecs[j]
                            # this datapoint has a 400-dimensional feature vector
                            feature = np.concatenate( (this, context) )
                            #feature = context
                            design_matrix.append(feature)

                    # This is the desired label
                    if gold_imgs:
                        # Identify WHICH object instance we're looking at using bbox
                        this_bounding = obj._bounding.bounding()
                        label = None
                        for gold_obj in  gold_imgs[img_name]._objs:
                            if this_bounding == gold_obj.bounding():
                                label = gold_obj._name
                        assert label!=None, 'Could not find annotation: %s -- %s' % (img_name, str(this_bounding))

                        # One label for each (this,context) pair
                        y = self._ind[label]
                        for j in range(N):
                            if i != j:
                                data_labels.append(y)

                else:
                    if self._architecture == 'cbow':
                        # Grab the caffe prediction vector (200 dim)
                        this    = vecs[i]

                        if self._USE_CONTEXT:
                            # compute BoW context vector (200 dim)
                            context = np.zeros(this.shape[0])
                            for j in range(N):
                                if i != j:
                                    context += vecs[j]
                            context /= N-1

                            # this datapoint has a 400-dimensional feature vector
                            #feature = context
                            feature = np.concatenate( (this, context) )

                        else:
                            # this datapoint has a 400-dimensional feature vector
                            feature = this

                    else:
                        # one-hot
                        bow_feats = [ 0 for _ in self._instance_vocab ]
                        key = (img_name, obj._bounding.bounding())
                        bow_feats[ self._instance_vocab[key] ] = 1

                        feature = np.array( [1] + bow_feats )

                    design_matrix.append(feature)

                    # This is the desired label
                    if gold_imgs:
                        # Identify WHICH object instance we're looking at using bbox
                        this_bounding = obj._bounding.bounding()
                        label = None
                        for gold_obj in  gold_imgs[img_name]._objs:
                            if this_bounding == gold_obj.bounding():
                                label = gold_obj._name
                        assert label!=None, 'Could not find annotation: %s -- %s' % (img_name, str(this_bounding))

                        y = self._ind[label]
                        data_labels.append(y)

        X = np.matrix(design_matrix)
        Y = np.array(data_labels)

        print
        print '\tX: ', X.shape
        print

        return X,Y


    #def format_predictions(self, pred_imgs, Y):
    def format_predictions(self, pred_imgs, top10_predictions):

        # Need to pop from this one-by-one
        #Y = Y.tolist()

        updated_pred_imgs = []
        for name,img in pred_imgs:
            objs = []

            # Build N object predictions for this image
            N = len(img._objs)
            for i in range(N):
                # get object
                obj = img._objs[i]
                xmin, xmax, ymin, ymax = obj._bounding.bounding()

                # Build object string (for ObjectPred constructor)
                obj_str  = ''
                obj_str += '%s.JPEG\n' % name
                obj_str += 'xmin, ymin, xmax, ymax\n'
                obj_str += '%3d, %3d, %3d, %3d\n' % (xmin,ymin,xmax,ymax)
                obj_str += 'name\n'

                #'''
                # print top-10 confidence scores
                this_object = top10_predictions.pop(0)
                for label,confidence in this_object:
                    obj_str += '%30s    %f\n' % (label,confidence)
                for _ in range(10-len(this_object)):
                    obj_str += '%30s    %f\n' % ('NULL',-1000000)
                #'''

                #this_label = Y.pop(0)
                #for i in range(10):
                #    obj_str += '%30s    %f\n' % (self._rev_ind[this_label],1.)

                obj_str += 'dtype: float32'

                '''
                print obj
                print 'this_label: ', this_label
                print 'this_label: ', self._rev_ind[this_label]
                print
                '''

                # Build an ObjectPred object
                obj_pred = ObjectPred(obj_str)
                objs.append( obj_pred )

            # Build an ImagePred object
            img = ImagePred(objs)
            updated_pred_imgs.append(img)

        return updated_pred_imgs


    def pred2vec(self, pred):
        vec = np.zeros(len(self._ind))
        for label,score in pred.items():
            vec[ self._ind[label] ] = np.exp(score)
        vec /= np.sum(vec)
        return vec


    def analyze(self):

        print 'conext: ', self._USE_CONTEXT
        print 'params: ', self._model.coef_.shape
        print 'intercept: ', self._model.intercept_.shape

        if self._USE_CONTEXT:
            params  = self._model.coef_

            this    = params[:,:200]
            context = params[:,200:]
            #context = params

            #normed = np.exp(context)
            #normed = 100 * normed / np.sum(normed)

            # Find the pairs with the strongest correlation
            top_values = []
            for i,ind in enumerate(self._labels_set):
                label_i = self._rev_ind[ind]
                for label_j,j in self._ind.items():
                    t = (label_i,label_j)
                    s = context[i,j]
                    #s = this[i,j]
                    if len(top_values) < 20:
                        top_values.append( (t,s) )
                    else:
                        to_beat = min([tup[1] for tup in top_values])
                        if s > to_beat:
                            top_values = [v for v in top_values if v[1]>to_beat]
                            top_values.append( (t,s) )

            for (lab1,lab2),score in sorted(top_values,key=lambda t:t[1]):
                print lab1
                print lab2
                print score
                print
            print '\n\n'

            '''
            # labels of interest
            labels = set()
            for (lab1,lab2),score in sorted(top_values,key=lambda t:t[1]):
                labels.add(lab1)
                labels.add(lab2)
            labels = list(labels)
            '''

            V = len(self._interest_labels)

            # submatrix of parameter weights for interest labels
            param_submatrix = [ [ 0 for _ in range(V) ] for __ in range(V) ]
            for i in range(V):
                for j in range(V):
                    lab1 = self._interest_labels[i]
                    lab2 = self._interest_labels[j]
                    ind1 = self._labels_set_ind[self._ind[lab1]]
                    ind2 = self._ind[lab2]
                    param_submatrix[i][j] = context[ind1, ind2]
                    #param_submatrix[i][j] = this[ind1, ind2]
            param_submatrix = np.array(param_submatrix)


            # submatrix of normed counts for interest labels
            count_submatrix = [ [ 0 for _ in range(V) ] for __ in range(V) ]
            for i in range(V):
                for j in range(V):
                    lab1 = self._interest_labels[i]
                    lab2 = self._interest_labels[j]
                    ind1 = self._labels_set_ind[self._ind[lab1]]
                    ind2 = self._ind[lab2]
                    count_submatrix[i][j] = self._cooccur[ind1, ind2]
            count_submatrix = np.array(count_submatrix)

            # normalize the cooccurrence counts matrix
            cond_c = count_submatrix / np.sum(count_submatrix, axis=0)
            avg_count = np.sum(cond_c) / (cond_c.shape[0] * cond_c.shape[1])
            normed_count = cond_c - avg_count


            # print info
            #print submatrix
            #print self._normed_cooccur
            print self._interest_labels
            print
            print 'counts'
            #print count_submatrix.tolist()
            print normed_count.tolist()
            print
            print '\n\n'
            print
            print 'weights'
            print param_submatrix.tolist()
            print
            print '\n\n'


        '''
        # Look at the bias terms (and whether they correspond to frequency)
        bias = self._model.intercept_

        ordered = sorted(enumerate(bias), key=lambda t:t[1])
        N = 24
        valsets = [ordered[:N], ordered[100-N/2:100+N/2], ordered[-N:]]
        for valset in valsets:
            for ind,val in valset:
                label_ind = self._labels_set[ind]
                label = self._rev_ind[label_ind]
                label_freq = self._freq_labels[label]
                print '%8.5f  %6d  %s' % (val,label_freq,label)
            print '\n'
        '''




