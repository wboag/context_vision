

from collections import defaultdict
import random
import os
import sys


basedir = os.path.dirname(os.path.abspath(__file__))



def main():

    # Load all images
    predfile = os.path.join(basedir, 'data/pred/log/allImage_Predictions_newAnnotations_noDupla.log')
    pred = Predictions(predfile)

    all_images = pred._images.values()

    CV_DIR = os.path.join(basedir, 'data/cv')

    contents = os.listdir(CV_DIR)
    if contents:
        print 'WARNING: there are already files in ', CV_DIR
        print 'delete these files? (y/n) ',
        choice = sys.stdin.readline().strip()
        if choice != 'y':
            exit()
        print 'erasing contents of ', CV_DIR
        for content in os.listdir(CV_DIR):
            content = os.path.join(CV_DIR, content)
            if os.path.isdir(content):
                os.rmdir(content)
            else:
                os.remove(content)

    # 5-fold Cross Validation
    random.shuffle(all_images)
    N = 5
    folds = [ [] for _ in range(N) ]
    for i,img in enumerate(all_images):
        folds[i%N].append(img)
    for i in range(N):
        test  = folds[i]
        train = reduce(lambda a,b:a+b, folds[:i] + folds[i+1:])

        train_file = os.path.join(CV_DIR, '%d-train.txt'%(i+1))
        print 'train: ', train_file
        with open(train_file, 'w') as f:
            for img in train:
                print >>f, img.dump()

        test_file  = os.path.join(CV_DIR, '%d-test.txt' %(i+1))
        print 'test:  ', test_file
        with open(test_file, 'w') as f:
            for img in test:
                print >>f, img.dump()
        print


    '''
    test_imgs = pred.test_images()
    for img in test_imgs:
        print img.dump()
    '''




class BoundingBox:
    def __init__(self, labels, pixels):
        box_labels = [ label      for label in labels.split(', ') ]
        box_pixels = [ int(pixel) for pixel in pixels.split(', ') ]
        for box_label,box_pixel in zip(box_labels,box_pixels):
            cmd = 'self._%s = %d' % (box_label,box_pixel)
            exec(cmd)
    def bounding(self):
        return (self._xmin, self._xmax, self._ymin, self._ymax)
    def __str__(self):
        retStr = ''
        retStr += '<BoundingBox: '
        for label,value in vars(self).items():
            retStr += '%s:%3d, ' % (label[1:],value)
        retStr = retStr[:-2]
        retStr += '>'
        return retStr


class ObjectPred:
    def __init__(self, obj_str):
        #print obj_str.strip()
        #print obj_str.strip().split('\n')
        #print len(obj_str.strip().split('\n'))
        #print '\n\n'
        #exit()
        #lines = obj_str.split('\n')
        lines = obj_str.strip().split('\n')
        assert len(lines) == 15, lines

        assert lines[0].endswith('.JPEG')
        self._name = lines[0][:-5]

        self._bounding = BoundingBox(lines[1], lines[2])

        self._preds = []
        for obj_line in lines[4:-1]:
            toks = obj_line.split()
            label = ' '.join(toks[:-1])
            confidence = float(toks[-1])
            #print 'label=[%s] confidence=[%f]' % (label,confidence)
            self._preds.append( (label,confidence) )


    def getName(self):
        return self._name


    def __str__(self):
        retStr = ''
        retStr += '%s\n' % self.getName()
        retStr += '%s\n' % str(self._bounding)
        for label,confidence in self._preds:
            retStr += '\t%-20s%f\n' % (label,confidence)
        return retStr

    def dump(self):
        xmin = self._bounding._xmin
        ymin = self._bounding._ymin
        xmax = self._bounding._xmax
        ymax = self._bounding._ymax

        # Build object string
        retStr  = ''
        retStr += '%s.JPEG\n' % self.getName()
        retStr += 'xmin, ymin, xmax, ymax\n'
        retStr += '%d, %d, %d, %d\n' % (xmin,ymin,xmax,ymax)
        retStr += 'name\n'

        for label,confidence in self._preds:
            retStr += '%-25s%f\n' % (label,confidence)

        retStr += 'dtype: float32'
        return retStr




class ImagePred:
    def __init__(self, objs):
        self._name = objs[0].getName()
        self._objs = objs

    def getName(self):
        return self._name

    def dump(self):
        retStr = ''
        for obj in self._objs:
            retStr += obj.dump() + '\n\n\n'
        return retStr

    def __str__(self):
        retStr = ''
        retStr = '<Image: %s (%d predictions)>'% (self.getName(),len(self._objs))
        return retStr




class Predictions:
    def __init__(self, predfile, verbose=1):
        # Save path to file (because why not?)
        self._file = predfile

       # Read object predictions from file
        with open(predfile, 'r') as f:
            text = f.read().strip()
            objs = [ ObjectPred(obj_str) for obj_str in text.split('\n\n') if obj_str.split() ]

        # Collect same-image objects
        collected = defaultdict(list)
        for obj in objs:
            collected[obj.getName()].append(obj)

        # Screen away bad predictions (AKA we didn't pick up multiple objs)
        if verbose > 0:
            print
            for name,preds in collected.items():
                if len(preds) < 2:
                    print '\tWARNING: image %s has <2 predictions' % name
            print

        # Build ImagePred representations
        self._images = {}
        for name,pred_objs in collected.items():
            self._images[name] = ImagePred(pred_objs)







if __name__ == '__main__':
    main()

