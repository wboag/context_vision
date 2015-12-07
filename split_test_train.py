

from collections import defaultdict
import random
import os
import sys


basedir = os.path.dirname(os.path.abspath(__file__))

# Partition the data into train and test
ALL_IMAGES   = os.path.join(basedir, 'data/meta/all_images.txt'  )
TRAIN_IMAGES = os.path.join(basedir, 'data/meta/train_images.txt')
TEST_IMAGES  = os.path.join(basedir, 'data/meta/test_images.txt' )



def main():

    # Read data
    predfile = os.path.join(basedir, 'data/pred/log/allImage_Predictions_newAnnotations_noDupla.log')
    with open(predfile, 'r') as f:
        objs = [ ObjectPred(obj_str) for obj_str in f.read().split('\n\n') ]

    imgs = list(set(obj.getName() for obj in objs))

    # Partition data into train/test
    random.shuffle(imgs)
    ind = int(len(imgs) * .6)
    train_imgs = imgs[:ind ]
    test_imgs  = imgs[ ind:]

    print '%d images' % len(      imgs)
    print '%d images' % len(train_imgs)
    print '%d images' % len( test_imgs)

    # Ensure good split
    a = len(train_imgs)
    b = len( test_imgs)
    assert a+b == len(imgs)
    assert set(train_imgs) & set(test_imgs) == set()

    # Dump ALL   image names
    if os.path.exists(ALL_IMAGES):
        print 'would you like to replace %s? (y/n) ' % ALL_IMAGES,
        choice = sys.stdin.readline().strip()
        if choice == 'y':
            with open(ALL_IMAGES, 'w') as f:
                for name in imgs:
                    print >>f, name

    # Dump TRAIN image names
    if os.path.exists(TRAIN_IMAGES):
        print 'would you like to replace %s? (y/n) ' % TRAIN_IMAGES,
        choice = sys.stdin.readline().strip()
        if choice == 'y':
            with open(TRAIN_IMAGES, 'w') as f:
                for name in train_imgs:
                    print >>f, name

    # Dump TEST  image names
    if os.path.exists(TEST_IMAGES):
        print 'would you like to replace %s? (y/n) ' % TEST_IMAGES,
        choice = sys.stdin.readline().strip()
        if choice == 'y':
            with open(TEST_IMAGES, 'w') as f:
                for name in test_imgs:
                    print >>f, name




class BoundingBox:
    def __init__(self, labels, pixels):
        box_labels = [ label      for label in labels.split(', ') ]
        box_pixels = [ int(pixel) for pixel in pixels.split(', ') ]
        for box_label,box_pixel in zip(box_labels,box_pixels):
            cmd = 'self._%s = %d' % (box_label,box_pixel)
            exec(cmd)
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
        #print obj_str
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
            retStr += '%-15s%f\n' % (label,confidence)

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








if __name__ == '__main__':
    main()

