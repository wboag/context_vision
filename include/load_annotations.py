
import xml.etree.ElementTree as ET
import os
import sys


from load_predictions import Predictions




basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Partition the data into train and test
TRAIN_IMAGES = os.path.join(basedir, 'data/meta/train_images.txt')
TEST_IMAGES  = os.path.join(basedir, 'data/meta/test_images.txt' )



# map codes to interpretable name
mapfile = os.path.join(basedir, 'data/mapping/small_to_english.txt')
small_to_english = {}
with open(mapfile, 'r') as f:
    for line in f.readlines()[1:]:
        toks = line.strip().split('\t')
        small_to_english[toks[1]] = toks[2]



# Load list of images we've predicted
predfile = os.path.join(basedir, 'data/meta/all_images.txt')
with open(predfile, 'r') as f:
    predicted_names = set(f.read().strip().split('\n'))



def main():

    # Grab all predictions
    predfile = os.path.join(basedir, 'data/pred/log/allImage_Predictions_newAnnotations_noDupla.log')
    pred = Predictions(predfile)
    pred_imgs = pred.train_images() + pred.test_images()

    # Build annotations
    annotations = Annotations(pred_imgs)

    if '--sanity' in sys.argv:
        train_ann_names= set(img.getName() for img in annotations.train_images())
        test_ann_names = set(img.getName() for img in annotations.test_images() )
        annotation_names = train_ann_names | test_ann_names

        print 'predicted:   ', len(predicted_names)
        print 'annotations: ', len(annotation_names)
        print

        print 'not annotated'
        for name in predicted_names:
            #print 'candidate: ', name
            if name not in annotation_names:
                print '\t', name
        print

        assert len(predicted_names) == len(train_ann_names)+len(test_ann_names)
        print 'SUCCESS!'
        exit()

    anns = annotations.train_images() + annotations.test_images()
    for img in anns:
        print img




def get_datadirs(data):
    dirs = []
    #for datadir in os.listdir(data)[:2]:
    for datadir in os.listdir(data):
        datadir_path = os.path.join(data, datadir)
        if os.path.isdir(datadir_path):
            dirs += get_datadirs(datadir_path)
    if dirs == []:
        dirs = [data]
    return dirs


class ObjectAnnotation:
    def __init__(self, object_xml_element):
        code = object_xml_element.find('name').text
        #self._name = small_to_english[big_to_small[code]]
        self._name = small_to_english[code]
        self._xmin = int(object_xml_element.find('bndbox').find('xmin').text)
        self._xmax = int(object_xml_element.find('bndbox').find('xmax').text)
        self._ymin = int(object_xml_element.find('bndbox').find('ymin').text)
        self._ymax = int(object_xml_element.find('bndbox').find('ymax').text)
    def bounding(self):
        return (self._xmin, self._xmax, self._ymin, self._ymax)
    def __str__(self):
        return '<ObjAnn %-12s x=[%3s %3s] y = [%3s %3s]>' % (self._name,self._xmin,self._xmax,self._ymin,self._ymax)


class ImageFile:
    def __init__(self, xml_file):
        # Save path to file (because why not?)
        self._xml_file = xml_file

        # Parse XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract image info
        self._filename = root.find('filename').text
        self._folder   = root.find('folder').text
        self._database = root.find('source').find('database').text
        self._height   = root.find('size').find('height').text
        self._width    = root.find('size').find('width').text

        # Extract objects
        self._objs = []
        for obj_element in root.findall('object'):
            obj = ObjectAnnotation(obj_element)
            self._objs.append(obj)

    def getName(self):
        return self._filename

    def getObjects(self):
        return [ obj._name for obj in self._objs ]

    def __str__(self):
        retStr = ''
        retStr += self._filename + '\n'
        for obj in self._objs:
            retStr += '\t' + str(obj) + '\n'
        retStr += '\n'
        return retStr



class Annotations:

    def __init__(self, predicted_images, verbose=1):

        # Build list of object instances to include
        # object instance will be defined by IMG_NAME + bounding box
        instances = set()
        for pred_img in predicted_images:
            for obj in pred_img._objs:
                name = obj._name
                xmin = obj._bounding._xmin
                xmax = obj._bounding._xmax
                ymin = obj._bounding._ymin
                ymax = obj._bounding._ymax
                tup = (name, xmin, xmax, ymin, ymax)
                instances.add(tup)

        datadirs= [os.path.join(basedir,'data/imagenet/ILSVRC2013_DET_bbox_val')]

        # For each xml_file, collect the annotation info
        annotations = {}
        for datadir in datadirs:
            for xml_file in os.listdir(datadir):
                # Only process image file if we have a prediction for it
                assert xml_file.endswith('.xml')
                name = xml_file[:-4]
                if name not in predicted_names: continue

                xml_file_path = os.path.join(datadir, xml_file)
                img = ImageFile(xml_file_path)

                # limit the annotated instances to ONLY those predicted
                new_objs = []
                for obj in img._objs:
                    xmin = obj._xmin
                    xmax = obj._xmax
                    ymin = obj._ymin
                    ymax = obj._ymax
                    tup = (img.getName(), xmin, xmax, ymin, ymax)
                    if tup in instances:
                        new_objs.append( obj )
                img._objs = new_objs

                name = img.getName()
                if name not in annotations:
                    annotations[name] = img
                else:
                    print >>sys.stderr, 'repeat: ', name
                    exit()

        # Full list of imgs (because not everything wants the frozen train/test)
        self._images = annotations

        # partition annotations into train and test
        with open(TRAIN_IMAGES, 'r') as f:
            train = set(f.read().split())
        self._train_images = {k:v for k,v in annotations.items() if (k in train)}

        with open(TEST_IMAGES, 'r') as f:
            test = set(f.read().split())
        self._test_images  = {k:v for k,v in annotations.items() if (k in test )}

        if verbose > 0:
            print 'train images: ', len(self._train_images)
            print 'test  images: ', len(self._test_images)


    def images(self):
        return self._images.values()

    def train_images(self):
        return self._train_images.values()


    def test_images(self):
        return self._test_images.values()





if __name__ == '__main__':
    main()

