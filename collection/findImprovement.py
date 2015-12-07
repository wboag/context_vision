import re
import os
import xml.etree.ElementTree as ET
import cv2


# Match our prediction and BB with the caffe's, and the correct annotation & BB.
# See who was correct, and draw the images with BB, scores, for ours and caffe's


def main():

        # Declarations
        caffePred = open('allImage_Predictions_newAnnotations_noDupla.log', 'r')
        ourPred = open('train-svm-context.pred.txt', 'r')
        synset_words = open('det_synset_words.txt', 'r')
        annotations_dir = 'new_annotations/'
        caffePredList= []
        ourPredList = []
        betterFiles = []
        worseFiles = []
        annotationDict = {}
        nameMapping = {}
        relationDict = {}



        #precompile some regex to find filename and annotation name (not synID)
        re_fileName = re.compile('ILSVRC2.*')
        re_name = re.compile('[a-z]+(\s[a-z]+)*')



        #load the Caffe predictions
        lines = caffePred.readlines()
        print 'Building Caffe prediction list'
        for x in xrange(len(lines)):
                if re_fileName.search(lines[x]) is not None:
                        name = re_name.match(lines[x+4][:-1])
                        score = lines[x+4][-10:]
                        caffePredList.append([lines[x][:-1], name.group(0), score.strip(), lines[x+2][:-1]])



        #load our predictions with co-occurrence
        lines = ourPred.readlines()
        print 'Building our prediction list'
        for x in xrange(len(lines)):
                if re_fileName.search(lines[x]) is not None:
                        name = re_name.match(lines[x+4][:-1])
                        score = lines[x+4][-10:]

                        ourPredList.append([lines[x][:-1], name.group(0), score.strip(), lines[x+2][:-1]])


        #Build annotation dictionary with BB coords
        #Since we scraped duplicate objects we need to make sure
        #we are looking at the correct BB and image.
        #Our predictions are a subset of the caffe predictions,
        #which is a subset of the total images.
        #Lookup with file name MINUS file ext
        print 'Building annotation dictionary'
        x = 0
        for xml_file in os.listdir(annotations_dir):

                img = ImageFile(annotations_dir + xml_file)
                name = img.getName()
                objectsAndBB = []

                for myObject in img._objects:
                        #xmin, ymin, xmax, ymax
                        objectsAndBB.append([myObject._name ,
                                             myObject._xmin + ', ' +
                                             myObject._ymin + ', ' +
                                             myObject._xmax + ', ' +
                                             myObject._ymax
                                            ])
                annotationDict[xml_file[:-4]] = objectsAndBB

                #print some dots so the user doesn't think we forgot them
                if x == 2000:
                        print '.'
                        x = 0
                x += 1


        # Lookup for name->synID
        print 'Building synset dictionary'
        for line in synset_words:
                nameMapping[line[:9]] = line[10:]


        #some stats
        ourMatches = 0
        caffeMatches = 0
        total = 0

        print 'Comparing results'
        #main search.
        #Run through all of our predictions.
        for ourEntry in ourPredList:
                total += 1
                #Find it in the caffe prediction list
                for theirEntry in caffePredList:
                        #Check to see if our file name match and BB match
                        if ourEntry[0] == theirEntry[0] and ourEntry[3] == theirEntry[3]:
                                # If we found it either make a new dictionary entry for it, or append
                                # it to the end of the correct set of bounding boxes
                                if ourEntry[0] not in betterFiles:
                                        relationDict[ourEntry[0]] = [[ourEntry], [theirEntry]]
                                else:
                                        relationDict[ourEntry[0]][0].append(ourEntry)
                                        relationDict[ourEntry[0]][1].append(theirEntry)

                                # Found the match, need to find the correct answer now
                                # The annotation dictionary has a list of objects.
                                # lookup with filename minus file ext.
                                for correctEntry in annotationDict[ourEntry[0][:-5]]:
                                        # find the one that matches our BB
                                        if ourEntry[3] == correctEntry[1]:
                                                #check what the real correct answer is.
                                                ourName = ourEntry[1]
                                                theirName = theirEntry[1]
                                                correctName = str(nameMapping[correctEntry[0]]).strip()
                                                if ourName == theirName and ourName == correctName:
                                                        #print "Both correct: " + ourName
                                                        ourMatches += 1
                                                        caffeMatches += 1
                                                elif ourName == correctName:
                                                        #print "We were correct: " + ourName + '\nCaffe had: ' + theirName #+'\n'
                                                        #print ourEntry
                                                        #print str(theirEntry) + '\n'
                                                        betterFiles.append(ourEntry[0])
                                                        ourMatches += 1
                                                elif theirName == correctName:
                                                        #print "Cafe was correct: " + theirName +'\nWe had: ' + ourName +'\n'
                                                        caffeMatches += 1
                                                        worseFiles.append(theirEntry[0])
                                                #else:
                                                        #print 'Both Wrong! \nWe had: ' + ourName + '\nCaffe had: ' + theirName +'\n' + 'Answer was: ' + correctName
                #More dots.  Let the user know we STILL didn't forget them
                if total % 1000 == 0:
                        print '.'



        print '\nTotal predictions: ' + str(total)
        print 'We matched: ' + str(ourMatches) + " %" + str(float(ourMatches) / float(total) * float(100))[:6]
        print 'Caffe Matched: ' + str(caffeMatches) + " %" + str(float(caffeMatches) / float(total) * float(100))[:6]

        # for better for for worse
        #for x in worseFiles:
        for x in betterFiles:
                #print str(relationDict[x]) + '\n\n'
                imName = 'images/'+ str(x).strip()
                print imName
                img = cv2.imread('%s' % (imName), 1)

                results = relationDict[x]
                if len(results[0]) > 2:
                        for y in range(0, len(results[0])):
                                print str(results[0]) + '\n\n'
                                print str(results[1]) + '\n\n'

                                coords = re.split(',', results[0][y][3])
                                cv2.rectangle(img, (int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3])),(255,0,0),2)
                                cv2.putText(img,results[0][y][1] + ': ' + str(results[0][y][2]) ,(int(coords[0]),int(coords[1])), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,255,0),1)
                                cv2.putText(img,results[1][y][1] + ': ' + str(results[1][y][2]) ,(int(coords[0]),int(coords[1])+15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),1)


                        if img is not None:

                                cv2.imshow('main ',img)
                                cv2.waitKey(0)




class ObjectAnnotation:
    def __init__(self, object_xml_element):
        self._name = object_xml_element.find('name').text
        self._xmin = object_xml_element.find('bndbox').find('xmin').text
        self._xmax = object_xml_element.find('bndbox').find('xmax').text
        self._ymin = object_xml_element.find('bndbox').find('ymin').text
        self._ymax = object_xml_element.find('bndbox').find('ymax').text
    def __str__(self):
        return '.JPEG,%3s,%3s,%3s,%3s' % (self._xmin,self._ymin,self._xmax,self._ymax)


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
        self._objects = []
        for obj_element in root.findall('object'):
            obj = ObjectAnnotation(obj_element)
            self._objects.append(obj)

    def getName(self):
        return self._filename

    def getObjects(self):
        return [ obj._name for obj in self._objects ]

    def __str__(self):


        if False:
            retStr = ''
            retStr += self._filename
            retStr += '\t' + ','.join( [ obj._name for obj in self._objects ] )
        else:
            retStr = ''
            for obj in self._objects:
                retStr += fullPath + self._filename + str(obj) + '\n'
        return retStr




if __name__ == '__main__':
    main()


