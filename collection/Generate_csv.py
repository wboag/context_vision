import argparse
import xml.etree.ElementTree as ET
import os



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations'     ,
                         dest = 'annotations' ,
                         help = 'The directory containing xml annotations'
    )

    parser.add_argument('--out'       ,
                        dest = 'out'  ,
                        help = 'The output file to store the annotations'
    )

    args = parser.parse_args()

    if (not args.annotations) or (not args.out):
        parser.print_help()
        exit()

    annotations_dir = args.annotations
    output_file     = args.out
    count = 0
    fileCount = 0
    # For each object in each xml_file, make a separate CSV file
    annotations = set()
    for xml_file in os.listdir(annotations_dir):

        xml_file_path = os.path.join(annotations_dir, xml_file)
        img = ImageFile(xml_file_path)
        name = img.getName()
        numUniqueObjects = 0
        nameList = [None] * 20 #enough?
        hasDupla = False

        # This xml file has multiple objects
        if(len(img._objects) > 1):

                for myObject in img._objects:
                        # count unique entries
                        if str(myObject._name) not in nameList:
                                nameList[numUniqueObjects] = str(myObject._name)
                                numUniqueObjects += 1
                        else:
                                #print 'Dupla found in ' + img._filename + str(myObject) + '\n'
                                #print 'found ' + str(myObject._name)+ ' in ' + str(nameList)
                                hasDupla = True
                                #break
                counter = 0
                tmpNameList = [None] * 20
                # if we have more than 1 unique entry
                if numUniqueObjects > 1:
                        # run through the objects again
                        for myObject in img._objects:
                                # and be sure to only add unique names
                                if str(myObject._name) not in tmpNameList:

                                        fullPath = '/home/xorlog/Desktop/imageWork/images/'
                                        myOutput = fullPath + img._filename + str(myObject) + '\n'
                                        #create a csv file for this annotation in this image
                                        f = open(output_file + '/'+ xml_file[:-4]+ '_'+ str(counter) +'_' + '.csv', 'w')
                                        print >>f, "filename,xmin,ymin,xmax,ymax"
                                        print >>f, myOutput
                                        # keep track of the name so we can make sure to only add unique ones
                                        tmpNameList[counter] = str(myObject._name)
                                        # controls our name list and the file name.
                                        counter += 1



        del nameList[:]
        del tmpNameList[:]
        if hasDupla:
                print img._filename


    #print count
    print fileCount

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
        fullPath = '/home/xorlog/Desktop/imageWork/images/'
        # Change this function to change what is printed to the file
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

