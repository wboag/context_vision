#!/usr/bin/python


# Created to find and draw predictions from the detector output which
# meet certain threshold criteria.  ie: find images where the top 2 predicted scores
# are within a certain range of each other or above/below, some threshold.

import re
import numpy as np
import cv2


def main():

        fIn = open('allImage_Predictions_newAnnotations_noDupla.log', 'r')
        predictions =   [None] * 10
        junk = [None] * 7
        while True:
                name = fIn.readline()
                if not name:
                        break
                junk[0] = fIn.readline()
                junk[1] = fIn.readline()
                junk[2] = fIn.readline() #coords
                for x in range(0, 10):
                        predictions[x] = fIn.readline()
                junk[3] = fIn.readline()
                junk[4] = fIn.readline()
                junk[5] = fIn.readline()
                #junk[6] = fIn.readline()
                score_1 = re.search('-?\d*\.\d+', predictions[0])
                score_2 = re.search('-?\d*\.\d+', predictions[1])
                score_1 = float(score_1.group(0))
                score_2 = float(score_2.group(0))
                diff = score_1 - score_2

                # Make sure we have relatively low confidence scores
                # with a difference within .1 of each other.
                if abs(diff) < .1 and score_1 > -.5 and score_1 < 0:
                        print name,
                        print junk[0], junk[1], junk[2],

                        for x in range(0, 10):
                                print predictions[x],

                        print junk[3],junk[4], junk[5],
                        print '\n'

                        # Load image & draw BB
                        loc = 'images/'+ name
                        img = cv2.imread('images/%s'% (name[:-1]), 1)
                        coords = re.split(',', junk[1])
                        cv2.rectangle(img,(int(coords[0]),int(coords[1])),(int(coords[2]),int(coords[3])),(255,0,0),2)

                        cv2.imshow('image',img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()




main()

