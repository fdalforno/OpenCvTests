import cv2
import opencv
from os.path import basename,splitext
import glob
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
ap.add_argument("-d", "--dataset", required=True, help="Path to make dataset")
ap.add_argument("-a", "--area", required=False, help="Area to match template ")
args = vars(ap.parse_args())

template = opencv.openMatchImage(args["template"])
tH,tW = template.shape[:2]

matchTemplates = []


area = []
if args.get('area'):
    sarea = args.get('area').split(';')
    for a in sarea:
        area.append(int(a))

for imagePath in glob.glob(args["images"] + "/*.*"):

    print(imagePath)
    base = basename(imagePath)
    file, ext = splitext(base)

    image = opencv.openDocument(imagePath, 1000)
    if image is None:
        print("Image ", image, " doesn't exists")
        continue
    height, width = image.shape[:2]

    matchImage = image
    if len(area) == 4:
        matchImage = image[area[0]:area[1],area[2]:area[3]]

    boxImage = opencv.simpleBox(matchImage, template)
    cv2.rectangle(matchImage, boxImage[0],boxImage[1],100,2)

    cv2.imshow('image', matchImage)

    k = cv2.waitKey(0)
    if k == ord('q'):
        match = [base,boxImage[0],boxImage[1]]
        matchTemplates.append(match)
    elif k == ord('s'):


    print(matchTemplates)

    cv2.destroyAllWindows()

    '''
    tempImage = opencv.simpleMatch(matchImage,template)
    tempFile = args["dataset"] + file + ".png"
    cv2.imwrite(tempFile, tempImage)
    '''