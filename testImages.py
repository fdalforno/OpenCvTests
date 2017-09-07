import cv2
import opencv
import numpy as np
import argparse
import glob

import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="folder to scan")
args = vars(ap.parse_args())

print("OpenCV version :  {0}".format(cv2.__version__))


for imagePath in glob.glob(args["folder"]):
    document = opencv.open_document(imagePath,1000)
    testImage = cv2.cvtColor(document, cv2.COLOR_GRAY2RGB)

    if document is None:
        print("document ", document, " doesn't exists")
        continue


    pagemask, page_outline = opencv.get_page_extents(document)
    contours = opencv.get_contours(document, pagemask,'block')

    for contour in contours:
        contour.draw_orientation(testImage)

    cv2.imshow("TestImage",testImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()