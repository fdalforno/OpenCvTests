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


for imagePath in glob.glob(args["folder"] + "*"):
    document = opencv.open_document(imagePath,1000)

    if document is None:
        print("document ", document, " doesn't exists")
        continue


    pagemask, page_outline = opencv.get_page_extents(document)
    contours = opencv.get_contours(document, pagemask,'block')

    deskew = opencv.deskew(document,contours,m=0.2)

    cv2.imshow("Deskew",deskew)
    cv2.waitKey(0)
    cv2.destroyAllWindows()