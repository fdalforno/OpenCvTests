import cv2
import opencv
import dscc
import time
import argparse
import sys

from random import randint


if __name__ == "__main__":
    print("OpenCV version :  {0}".format(cv2.__version__))

    gray = cv2.imread('./test/cai.png', cv2.IMREAD_GRAYSCALE)


    if gray is None:
        print("document doesn't exists")
        sys.exit()




    testImage = cv2.bitwise_not(gray)
    runLens = dscc.calcRunLen(testImage)


    for runLen in runLens:
        runLen.print()

    dscc.merenge(runLens)

    cv2.imshow("TestImage", testImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




