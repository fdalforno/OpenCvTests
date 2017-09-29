import cv2
import opencv
import dscc
import time
import argparse
import sys
import numpy as np

from random import randint

import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image to scan")
args = vars(ap.parse_args())

print("OpenCV version :  {0}".format(cv2.__version__))

#-i C:\PycharmProjects\Lavoro\cai\Cai2804\20170427_094918_20170427_094918_CAI.png
#-i C:\PycharmProjects\Lavoro\cai\Cai2804\20170427_095509_F3.JPG
#-i C:\PycharmProjects\Lavoro\cai\Cai2804\20170427_144309_FOTO0008.jpg
#-i C:\PycharmProjects\Lavoro\cai\Cai2704\20160429_185152_CAI.png
#-i C:\PycharmProjects\Lavoro\cai\Cai2704\20160429_180445_CID009920016024504.png

image = args["image"]

document = opencv.open_document(image,1000)


if document is None:
    print("document ", document, " doesn't exists")
    sys.exit()

#testImage = cv2.cvtColor(document, cv2.COLOR_GRAY2RGB)

height, width = document.shape[:2]
testImage = np.zeros((height,width,3), np.uint8)
testImage[:,:] = (255,255,255)

pagemask, page_outline = opencv.get_page_extents(document)
binary = opencv.get_binary(document,pagemask)

start = time.time()
runLens = dscc.calcRunLen(binary,0)
end = time.time()
print(end - start)


#print(runLens)

i = 0
for runLen in runLens:
    color = [randint(0,255),randint(0,255),randint(0,255)]

    if runLen.length > 10:

        runLen.print()

        while runLen is not None:
            testImage[runLen.midpoint, runLen.pos] = color

            #testImage[runLen.pos,runLen.midpoint ] = color

            runLen = runLen.child



runLens = dscc.calcRunLen(binary,1)

for runLen in runLens:
    color = [randint(0,255),randint(0,255),randint(0,255)]

    if runLen.length > 5:

        runLen.print()

        while runLen is not None:
            #testImage[runLen.midpoint, runLen.pos] = color
            if not runLen.bad:
                testImage[runLen.pos,runLen.midpoint ] = color

            runLen = runLen.child




plt.imshow(testImage)
plt.show()



