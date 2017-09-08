import cv2
import sys
import os
import numpy as np
import math

import matplotlib.pyplot as plt

ADAPTIVE_WINSZ = 55
PAGE_MARGIN_X = 10  # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 10  # reduced px to ignore near T/B edge
TEXT_MIN_WIDTH = 15  # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2  # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5  # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour





def rescale(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    scalatura dell'immagine
    INTER_AREA buono per creare immagini piu' piccole
    INTER_LINEA buono per creare immagini piu' grandi
    :param image: immagine da lavorare
    :param width: larghezza immagine
    :param height: altezza
    :param inter: metodo scalatura
    :return: 
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def image_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

def open_document(image, doc_height):
    """
    apro il documento e lo imposto sui 1000 px di altezza
    :return:
    :param image:
    :param docHeight:
    :return:
    """
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = gray.shape[:2]
    print("image size", width, "x", height)

    img = None
    if (height > doc_height):
        img = rescale(gray, None, doc_height)
    elif (height < doc_height):
        img = rescale(gray, None, doc_height, inter=cv2.INTER_LINEAR)
    else:
        img = gray

    return img


def get_page_extents(image):
    """
    Creo una maschera su cui lavorare il documento
    :param image: pagina caricata
    :return:
    """
    height, width = image.shape[:2]
    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width - PAGE_MARGIN_X
    ymax = height - PAGE_MARGIN_Y
    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_mask(image, pagemask, maxVal=255, masktype='block'):
    mask = None

    if masktype == 'text':
        mask = cv2.adaptiveThreshold(image, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25)
        mask = cv2.dilate(mask, box(9, 1))
        mask = cv2.erode(mask, box(1, 3))
    else:
        mask = cv2.adaptiveThreshold(image, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ,7)
        mask = cv2.erode(mask, box(3, 1), iterations=3)
        mask = cv2.dilate(mask, box(8, 2))

    return np.minimum(mask, pagemask)


class ContourInfo(object):
    def __init__(self, contour, rect):
        self.contour = contour
        self.rect = rect
        self.calc_center_angle()

    def calc_center_angle(self):
        moments = cv2.moments(self.contour)

        # area e contro di massa dell'immagine
        area = moments['m00']
        self.x = moments['m10'] / area
        self.y = moments['m01'] / area

        # da capire
        moments_matrix = np.array([
            [moments['mu20'], moments['mu11']],
            [moments['mu11'], moments['mu02']]
        ]) / area

        _, svd_u, _ = cv2.SVDecomp(moments_matrix)

        tangent = svd_u[:, 0].flatten().copy()

        print("svd_u")
        print(svd_u)
        print("svd_u[:, 0]")
        print(svd_u[:, 0])
        print("svd_u[:, 0].flatten()")
        print(svd_u[:, 0].flatten())
        print("svd_u[:, 0].flatten().copy()")
        print(svd_u[:, 0].flatten().copy())

        self.angle = np.arctan2(tangent[1], tangent[0])


    def draw_contour(self,image):
        cv2.drawContours(image, self.contour, -1, (0, 255, 0), 3)

    def draw_orientation(self, image):
        length = 30
        cv2.circle(image, (int(self.x), int(self.y)), 3, (0, 0, 255), -1)
        x1 = self.x + length * math.cos(self.angle)
        y1 = self.y + length * math.sin(self.angle)

        #print(math.degrees(self.angle))

        cv2.arrowedLine(image,(int(self.x), int(self.y)),(int(x1), int(y1)),(0, 0, 255),3)

    def angleDiff(self,angleB,angleA):
        return abs(angleB - angleA)

    def calcDistance(self,cInfo):
        pass

    def calcCost(self,cInfo):
        cInfoA = self
        cInfoB = cInfo

        if cInfoA.x > cInfoB.x:
            tmp = cInfoA
            cInfoA = cInfoB
            cInfoB = tmp

        delta_x = cInfoB.x - cInfoA.x
        delta_y = cInfoB.y - cInfoA.y

        overallAngle = np.arctan2(delta_y,delta_x)
        delta_angle = max(self.angleDiff(cInfoA.angle,overallAngle),self.angleDiff(cInfoB.angle,overallAngle))





def get_contours(image, pagemask, masktype):
    mask = get_mask(image, pagemask, masktype=masktype)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect
        if (width < TEXT_MIN_WIDTH or
                    height < TEXT_MIN_HEIGHT or
                    width < TEXT_MIN_ASPECT * height):
            continue

        contours_out.append(ContourInfo(contour, rect))

    return contours_out


def detect_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

def deskew(image,contours,m):

    x = np.array([])
    angle = np.array([])
    outlier = np.array([])

    for contour in contours:
        x = np.append(x, np.array([contour.x]))
        angle = np.append(angle, np.array([contour.angle]))

    ouliers = detect_outliers(angle, m)

    c = np.where(ouliers == False)
    correctAngle = angle[c]

    #plt.scatter(x, angle, c=1 * ouliers)
    #plt.show()

    degree = np.degrees(np.mean(correctAngle)) * 1

    #print(degree)

    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree , 1)
    return cv2.warpAffine(image, M, (cols, rows),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def openMatchImage(image):
    """
    apro l'immagine con cui fare il matching
    :param image: 
    :return: 
    """
    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = gray.shape[:2]
    print("image size", width, "x", height)
    return gray


def simpleMatch(image, template):
    """
    eseguo un match semplice del template sull'immagine utile per creare il dataset iniziale
    :param image:
    :param template:
    :return: immagine estratta più simile al template
    """
    tH, tW = template.shape[:2]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    return image[maxLoc[1]:maxLoc[1] + tH, maxLoc[0]:maxLoc[0] + tW]


def simpleBox(image, template):
    """
     eseguo un match semplice del template sull'immagine ed estraggo il bounding box
    :param image:
    :param template:
    :return:
    """

    tH, tW = template.shape[:2]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    print(maxLoc)

    top_left = maxLoc
    bottom_right = (top_left[0] + tW, top_left[1] + tH)

    return [top_left, bottom_right]
