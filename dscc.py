import numpy as np
import math

HORIZONTAL = 0
VERTICAL = 1

MIN_LEN = 1

class RunLen(object):
    def __init__(self,pos,start,end):
        self.pos = pos

        self.parents = 0
        self.children = 0

        self.length = 0
        self.medium_height = 0

        self.child = None
        self.see = False

        self.bad = False

        self.coeff = []

        if start < end:
            self.start = start
            self.end = end
        else:
            self.start = end
            self.end = start

        self.midpoint = self.start + int((self.end - self.start) / 2)

    def get_overlap(self,start,end):
        return max(0, min(self.end, end) - max(self.start, start)) > 0

    def singleConnected(self):
        if self.children == 1 and self.parents == 1:
            return True
        else:
            return False

    def prune(self):
        self.child = None

    def connect(self,child):
        if  self.pos == child.pos - 1 and self.get_overlap(child.start,child.end):

            child.parents += 1
            self.children += 1

            self.child = child

            return True
        else:
            return False

    def getHeight(self):
        return self.end - self.start


    def print(self):
        print("Runlen {0} \t {1} \t {2}".format(self.pos,self.length,self.medium_height))

    def project(self,x):
        return  self.coeff[0] * x + self.coeff[1]






def lineDistance(a,b):
    return max(a.pos, b.pos) - min(a.pos + a.length,b.pos + b.length)

def colineDistance(a,b):
    dx = lineDistance(a,b)
    if dx <= 0:
        return np.inf
    else:
        run = b
        sum = 0
        M = 0

        while run is not None:
            if not run.bad:
                sum += ((run.end + run.start) / 2 - a.project(run.pos))**2
                M += 1
            run = run.child

        if M > 0:
            return dx  + sum / M
        else:
            return np.inf


def merenge(runLen):
    for i in range(len(runLen)):
        for j in range(len(runLen)):
            if i != j:
                dx = lineDistance(runLen[i],runLen[j])
                dcc = colineDistance(runLen[i],runLen[j])

                print(i,j,dcc - dx, math.sqrt(dcc - dx) < runLen[j].length)


def fitLine(run):
    if run.length < 2:
        return

    start = run

    pos = []
    midPoints = []

    while run is not None:
        if run.getHeight() <= 2 * start.medium_height:
            pos.append(run.pos)
            midPoints.append(run.midpoint)
        else:
            run.bad = True
            #print("Scartato punto {0} : altezza {1} media {2}".format(run.pos,run.getHeight(),start.medium_height))

        run = run.child

    if(len(pos) >= 2):
        coeff = np.polyfit(pos, midPoints, 1, full=False)
        start.coeff = coeff


def navigate(run):
    length = 0
    medium_height = 0

    #print("Start {0}".format(run.pos))

    if run.child is None:
        #print("End {0}".format(run.pos))
        run.length = 1
        run.medium_height = run.getHeight()
        return 1

    start = run


    while run is not None and run.singleConnected():
        run.see = True

        length += 1
        medium_height += run.getHeight()

        endRun = run
        run = run.child

    start.length = length
    start.medium_height = int(medium_height / length)

    #calcoloFunzione di interpolazione
    #z = np.polyfit(x, y, 3)

    endRun.prune()
    return length

    #print("End {0} {1}".format(endRun.pos,length))



def detectSingleConnections(runLen):

    singleConnection = []

    for row in runLen:
        for run in row:
            if not run.singleConnected():
                run.prune()

            if run.singleConnected() and not run.see:
                if navigate(run) > MIN_LEN:
                    singleConnection.append(run)
                    fitLine(run)

    return singleConnection




def connect(parents,children):

   for i in range(len(parents)):
        for j in range(len(children)):
            parent = parents[i]
            child = children[j]
            parent.connect(child)


def calcRunLen(image,direction = 0):
    runLen = []
    (h, w) = image.shape[:2]

    c = h
    if direction == 0:
        c = w

    for i in range(c):
        #print("riga numero {0}".format(i))

        if direction == 0:
            data = image[:,i]
        else:
            data = image[i,:]

        bounded = np.hstack(([0], data, [0]))
        difs = np.diff(bounded)
        run_starts, = np.where(difs > 0)
        run_ends, = np.where(difs < 0)


        row = []

        for rle in zip(run_starts,run_ends):
            row.append(RunLen(i,rle[0],rle[1]))

        runLen.append(row)

        if i > 0:
            connect(runLen[i - 1], runLen[i])



    return detectSingleConnections(runLen)