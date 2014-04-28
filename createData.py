import numpy as np
import numpy.random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import sys

from PIL import Image

# returns RGB value given a [0,1] value
def getRGBValue(c):
  def interpolate(c1, c2, s):
    def d(i):
      return c2[i]-int((c2[i]-c1[i])*s)

    final = (d(0), d(1), d(2))
    return final

  G = (56, 180, 73)
  B = (27, 117, 187)
  Y = (255, 241, 0)

  #return interpolate(B,Y,c)

  if c < .5:
    return interpolate(B,G,c*2)
  else:
    return interpolate(Y,B,(c-.5)*2)



# from http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def makeUniformNoise(maxVal, numPts):
  scaleX = np.vectorize(lambda x: float(int(x*maxVal)))
  return scaleX(np.random.random(size=numPts))

# For each point draws value on a normal distribution
def makeNormalNoise(center,stddev,imgLen):
  return [[np.random.normal(center,stddev) for i in range(imgLen)] for j in range(imgLen)]

def makeCluster(xMean, yMean, xCovar, yCovar, numPts):
  means = [xMean, yMean]
  covar = [[xCovar, 0],
           [0, yCovar]]
  x, y = np.random.multivariate_normal(means, covar, numPts).T
  makeInt = np.vectorize(lambda x: float(int(x)))
  return makeInt(x),makeInt(y)

def makeIrregularCluster(xMean, yMean,xCovar, yCovar, numPts, numCentroids):
  if numCentroids < 1:
    return
  xs,ys = makeCluster(xMean, yMean,xCovar, yCovar, numPts)

  if numCentroids < 2:
    return
  for c in range(numCentroids-1):
    xSign = -1 if np.random.random() < .5 else 1
    ySign = -1 if np.random.random() < .5 else 1
    # xSign = ySign = 1

    xOffset = np.random.random_integers(min(xMean/10, xCovar/10), xCovar/4)
    yOffset = np.random.random_integers(min(yMean/10, yCovar/10), yCovar/4)
    tmpXMean = xMean + (xSign * xOffset)
    tmpYMean = yMean + (ySign * yOffset)

    tmpXs, tmpYs = makeCluster(tmpXMean, tmpYMean, xCovar, yCovar, numPts)
    xs = np.concatenate((xs, tmpXs))
    ys = np.concatenate((ys, tmpYs))
  makeInt = np.vectorize(lambda x: float(int(x)))
  return makeInt(xs), makeInt(ys)

def applyNoise(ptFrequency):
  # Add uniform noise
  # numNoisePts = 1000000.
  # maxFreq = max([freqs for pts in ptFrequency.values() for y,freqs in pts.iteritems()])
  # noise_x = makeUniformNoise(setSize, numNoisePts)
  # noise_y = makeUniformNoise(setSize, numNoisePts)
  # for x,y in zip(noise_x, noise_y):
  #   ptFrequency[x][y] = ptFrequency[x][y] + 1

  noise = makeNormalNoise(40, 3, setSize)
  for x,freqX in zip(ptFrequency.keys(), noise):
    for y, freqVal in zip(ptFrequency[x], freqX):
      if ptFrequency[x][y] == baseCount:
        ptFrequency[x][y] = freqVal
      else:
        ptFrequency[x][y] = baseCount - freqVal + ptFrequency[x][y]
  return ptFrequency

# Create an artificially inflated point baseline for all possible points
ptFrequency = {}
setSize = 600
baseCount = 40

for x in range(0, setSize):
  ptFrequency[x] = {}
for x in ptFrequency.keys():
  for y in range(0, setSize):
    ptFrequency[x][y] = baseCount

################################################################################
# ADDITIVE CLUSTERS
# make and add the first cluster
clusterPts = 25000
# xs, ys = makeCluster(160, 160, 200, 200, clusterPts)
# for x,y in zip(xs,ys):
#   ptFrequency[x][y] = ptFrequency[x][y] + 1

xs, ys = makeIrregularCluster(200, 160, 200, 200, clusterPts, 6)
for x,y in zip(xs,ys):
  if x >= setSize or y >= setSize:
    continue
  ptFrequency[x][y] = ptFrequency[x][y] + 1

################################################################################
# SUBTRACTIVE CLUSTERS
# make and subtract the second cluster
# xs, ys = makeCluster(400, 400, 200, 200, clusterPts)
# for x,y in zip(xs,ys):
#   ptFrequency[x][y] = ptFrequency[x][y] - 1

xs, ys = makeIrregularCluster(380, 380, 200, 200, clusterPts, 2)
for x,y in zip(xs,ys):
  if x >= setSize or y >= setSize:
    continue
  ptFrequency[x][y] = ptFrequency[x][y] - 1

################################################################################
# NOISE
ptFrequency = applyNoise(ptFrequency)

# Create np.array for x and y; this can be improved speed-wise
xs = [x for x, pts in ptFrequency.iteritems() for y,freqs in pts.iteritems()]
ys = [y for pts in ptFrequency.values() for y,freqs in pts.iteritems()]
freqs = [freqs for pts in ptFrequency.values() for y,freqs in pts.iteritems()]
maxFreq = max(freqs)

# Create image
img = Image.new('RGB', (setSize, setSize), 'black')
pixels = img.load()

for i in range(img.size[0]):    # for every pixel:
  for j in range(img.size[1]):
    normVal = (((ptFrequency[i][j] - baseCount)/baseCount)+1)/2
    rgb = getRGBValue(normVal)
    pixels[i,j] = rgb # set the colour accordingly


for i in range(img.size[1]):
  pixels[setSize/2,i] = (255,0,0)

img.show()
