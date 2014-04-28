import numpy as np
import numpy.random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import sys

from PIL import Image

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

def applyNoise(ptFrequency):
  # Add uniform noise
  # numNoisePts = 1000000.
  # maxFreq = max([freqs for pts in ptFrequency.values() for y,freqs in pts.iteritems()])
  # noise_x = makeUniformNoise(setSize, numNoisePts)
  # noise_y = makeUniformNoise(setSize, numNoisePts)
  # for x,y in zip(noise_x, noise_y):
  #   ptFrequency[x][y] = ptFrequency[x][y] + 1

  noise = makeNormalNoise(40, 5, setSize)
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

# make and add the first cluster
clusterPts = 100000
xs, ys = makeCluster(160, 160, 200, 200, clusterPts)
for x,y in zip(xs,ys):
  ptFrequency[x][y] = ptFrequency[x][y] + 1

# make and subtract the second cluster
xs, ys = makeCluster(400, 400, 200, 200, clusterPts)
for x,y in zip(xs,ys):
  ptFrequency[x][y] = ptFrequency[x][y] - 1

# create and add noise
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
    #rgb = (i, j, 100)
    normVal = int((ptFrequency[i][j] / float(maxFreq)) * 255)
    rgb = (normVal,normVal,normVal)
    pixels[i,j] = rgb # set the colour accordingly


for i in range(img.size[1]):
  pixels[setSize/2,i] = (255,0,0)

img.show()
