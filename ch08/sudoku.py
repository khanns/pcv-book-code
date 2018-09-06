# -*- coding: utf-8 -*-
from PIL import Image

from pylab import *
from scipy.ndimage import measurements
from PCV.tools import imtools
from PCV.geometry import homography
from svmutil import *

from scipy import ndimage

# helper function for geometric_transform
def warpfcn(x):
    x = array([x[0],x[1],1])
    xt = dot(H,x)
    xt = xt/xt[2]
    return xt[0],xt[1]


imname = '../data/sudoku_images/sudokus/sudoku8.jpg'
im = array(Image.open(imname).convert('L'))

# mark corners
figure()
imshow(im)
gray()
x = ginput(4)

# top left, top right, bottom right, bottom left
fp = array([array([p[1],p[0],1]) for p in x]).T
tp = array([[0,0,1],[0,1000,1],[1000,1000,1],[1000,0,1]]).T

# estimate the homography
H = homography.H_from_points(tp,fp)

# warp image with full perspective transform
im_g = ndimage.geometric_transform(im,warpfcn,(1000,1000))

imsave('test.png', im_g)

figure()
imshow(im_g)