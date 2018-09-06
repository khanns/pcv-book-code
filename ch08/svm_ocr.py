# -*- coding: utf-8 -*-
# from pylab import *
from numpy import *
from svmutil import *
import pickle
from PIL import Image
from PCV.tools import imtools
import os
from scipy.ndimage import measurements

def compute_feature(im):
    """ Returns a feature vector for an
    ocr image patch. """
    # resize and remove border
    norm_im = imtools.imresize(im,(30,30))
    norm_im = norm_im[3:-3,3:-3]
    return norm_im.flatten()

def load_ocr_data(path):
    """ Return labels and ocr features for all images
    in path. """
    # create list of all files ending in .jpg
    imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    # create labels
    labels = [int(imfile.split('/')[-1][0]) for imfile in imlist]
    # create features from the images
    features = []

    for imname in imlist:
        im = array(Image.open(imname).convert('L'))
        features.append(compute_feature(im))

    return array(features),labels

def convert_labels(labels,transl):
    """ Convert between strings and numbers. """
    return [transl[l] for l in labels]

def find_sudoku_edges(im, axis=0):
    """ 寻找对齐后数独图像的的单元边线 """
    # threshold and sum rows and columns
    #阈值化，像素值小于128的阈值处理后为1，大于128的为0
    trim = 1*(128 > im)
    #阈值处理后对行（列）相加求和
    s = trim.sum(axis=axis)
    print s
    # find center of strongest lines
    # 寻找连通区域
    s_labels, s_nbr = measurements.label((0.5*max(s)) < s)
    print s_labels
    print s_nbr
    #计算各连通域的质心
    m = measurements.center_of_mass(s, s_labels, range(1, s_nbr+1))
    print m
    #对质心取整，质心即为粗线条所在位置
    x = [int(x[0]) for x in m]
    print x
	# if only the strong lines are detected add lines in between
    # 如果检测到了粗线条，便在粗线条间添加直线
    if 4 == len(x):
        dx = diff(x)
        x = [x[0], x[0]+dx[0]/3, x[0]+2*dx[0]/3, x[1], x[1]+dx[1]/3, x[1]+2*dx[1]/3, x[2], x[2]+dx[2]/3, x[2]+2*dx[2]/3, x[3]]
    if 10 == len(x):
        return x
    else:
        raise RuntimeError('Edges not detected.')

if __name__ == "__main__":
    # TRAINING DATA
    features, labels = load_ocr_data('../data/sudoku_images/ocr_data/training/')
    # TESTING DATA
    test_features, test_labels = load_ocr_data('../data/sudoku_images/ocr_data/testing/')
    # train a linear SVM classifier
    features = map(list, features)
    test_features = map(list, test_features)

    classnames = unique(labels)

    # create conversion function for the labels
    transl = {}
    for i,c in enumerate(classnames):
        transl[c],transl[i] = i,c


    prob = svm_problem(convert_labels(labels,transl), features)
    param = svm_parameter('-t 0')
    m = svm_train(prob, param)
    # how did the training do?
    res = svm_predict(convert_labels(labels,transl), features, m)
    # how does it perform on the test set?
    res = svm_predict(convert_labels(test_labels,transl), test_features, m)

    # 加载sudoku18.jpg
    imname = '../data/sudoku_images/sudokus/sudoku18.jpg'
    vername = '../data/sudoku_images/sudokus/sudoku18.sud'
    im = array(Image.open(imname).convert('L'))

    # find the cell edges
    # 寻找x方向的单元边线
    x = find_sudoku_edges(im, axis=0)
    # 寻找y方向的单元边线
    y = find_sudoku_edges(im, axis=1)

    # crop cells and classify
    crops = []
    for col in range(9):
        for row in range(9):
            crop = im[y[col]:y[col + 1], x[row]:x[row + 1]]
            crops.append(compute_feature(crop))

    res = svm_predict(loadtxt(vername), map(list, crops), m)[0]
    res_im = array(res).reshape(9, 9)
    print 'Result:'
    print res_im