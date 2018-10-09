# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pytesseract
from PIL import Image

from matplotlib import pyplot as plt

'''
dir = 0, 投影至垂直方向
      1，投影至水平方向
min_th      : 直方图上的最小阈值
min_range   : 直方图上峰值的最小高度
'''
def get_peaks(im_in, dir = 1, min_th =2, min_range = 10):
    hist_v = np.sum(im_in, axis=dir)
    begin = 0
    end = 0
    # min_th = 2
    # min_range = 10
    peak_pair = []
    for i in range(0, hist_v.shape[0]):

        if hist_v[i] > min_th and begin == 0:
            begin = i
        elif hist_v[i] > min_th and begin != 0:
            continue
        elif hist_v[i] < min_th and begin != 0:
            end = i
            if end - begin > min_range:
                peak_pair.append([begin, end])
                begin = 0
                end = 0
        elif hist_v[i] < min_th or begin == 0:
            continue
        else:
            print "error!"


    return len(peak_pair), peak_pair

if __name__ == "__main__":
    img = cv2.imread('../data/12.png')
    h, w, _ = img.shape

    img = cv2.resize(img, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
    h, w, _ = img.shape

    #1, 二值化
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 9, 2)
    mask_inv = cv2.bitwise_not(th3)

    #'''
    # T.1 测试image_to_boxes函数的输出
    boxes = pytesseract.image_to_boxes(mask_inv, lang='eng')  # also include any config options you use
    # draw the bounding boxes on the image
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)

    # show annotated image and wait for keypress
    cv2.imshow('results', img)
    cv2.imwrite('resutls.png', img)
    cv2.waitKey(0)
    # '''

    #2, 水平方向投影
    peak_num, peak_pairs = get_peaks(mask_inv, 1)

    for i in range(0, peak_num):
        h1 = peak_pairs[i][0]
        h2 = peak_pairs[i][1]
        one_line = mask_inv[h1 - 1:h2 + 1, :]
        peak_num_hor, peak_pairs_hor = get_peaks(one_line, 0, 1, 2)

        #3, 垂直方向投影
        for j in range(0, peak_num_hor):
            w1 = peak_pairs_hor[j][0];
            w2 = peak_pairs_hor[j][1];
            one_char = one_line[:,w1 - 1: w2 +1]
            # cv2.imwrite('char/char'+str(i)+'_'+str(j)+'.png', one_char)
        cv2.imwrite('line'+str(i)+'.png', one_line)
        print(pytesseract.image_to_string(one_line, lang='eng'))

    cv2.waitKey(0)
    '''
    plt.plot(np.sum(mask_inv, axis=1))
    plt.show()
    '''

