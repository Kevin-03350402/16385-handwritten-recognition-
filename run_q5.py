import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import math

from nn import *
from q5 import *
import cv2


# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))


    bboxes, bw = findLetters(im1)
    
    
    plt.imshow(bw,cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    lines = dict()
    totalh = 0
    count = 0
    # my strategy is clustering
    # start with finding the average heigh 
    for bbox in bboxes:
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        totalh += abs(y2-y1)
        count += 1
    avgh = totalh/count
    for bbox in bboxes:
        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        center_row = (y1+y2)/2
        if len(lines) == 0:
            lines[1] = []
            lines[1].append(center_row)
            lines[1].append(bbox)
        else:
            closest_p = -1
            closest_d = 999999999
            for row in lines.keys():
                row_c = lines[row][0]
                if abs(row_c-center_row) < closest_d:
                    closest_p  = row
                    closest_d = abs(row_c-center_row)
            if closest_d <= avgh:
                lines[closest_p][0] = (lines[closest_p][0]*(len(lines[closest_p])-1)+center_row)/(len(lines[closest_p]))
                lines[closest_p].append(bbox)
            else:
                rs = len(lines)
                lines[rs+1] = []
                lines[rs+1].append(center_row)
                lines[rs+1].append(bbox)
    for k in lines.keys():
        print(lines[k][0])

    
    newbox = []
    for k in lines.keys():
        ar = lines[k][1:]
        # sort in a row based on y
        ar = sorted(ar,key=lambda l:l[3])
        newbox.extend(ar)
        newbox.append('newline')


    input = []
    for bbox in newbox:
        if bbox == 'newline':
            input.append('newline')
            continue

        y1 = bbox[0]
        x1 = bbox[1]
        y2 = bbox[2]
        x2 = bbox[3]
        word = bw[y1:y2,x1:x2]

        height = word.shape[0]
        width = word.shape[1]
        if height > width:
            diff = height-width
            left = diff//2
            right = math.ceil(diff/2)
            word = np.pad(word, ((0,0),(left,right)), mode='constant', constant_values=1)
        if height < width:
            diff = width - height
            up= diff//2
            down = math.ceil(diff/2)
            word = np.pad(word, ((up,down),(0,0)), mode='constant', constant_values=1)
        # You can check my rows by using imshow to plot my word
        edge = int(0.2*(word.shape[0]))
        word = np.pad(word, ((edge,edge),(edge,edge)), mode='constant', constant_values=1)
        word  = np.array(word, dtype='uint8')
        word = cv2.resize(word, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        input.append((np.transpose(word)).flatten())


    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    sentence = ''


    for ele in input:
        if ele == 'newline':

            print(sentence)
            sentence = ''

        else:
            ele = ele.reshape(1,len(ele))
            h1 = forward(ele, params, "layer1")
            probs = forward(h1, params, "output", softmax)

            yhat = np.argmax(probs)   
            sentence += ' '     
            sentence += (letters[yhat])
    print(sentence)


