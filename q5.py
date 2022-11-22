import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import skimage.color
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import skimage.filters
import skimage.morphology
import skimage.segmentation
from skimage.util import random_noise
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    
    bboxes = []
    bw = None
    denoised = denoise_wavelet(image,multichannel = True)
    grayscaled = rgb2gray(denoised)
    threshold = threshold_otsu(grayscaled)
    bw = closing(grayscaled < threshold, square(5))
    
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 25:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minr, minc, maxr, maxc])

    bw = np.where((bw==0)|(bw==1), bw^1, bw)



    
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    
    return bboxes, bw