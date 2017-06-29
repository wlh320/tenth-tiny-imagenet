# -*- coding:utf-8 -*-
import numpy as np
from skimage import feature as skif


def color_feature(img, nbin=16, xmin=0, xmax=255, normalized=True):

    """ HSV 颜色直方图特征 """
    if img.mode == "L":
        bins = np.linspace(xmin, xmax, nbin+1)
        imhist, bin_edges = np.histogram(img, bins=256, density=normalized)
        imhist = imhist * np.diff(bin_edges)
        
        return np.concatenate((imhist,imhist[0:16],imhist[0:16]))
        
    hsv = np.array(img.convert("HSV"))
    ndim = hsv.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    imhist, bin_edges = np.histogram(hsv[:,:,0], bins=256, density=normalized)
    imhist = imhist * np.diff(bin_edges)
    imhist_s, bin_edges_s = np.histogram(hsv[:,:,1], bins=16, density=normalized)
    imhist_s = imhist_s * np.diff(bin_edges_s)
    imhist_v, bin_edges_v = np.histogram(hsv[:,:,2], bins=16, density=normalized)
    imhist_v = imhist_v * np.diff(bin_edges_v)

    return np.concatenate((imhist,imhist_s,imhist_v))


# def hu_moments(I):
#     """hu不变矩"""
#     X = np.arange(I.shape[0]).reshape((1,I.shape[0]))
#     Y = np.arange(I.shape[0]).reshape((I.shape[1],1))
    
#     m00 = np.sum(I);    
#     X_m = np.sum(np.dot(X,I))/m00
#     Y_m = np.sum(np.dot(I,Y))/m00
#     X = X - X_m
#     Y = Y - Y_m

#     n11 = np.dot(X,np.dot(I,Y))    / np.power(m00,2.0)
#     n20 = np.sum(np.dot(X**2,I))   / np.power(m00,2.0)
#     n02 = np.sum(np.dot(I,Y**2))   / np.power(m00,2.0)
#     n30 = np.sum(np.dot(X**3,I))   / np.power(m00,2.5)
#     n03 = np.sum(np.dot(I,Y**3))   / np.power(m00,2.5)
#     n21 = np.dot(X**2,np.dot(I,Y)) / np.power(m00,2.5)
#     n12 = np.dot(X,np.dot(I,Y**2)) / np.power(m00,2.5)

#     N = np.zeros(7)
#     N[0] = n20 + n02
#     N[1] = (n20-n02)**2 + 4.0*n11**2
#     N[2] = (n30-3.0*n12)**2 + (n03-3.0*n21)**2
#     N[3] = (n30+n12)**2 + (n03+n21)**2
#     N[4] = (n30-3.0*n12)*(n30+n12)*((n30+n12)**2 -3.0*(n21+n03)**2)-(3.0*n21-n03)*(n21+n03)*(3.0*(n30+n12)**2-(n21+n03)**2)
#     N[5] = (n20-n02)*((n30+n12)**2-(n21+n03)**2)+4.0*n11*(n30+n12)*(n03+n21)
#     N[6] = (3.0*n21-n03)*(n30+n12)*((n30+n12)**2 -3.0*(n21+n03)**2) - (n30-3.0*n12)*(n21+n03)*(3.0*(n30+n12)**2 -(n21+n03)**2)
    
#     return N



def shape_feature(img):
    """
    HOG特征算子作为形状特征
    尝试过Hu不变矩发现效果并不好,遂放弃
    """
    # return hu_moments(np.array(img))
    hog = skif.hog(np.array(img), block_norm='L2-Hys')
    return hog
    
