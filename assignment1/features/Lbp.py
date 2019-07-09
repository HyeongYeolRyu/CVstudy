import numpy as np
import cv2
import os
import pprint


def get_LBP_fast(image):
        X = image
        X = (1<<7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
            +(1<<6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
            +(1<<5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
            +(1<<4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
            +(1<<3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
            +(1<<2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
            +(1<<1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
            +(1<<0) * (X[1:-1, :-2] >= X[1:-1, 1:-1]) 

        img = np.zeros_like(image)
        img[1:-1, 1:-1] = X    
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        return hist


def split_into_blocks(data, block_size):
    height, width = data.shape
    blocks = []
    # each block's width, height
    block_w, block_h = int(np.ceil(width / block_size)), int(np.ceil(height / block_size))
    for i in range(block_size):
        for j in range(block_size):
            block = data[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            blocks.append(np.array(block))
    
    return blocks


def get_LBP(image):
    height, width = image.shape
    lbp = []
    for h in range(1, height-1):
        for w in range(1, width-1):
            curr = image[h,w]
            val = 0
            val |= 1 << 7 if curr <= image[h, w-1] else 0
            val |= 1 << 6 if curr <= image[h+1, w-1] else 0
            val |= 1 << 5 if curr <= image[h+1, w] else 0
            val |= 1 << 4 if curr <= image[h+1, w+1] else 0
            val |= 1 << 3 if curr <= image[h, w+1] else 0
            val |= 1 << 2 if curr <= image[h-1, w+1] else 0
            val |= 1 << 1 if curr <= image[h-1, w] else 0
            val |= 1 << 0 if curr <= image[h-1, w-1] else 0
            lbp.append(val)
            
    lbp = np.array(lbp).astype('uint8').reshape(height-2, width-2)
    lbp = cv2.calcHist([lbp], [0], None, [256], [0,256]) 
    return lbp


def extract_feature_lbp(data, block_size = 7, verbose = False):
    num = data.shape[0]
    X_feats = []
    count = 0
    for i in range(num):
        """
        blocks = split_into_blocks(data[i], block_size = block_size)
        hists = []
        for block in blocks:
            lbp = get_LBP(block)
            hists += lbp
        X_feats.append(cv2.calchist( [hists], [0], None, [256], [0,256] ))
        """
        #X_feats.append(get_LBP(data[i]))
        X_feats.append(get_LBP_fast(data[i]))
        count += 1
        if verbose and count % 10 == 0:
            print(str(count) + " featurizing completed.")
    X_feats = np.array(X_feats)
    if verbose and count % 10 != 0:
            print(str(count) + " featurizing completed.")
    if verbose:
        print("==========================")
        print("All featurizing completed.")
        
    return X_feats
