import numpy as np
import cv2

def gradient_vector(img):
    h,w = img.shape
    x = np.zeros((h-2, w-2))
    y = np.zeros((h-2, w-2))
    # compute row
    x += img[1:-1, -(w-2):] 
    x -= img[1:-1, :w-2]
    # compute column
    y += img[:h-2, 1:-1]
    y -= img[-(h-2):, 1:-1]
    magnitude = np.sqrt( np.square(x) + np.square(y) )
    orientation = np.arctan2( y, x+1e-12 ) * (180/np.pi)
    #normalize
    orientation += 180
    orientation %= 180
    
    return magnitude, orientation


def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def calc_hist(mag, ori, bins):
    hist = np.zeros(bins)
    y,x = ori.shape
    bin_val = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    for i in range(y):
        for j in range(x):
            if 0 <= ori[i,j] < 20:
                hist[1] += mag[i,j] * (ori[i,j] - bin_val[0]) / 20
                hist[0] += mag[i,j] * (bin_val[1] - ori[i,j]) / 20
            elif 20 <= ori[i,j] < 40:
                hist[2] += mag[i,j] * (ori[i,j] - bin_val[1]) / 20
                hist[1] += mag[i,j] * (bin_val[2] - ori[i,j]) / 20
            elif 40 <= ori[i,j] < 60:
                hist[3] += mag[i,j] * (ori[i,j] - bin_val[2]) / 20
                hist[2] += mag[i,j] * (bin_val[3] - ori[i,j]) / 20
            elif 60 <= ori[i,j] < 80:
                hist[4] += mag[i,j] * (ori[i,j] - bin_val[3]) / 20
                hist[3] += mag[i,j] * (bin_val[4] - ori[i,j]) / 20
            elif 80 <= ori[i,j] < 100:
                hist[5] += mag[i,j] * (ori[i,j] - bin_val[4]) / 20
                hist[4] += mag[i,j] * (bin_val[5] - ori[i,j]) / 20
            elif 100 <= ori[i,j] < 120:
                hist[6] += mag[i,j] * (ori[i,j] - bin_val[5]) / 20
                hist[5] += mag[i,j] * (bin_val[6] - ori[i,j]) / 20
            elif 120 <= ori[i,j] < 140:
                hist[7] += mag[i,j] * (ori[i,j] - bin_val[6]) / 20
                hist[6] += mag[i,j] * (bin_val[7] - ori[i,j]) / 20
            elif 140 <= ori[i,j] < 160:
                hist[8] += mag[i,j] * (ori[i,j] - bin_val[7]) / 20
                hist[7] += mag[i,j] * (bin_val[8] - ori[i,j]) / 20
            elif 160 <= ori[i,j] < 180:
                hist[0] += mag[i,j] * (ori[i,j] - bin_val[8]) / 20
                hist[8] += mag[i,j] * (180 - ori[i,j]) / 20
    
    return hist.tolist()
    
    
def get_HOG(image, block_size=2, cell_size=8):
    # 1. gamma normalization
    #image = adjust_gamma(image, gamma=0.5)
    
    # 2. gradient computation
    # 3. spatial/orientation binning
    grad_mag, grad_ori = gradient_vector(image)
    
    # 4. calculate cell_histogram
    sy, sx = image.shape
    ncell_y, ncell_x = sy//cell_size, sx//cell_size
    bins = 9
    hists = np.zeros((ncell_y, ncell_x, bins))
    for i in range(ncell_y):
        for j in range(ncell_x):
            tmp_mag = grad_mag[i*cell_size: (i+1)*cell_size, j*cell_size: (j+1)*cell_size]
            tmp_ori = grad_ori[i*cell_size: (i+1)*cell_size, j*cell_size: (j+1)*cell_size]
            cell_hist = calc_hist(tmp_mag, tmp_ori, bins)
            hists[i,j,:] = cell_hist
    
    # 5. normalization blocks
    # 6. concatenate hists => HOG feature
    hog_feature = []
    nblock_y, nblock_x = (ncell_y - 1, ncell_x - 1)
    for i in range(nblock_y):
        for j in range(nblock_x):
            tmp_blocks = np.copy(hists[i: i+block_size, j: j+block_size])
            hog_feature += ( tmp_blocks / (np.sqrt(np.sum(tmp_blocks))+1e-12) ).ravel().tolist()
 
    return hog_feature
    

def extract_feature(data, block_size = 2, cell_size = 8, verbose = False):
    num = data.shape[0]
    X_feats = []
    count = 0
    for i in range(num):
        X_feats.append(get_HOG(data[i]))
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