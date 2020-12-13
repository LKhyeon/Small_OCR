import numpy as np
import cv2
import ImageProcessing

def getFeatures(img):
    """
    Features that will be used for the training was originally based on features
    from the following paper:
    ​https://pdfs.semanticscholar.org/dd13/1b9105c16e16a8711faa80f232de01df02c1.pdf

    *** Explanation of the code ***
    There are 32 features.
    First set of features are percentage of non-zero pixel values in each quadrants.
    Second set of features are number of corners in each quadrants.
    Third set of features are percentage of area portion that is encapsulated by convex hull takes.
    Additional set of features that are combinations of features from each quadrants are also included to increase the accuracy.
    a. features from quadrant 1 + eatures from quadrant 2
    b. features from quadrant 2 + eatures from quadrant 3
    c. features from quadrant 3 + eatures from quadrant 4
    d. features from quadrant 1 + eatures from quadrant 4
    e. features from quadrant 1 + eatures from quadrant 3
    f. features from quadrant 2 + eatures from quadrant 4
    """
    dim = img.shape
    row = dim[0]
    col = dim[1]
    area = row * col
    first_quad = img[:int(row/2),int(col/2):]
    second_quad = img[:int(row/2),:int(col/2)]
    third_quad = img[int(row/2):,:int(col/2)]
    fourth_quad = img[int(row/2):,int(col/2):]
    f_collec = []
    # First set of data:
    f_collec.append(float(np.sum(np.sum(first_quad, axis=0)))/(255*area)*100)
    f_collec.append(float(np.sum(np.sum(second_quad, axis=0)))/(255*area)*100)
    f_collec.append(float(np.sum(np.sum(third_quad, axis=0)))/(255*area)*100)
    f_collec.append(float(np.sum(np.sum(fourth_quad, axis=0)))/(255*area)*100)  
    # Second set of data:
    f_collec.append(float(getCornerCount(first_quad)))
    f_collec.append(float(getCornerCount(second_quad)))
    f_collec.append(float(getCornerCount(third_quad)))
    f_collec.append(float(getCornerCount(fourth_quad)))    
    # Third set of data:
    f_collec.append(float(getConvexArea(first_quad)/area)*100)
    f_collec.append(float(getConvexArea(second_quad)/area)*100)
    f_collec.append(float(getConvexArea(third_quad)/area)*100)
    f_collec.append(float(getConvexArea(fourth_quad)/area)*100)
    f_collec.append(float(getCornerCount(img)))
    f_collec.append(float(getConvexArea(img)/area)*100)     
    # Additional features that are created by combination of each kinds of features:
    f_collec.append((f_collec[0] + f_collec[1]))
    f_collec.append((f_collec[1] + f_collec[2]))
    f_collec.append((f_collec[2] + f_collec[3]))
    f_collec.append((f_collec[3] + f_collec[0]))
    f_collec.append((f_collec[1] + f_collec[3]))
    f_collec.append((f_collec[0] + f_collec[2]))  
    f_collec.append((f_collec[4] + f_collec[5]))
    f_collec.append((f_collec[5] + f_collec[6]))
    f_collec.append((f_collec[6] + f_collec[7]))
    f_collec.append((f_collec[7] + f_collec[4]))
    f_collec.append((f_collec[5] + f_collec[7]))
    f_collec.append((f_collec[4] + f_collec[6]))  
    f_collec.append((f_collec[8] + f_collec[9]))
    f_collec.append((f_collec[9] + f_collec[10]))
    f_collec.append((f_collec[10] + f_collec[11]))
    f_collec.append((f_collec[11] + f_collec[8]))
    f_collec.append((f_collec[9] + f_collec[11]))
    f_collec.append((f_collec[8] + f_collec[10]))      
    return f_collec

def getFeaturesInStr(img, letter_int):
    """
    Does same thing as getFeatures() but it returns string representation of it
    along with type of character (in integer) at the end. 
    The purpose of this function is to provide a string that can be written in 
    .csv file that will serve as training database.
    """
    f_collec = getFeatures(img)
    f_collec_str = ""
    for f in f_collec:
        f_collec_str += str(f) + ","
    f_collec_str += letter_int + "\n"
    return f_collec_str

def getCornerCount(img):
    corner_count = 0
    img_f = np.float32(img)
    dst = cv2.cornerHarris(img_f, 2, 3, 0.04)
    dst_norm = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    dim = dst_norm.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if (dst_norm[i, j] > 200):
                corner_count += 1
    return corner_count

def getConvexArea(img):
    if not np.count_nonzero(img):
        return 0
    non_zero_points = cv2.findNonZero(img)
    hull = cv2.convexHull(non_zero_points)
    return cv2.contourArea(hull)
