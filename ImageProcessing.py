import cv2
import numpy as np

def getBinaryImage(image_path):
    # Binary image is required to accurately extract lines and letters from the image:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresh, img_bi = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # For the purpose of letter extraction, it is important to reverse 1 and 0:
    img_bi = cv2.bitwise_not(img_bi)
    return img_bi

def adjustImageRotation(img, factor=1):
    """
    Adjust the rotation of the image so that lines are as straight as possible:
    
    *** Explanation of the code ***
    cv2.minAreaRect finds rotated rectangle of the minimum area enclosing the input image.
    As an input, it takes inputarray of Point class variables.
    But since img is a numpy array, turn collect non zero values of img and turn it into 
    a collection of Point variables that represents each location of the non-zero value.
    This is done using cv2.findNonZero function.
    """
    dim = img.shape
    non_zero_points = cv2.findNonZero(img)
    est_rect = cv2.minAreaRect(non_zero_points)
    #showRectEstimation(img, dim, est_rect)
    rect_center = est_rect[0]
    rect_angle = est_rect[2]
    if est_rect[1][1] > est_rect[1][0]:
        rect_angle += 90
    rotationMat = cv2.getRotationMatrix2D(rect_center, rect_angle, 1)
    adjusted = cv2.warpAffine(img, rotationMat, (dim[1], dim[0]))
    thresh, img_bi = cv2.threshold(adjusted, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bi

def showRectEstimation(img, dim, est_rect):
    # Debugging tool that visually shows bounding box estimated by cv2.minAreaRect.
    box = cv2.boxPoints(est_rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255,0,0) ,2)
    cv2.imshow("showRectEstimation", cv2.resize(img, (dim[1]/2, dim[0]/2)))
    cv2.waitKey()    
    
def extractLines(img):
    dim = img.shape
    lines = []
    row_ind = 0
    while row_ind < dim[0]:
        next_ind = row_ind
        while (next_ind < dim[0]) and (np.sum(img[next_ind, :]) > 0):
            next_ind += 1
        if next_ind != row_ind:
            lines.append(img[row_ind:next_ind, :])   
            row_ind = next_ind+1
        else:
            row_ind += 1
    return lines

def doubleProcessLine(lines, threshold):
    """ 
    Double process the image of lines so that image of multiple lines are
    divided into multiple images of separate lines.
    This function serves as a solution to a case in which an image is rotated
    to the point where it is unable to extract single line.
    """
    doub_processed_line = []
    for line in lines:
        if line.shape[0] > threshold:
            line_rot = adjustImageRotation(line)
            doub_processed = extractLines(line_rot)
            for clean_line in doub_processed:
                doub_processed_line.append(clean_line)    
        else:
            doub_processed_line.append(line)   
    return doub_processed_line    
    
def extractCharacter(img):
    dim = img.shape
    characters = []
    col_ind = 0
    char_size = getCharacterSize(img)
    while col_ind < dim[1]:
        next_ind = col_ind
        while (next_ind < dim[1]) and (np.sum(img[:, next_ind]) > 0):
            next_ind += 1
        # Check if there exists a space:
        space_ind = col_ind
        while (space_ind < dim[1]) and  (np.sum(img[:, space_ind]) == 0):
            space_ind += 1
        if space_ind - col_ind >= char_size:
            characters.append(img[:, col_ind:space_ind])   
            col_ind = space_ind+1            
        elif next_ind != col_ind:
            characters.append(img[:, col_ind:next_ind])   
            col_ind = next_ind+1
        else:
            col_ind += 1      
    return characters    

def getCharacterSize(img):
    # Get horizontal length of the character block.
    largest_size = 0
    dim = img.shape
    col_ind = 0
    while col_ind < dim[1]:
        next_ind = col_ind
        while (next_ind < dim[1]) and (np.sum(img[:, next_ind]) > 0):
            next_ind += 1
        if next_ind != col_ind:
            if next_ind - col_ind > largest_size:
                largest_size = next_ind - col_ind
            col_ind = next_ind+1
        else:
            col_ind += 1    
    return largest_size

def processImage(image_path):
    img_bi = getBinaryImage(image_path)
    lines = extractLines(img_bi)
    row_total = 0
    for line in lines:
        row_total += line.shape[0]
    row_avg = float(row_total/len(lines))
    doub_processed = doubleProcessLine(lines, 0)     
    characters = []
    for i in range(len(doub_processed)):
        characters.append(extractCharacter(doub_processed[i]))
    return characters
