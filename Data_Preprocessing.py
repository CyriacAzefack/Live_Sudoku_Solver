# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:26:37 2018

@author: cyriac.azefack
"""

import cv2
import numpy as np
import time as t
import math
import glob
import pickle
import sys, os
import matplotlib.pyplot as plt
print ("OpenCV Version : %s " % cv2.__version__)


NUMBER_IMG_SIZE = 50 #choosed randomly
SUDOKU_IMG_SIZE = NUMBER_IMG_SIZE*9
NB_MIN_ACTIVE_PIXELS = 100

                
def extract_number(x, y, sudoku_image):
    '''
    Extract the number image, thesholded image and number of active pixels
    '''
    #Coordinate in the sudoku square like (4, 5)
    
    #Crop the image to get the number
    img_number = sudoku_image[y*NUMBER_IMG_SIZE:(y+1)*NUMBER_IMG_SIZE][:, x*NUMBER_IMG_SIZE:(x+1)*NUMBER_IMG_SIZE]
    
    #threshold
    img_number_thresh = cv2.adaptiveThreshold(img_number, 255, 1, 1, 3, 5)
    #img_number_thresh = img_number.copy()
    #Delete active pixels in a radius (from center)
    #We assume that the number is at the center of the square box, so we delete pixels too far from the center
    for i in range(img_number.shape[0]):
        for j in range(img_number.shape[1]):
            dist_center = math.sqrt( (NUMBER_IMG_SIZE/2 - i)**2  + (NUMBER_IMG_SIZE/2 - j)**2);
            if dist_center > 20:
                img_number_thresh[i,j] = 0
                
    n_active_pixels = cv2.countNonZero(img_number_thresh)
    
    return img_number, img_number_thresh, n_active_pixels


def find_biggest_bounding_box(img_number_thresh):
    '''
    Find the biggest box in the image (should include the number or nothing)
    '''
    _, contours, hierarchy = cv2.findContours(img_number_thresh.copy(),
                                         cv2.RETR_CCOMP,
                                         cv2.CHAIN_APPROX_SIMPLE)

    biggest_bound_rect = [];
    bound_rect_max_size = 0;
    for contour in contours:
        bound_rect = cv2.boundingRect(contour)
        size_bound_rect = bound_rect[2]*bound_rect[3]
        if  size_bound_rect  > bound_rect_max_size:
                 bound_rect_max_size = size_bound_rect
                 biggest_bound_rect = bound_rect
    #bounding box a little more bigger
    x_b, y_b, w, h = biggest_bound_rect;
    x_b = x_b - 1
    y_b = y_b - 1
    w = w + 2;
    h = h + 2; 
                
    return [x_b, y_b, w, h]

def recognize_number(x, y, sudoku_image):
    """
    Recognize the number in the rectagle
    """

    [img_number, img_number_thresh, nb_active_pixels] = extract_number(x, y, sudoku_image)

    img_number = cv2.resize(img_number, (NUMBER_IMG_SIZE, NUMBER_IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    if nb_active_pixels > NB_MIN_ACTIVE_PIXELS:
        [x_b, y_b, w, h] = find_biggest_bounding_box(img_number_thresh)
        number = img_number[y_b:y_b+h, x_b:x_b+w]

        if number.shape[0] * number.shape[1] > 0:
            ret, img_number = cv2.threshold(img_number, 127, 255, 0)

    img_number = img_number.reshape(NUMBER_IMG_SIZE * NUMBER_IMG_SIZE)

    # plt.imshow(img_number.reshape((50, 50)))
    # plt.show()

    return img_number

def process_image_file(filepath, training = True)  :
    img = cv2.imread(filepath)
    process_image(img, filepath = filepath, training=training)

def process_image(img, filepath=None, training = True, live = False) :
    '''
    Process an image and extract features and labels
    '''
    
    features = []
    labels = []
    
    #Convert it to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    #Apply Adaptative tresholding
    thresh = cv2.adaptiveThreshold(gray, 255, 0, 1, 17, 2)
    
    #Find the largest BOX
    #Start by finding contours in the tresholded image
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest = None
    max_area = 0
    for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02*peri, True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
    
    # We have the 4 corners of the puzzle in the image, now let's identify the 
    #   TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT and BOTTOM-LEFT

    if biggest is None and live :
        return img

    biggest = biggest.reshape((4, 2))

    if live:
        img_contours = cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
        return img_contours

    corners = np.zeros((4, 2), dtype=np.float32)
    
    add = biggest.sum(1)
    corners[0] = biggest[np.argmin(add)] #TOP-LEFT
    corners[2] = biggest[np.argmax(add)] #BOTTOM-RIGHT
    
    diff = np.diff(biggest, axis = 1)
    corners[1] = biggest[np.argmin(diff)] #TOP-RIGHT
    corners[3] = biggest[np.argmax(diff)] #BOTTOM-LEFT
    
    #Fixing the sudoku square distortion
    '''
    Now we have the sudoku puzzle segmented. We have got the corner points of 
    the puzzle. It's currently not really usable for much. The sudoku puzzle is
    a bit distorted. It's necessary to correct the skewed perspective of the image.
    We need a way to mapping from the puzzle in the original picture back into a 
    square. Where each corner of the sudoku puzzle corresponds to a corner on the 
    a new image.
    '''
    #Reshape the image

    retval = cv2.getPerspectiveTransform(corners, np.array([ [0,0],[SUDOKU_IMG_SIZE-1, 0],[SUDOKU_IMG_SIZE-1, SUDOKU_IMG_SIZE-1],
                                                            [0, SUDOKU_IMG_SIZE-1] ],np.float32))
    warp = cv2.warpPerspective(img, retval,(SUDOKU_IMG_SIZE, SUDOKU_IMG_SIZE))
    warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)



    #Segmenting the numbers
    
    '''
    Sometimes after remove the noise of the images can appear other particles 
    close to the number. Because of this, we find the biggest bounding box in 
    the small squared. We make the bounding box a little more bigger to improve
    the matching with the datase.
    '''
    sudoku_feat = np.zeros(shape=(9*9, NUMBER_IMG_SIZE*NUMBER_IMG_SIZE))
    
    if training :
        sudoku_labels = np.loadtxt(filepath[:-4]+'.dat')
        if len(sudoku_labels) == 81:
            sudoku_labels = sudoku_labels.reshape( (9, 9) )
            
            
        for i in range(9):
            for j in range(9):
                if int(sudoku_labels[i][j]) != 0:
                    feat = recognize_number(i, j, warp_gray)
                    if feat is not None:
                        sudoku_feat[i*9 + j] = feat
                        features.append(feat)
                        label = int(sudoku_labels[j][i])
                        
                        labels.append(label)
                        
                        filename = './numbers_images/' + str(label) + '_' + os.path.basename(filepath)
                        cv2.imwrite(filename, feat.reshape((NUMBER_IMG_SIZE, NUMBER_IMG_SIZE)))
    else :
        
        labels = None
        for i in range(9):
            for j in range(9):
                    feat = recognize_number(i, j, warp_gray)
                    if feat is not None:
                        sudoku_feat[i*9 + j] = feat

    return features, labels, sudoku_feat, img_contours
                    

if __name__ == "__main__" :
    all_features = []
    all_labels = []
    
    t_start = t.process_time()
    all_files = glob.glob('../images/*.jpg')
    for filename in all_files:
        
        i = all_files.index(filename) + 1
        sys.stdout.write("\r%.2f %% of images treated!!" % (100 * i / len(all_files)))
        sys.stdout.flush()
        features, labels, _ = process_image_file(filename)
        
        all_features = all_features + features
        all_labels = all_labels  + labels
    sys.stdout.write("\n")
    elapsed_time = t.process_time() - t_start
    
    print("{:0.2f}s to process all the images".format(elapsed_time))

    all_features = np.asarray(all_features)
    all_labels = np.asarray(all_labels)
    
    data_dict = {'features' : all_features,
            'labels' : all_labels}
    with open('data.pickle', 'wb') as file:
        pickle.dump(data_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed_time = t.process_time() - elapsed_time
        
    print("{:0.2f}s to save all the features and the labels".format(elapsed_time))



