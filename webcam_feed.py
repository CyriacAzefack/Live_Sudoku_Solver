# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:49:04 2018

@author: cyriac.azefack
"""

"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
from Data_Preprocessing import process_image
from Sudoku import *
import matplotlib.pyplot as plt
import numpy as np
from Sudoku_Solver import digit_recognitions, solve


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img_contour = process_image(img.copy(), live=True)
        cv2.imshow('my webcam', img_contour)
        if cv2.waitKey(1) == 13: # Enter to validate
            print('Image Validated!!')
            plt.imshow(img)
            plt.show()
            _, _, sudoku_features = process_image(img, training=False)

            sudoku = np.zeros((9, 9))
            labels = digit_recognitions(sudoku_features)

            # sudoku[j][i] = int(label)

            for i in range(9):
                for j in range(9):
                    sudoku[i][j] = int(labels[j * 9 + i])

            print('Sudoku Detected')

            print(sudoku)

            # solve(sudoku)
            break
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()



def main():
    show_webcam(mirror=False)


if __name__ == '__main__':
    main()