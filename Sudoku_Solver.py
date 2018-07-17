# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 15:06:29 2018

@author: cyriac.azefack
"""

from sklearn.externals import joblib
from Data_Preprocessing import process_image_file
from tensorflow.contrib import predictor
import numpy as np
from datetime import datetime
from Sudoku import *

def main():
    filepath = "d:/cyriac.azefack/Documents/Sudoku_Image_Solver/Live_Sudoku_Solver/Image_Test/sudoku_test.JPG"
    # filepath = "d:/cyriac.azefack/Documents/Sudoku_Image_Solver/images/image58.jpg"

    # for i in range(10):
    now = datetime.now()
    sudoku = image2sudoku(filepath)
    print('Time for digit recognition : %ss.' % (datetime.now() - now).total_seconds())
    now = datetime.now()
    solve(sudoku)
    print('Time for Solving Sudoku : %ss.' % (datetime.now() - now).total_seconds())

def image2sudoku(filepath):
    _, _, sudoku_feature = process_image_file(filepath=filepath, training=False)

    sudoku = np.zeros((9, 9))
    labels = digit_recognitions(sudoku_feature)
    # sudoku[j][i] = int(label)

    for i in range(9):
        for j in range(9):
            sudoku[i][j] = int(labels[j*9 + i])

    print('Sudoku Detected')

    print(sudoku)

    return sudoku
    

def digit_recognitions(features, model='CNN'):
    """
    Predict the Number lbel
    :param feature:
    :return: Sudoku array
    """

    if model == 'RandomForest':
        clf = joblib.load('RandomForest_Classifier.pkl')
        label = clf.predict(features)
        return label
    elif model == 'CNN':
        folder_path = './classifier_model/1531763776'

        predict_fn = predictor.from_saved_model(folder_path)
        predictions = predict_fn({
            "x": features
        })

        return predictions['classes']

def solve(sudoku):

    sudoku = Sudoku(sudoku=sudoku)
    solved = sudoku.solve()
    if solved:
        sudoku_solution = sudoku.solution
        divider = ' '.join('-' for i in range(len(sudoku_solution)))
        print('The sudoku has been solved:')
        print(divider)
        # print(Sudoku.format(sudoku_solution))
        # noinspection PyUnusedLocal
        print(divider)
        solution = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                solution[i][j] = int(sudoku_solution[i][j][0])
        print(solution)
        return solution


    else:
        print('Failed to solve !!!')
        return None

if __name__ == '__main__':
    main()