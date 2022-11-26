import cv2 as cv
import os
import numpy as np
import string


def strcmp(left, right):
    length_of_left, length_of_right = len(left), len(right)
    result = 0

    for index in range(length_of_left):
        try:
            if left[index] > right[index]:
                result = 1
            elif left[index] < right[index]:
                result = 0
        except (IndexError,):
            result = 1
            break

    return 0 if length_of_right > length_of_left else result




def CreateDatabase(TrainDatabase = 'TrainDatabase'):


    Train_Number = 0
    TrainFiles = os.listdir('TrainDatabase')



    for i in range(0, len(TrainFiles)):
        if not(strcmp(TrainFiles[i], '.') & strcmp(TrainFiles[i], '..') & strcmp(TrainFiles[i], 'Thumbs.db')):
            Train_Number += 1



    T = []
    for i in range(1, Train_Number):
        str_i = str(i)
        str_i = '\\' + str_i + '.jpg'
        str_path = 'TrainDatabase\\' + str_i

        img = cv.imread(str_path)
        img = cv.cvtColor(img, 10)
        # print(img.shape)
        irow, icol = img.shape
        temp = img.reshape(irow * icol, 1)
        T.append(temp)
    T = np.array(T)
    T = np.squeeze(T)
    T = T.T

    return T

        # cv.imshow('img', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
CreateDatabase()

