import cv2 as cv
import os
import numpy as np
import string


def Recognition(TestImage, m, A, Eigenfaces):
    ProjectedImages = []
    Train_Number = Eigenfaces.shape[1]
    for i in range(0, Train_Number):
        temp =  np.dot(Eigenfaces.T, A[:, i])
        ProjectedImages.append(temp)
    ProjectedImages = np.array(ProjectedImages)

    ProjectedImages = np.transpose(ProjectedImages)#重要


    InputImage = cv.imread(TestImage)

    # temp = InputImage[:, :, 0]

    temp = cv.cvtColor(InputImage, 10)


    InImage = temp.flatten()
    Difference = InImage

    # print(m.shape)
    Difference = Difference - m
    # print(Difference)
    # print(Difference)
    ProjectedTestImage = np.dot(Eigenfaces.T,  Difference)
    print(ProjectedTestImage)

    # Euc_dist = np.empty(shape=(0, 20))
    Euc_dist = []
    for i in range(0,  Train_Number):
    # for i in np.arange(0, Train_Number):
        q = ProjectedImages[: ,i]
        temp = (cv.norm(ProjectedTestImage - q)) ** 2
        Euc_dist.append(temp)


    Euc_dist = np.array(Euc_dist)
    print(Euc_dist)
    Euc_dist = np.transpose(Euc_dist)#重要

    Recognized_index = np.argmin(Euc_dist)


    OutputName = str(Recognized_index + 1) + '.jpg'
    return OutputName