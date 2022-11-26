import cv2 as cv
import os
import numpy as np
import string
# import scipy.linalg as linalg

def EigenfaceCore(T):

    m = np.mean(T, axis=1)

    Train_Number = np.size(T, 1)
    A = []

    for i in range(0, Train_Number):
        temp = T[:, i] - m
        A.append(temp)
    A = np.array(A)
    A1 = np.transpose(A)
    L = np.dot(A, A1)

    [D, V] = np.linalg.eig(L)
    D = np.diag(D)

    L_eig_vec = np.empty(shape=(0, 20))
    for i in np.arange(V.shape[1]):
        if D[i, i] > 1:
            L_eig_vec = np.vstack((L_eig_vec, V[:, i]))
    L_eig_vec = np.transpose(L_eig_vec)

    Eigenfaces = np.dot(A1, L_eig_vec)

    return [m, A1, Eigenfaces]

    # for i in range(0, V.shape[1]):
   # V = V.T
    # print('D = {}'.format(D))
    # print(V.shape)