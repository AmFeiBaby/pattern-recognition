from tkinter import simpledialog
import tkinter
import cv2 as cv
import CreateDatabase
import EigenfaceCore
import Recognition

tk = tkinter.Tk()
tk.withdraw()
num = simpledialog.askinteger('Input of PCA-Based Face Recognition System', prompt='Enter test image name (a number between 1 to 10):', initialvalue=1, minvalue=1, maxvalue=10)
TestImage = num
TestImage = 'TestDatabase' + '/' + str(TestImage) + '.jpg'

im = cv.imread(TestImage)
T = CreateDatabase.CreateDatabase('TrainDatabase')
m, A, Eigenfaces = EigenfaceCore.EigenfaceCore(T)


OutputName  = Recognition.Recognition(TestImage, m, A, Eigenfaces)
SelectedImage = 'TrainDatabase' +  '/' + OutputName
SelectedImage = cv.imread(SelectedImage)
cv.imshow('Test Image', im)
cv.imshow('Equivalent Image', SelectedImage)
cv.waitKey(0)
cv.destroyAllWindows()
str = 'Matched image is :' + OutputName
print(str)