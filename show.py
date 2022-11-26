import cv2 as cv
import CreateDatabase
import EigenfaceCore
import Recognition


TestImage = 'E:/code/patternRecognition/TestDatabase/4.jpg'
im = cv.imread(TestImage)
T = CreateDatabase.CreateDatabase('TrainDatabase')
[m, A, Eigenfaces] = EigenfaceCore.EigenfaceCore(T)
OutputName = Recognition.Recognition(TestImage, m, A, Eigenfaces)
SelectedImage = 'TrainDatabase' +  '/' + OutputName
SelectedImage = cv.imread(SelectedImage)
cv.imshow('Test Image', im)
cv.imshow('Equivalent Image', SelectedImage)
cv.waitKey(0)
cv.destroyAllWindows()
str = 'Matched image is :' + OutputName
print(str)