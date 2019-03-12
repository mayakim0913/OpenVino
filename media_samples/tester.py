import cv2 as cv
import PIL as Image

f = '~/Desktop/classroom_error.png'
im = cv.imread(f, 10)
cv.imshow('image', im)
cv.waitKey(0)
cv.destroyAllWindows()
