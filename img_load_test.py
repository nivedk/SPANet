import cv2
import numpy as np

a = cv2.imread("3.jpg")
s = a.shape
#print(s)
x = int(s[1]/2)
#print(x)
a1 = a[:,:x,:]
a2 = a[:,x:,:]

cv2.imwrite("a1.png", a1)
cv2.imwrite("a2.png", a2)

