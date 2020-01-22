import cv2 as cv
import random
import numpy as np
from matplotlib import pyplot as plt
img0=cv.imread('woody.jpg')
rows,cols,channel=img0.shape
img=img0[0:rows-40,10:cols-20]

def change_color(b_rand,g_rand,r_rand,img):
    # brightness
    B, G, R = cv.split(img)


    if b_rand == 0 or b_rand<-50 or b_rand>50 :
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)


    if g_rand == 0 or g_rand<-50 or g_rand>50:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)


    if r_rand == 0 or r_rand<-50 or r_rand>50 :
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img = cv.merge((B, G, R))
    return img
#input myself
img = change_color(25,-28,36,img)
#rotation
M=cv.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),15,1.1)
img=cv.warpAffine(img,M,(img.shape[1],img.shape[0]))
#perspective
pts1=np.float32([[0,0],[cols-1,0],[0,rows-1]])
pts2=np.float32([[cols*0,rows*0],[cols*0.9,rows*0.2],[cols*0.2,rows*0.9]])
M=cv.getAffineTransform(pts1,pts2)
img=cv.warpAffine(img,M,(cols,rows))


cv.imshow('Woody Allen',img)
cv.imwrite('woody_change.jpg',img)
key=cv.waitKey()
if key==27:
    cv.destroyAllWindows()
