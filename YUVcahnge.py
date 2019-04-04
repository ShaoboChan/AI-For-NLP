import cv2 as cv
import numpy as np
def cb1():
    pass
def cb2():
    pass
img=cv.imread('1.jpg')
cv.imshow('lena',img)
img_yuv0=cv.cvtColor(img,cv.COLOR_BGR2YUV)

img_yuv0[:,:,2] = cv.equalizeHist(img_yuv0[:,:,2])#v channel

img_output = cv.cvtColor(img_yuv0, cv.COLOR_YUV2BGR)
cv.imshow('Histogram equalized_V', img_output)

y, u, v = cv.split(img_yuv0)

cv.imshow('y', y)
cv.imshow('u', u)
cv.imshow('v', v)


dst = np.array(img.shape, np.uint8)

cv.namedWindow('image')
cv.createTrackbar('U', 'image', 0, 255, cb1)
cv.createTrackbar('V', 'image', 0, 255, cb2)

while (1):
    cv.imshow('image', dst)
    k = cv.waitKey(1) & 0xFFFF
    if k == 27:
        break

    u = cv.getTrackbarPos('U', 'image')
    v = cv.getTrackbarPos('V', 'image')

    print
    u, v


    tmp = img + [0, (u - 100) * 100.0, (v - 100) * 100.0]
    tmp2 = np.array(tmp, np.uint8)

    dst = cv.cvtColor(tmp2, cv.COLOR_YUV2BGR)

cv.imshow('YUV0',img_yuv0)

key=cv.waitKey()
