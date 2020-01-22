import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pylab
img=cv.imread('dw.jpg',0)
def MedianBlur(img,kernel,padding_way):
    cv.imshow('DoctorWho',img)
    key=cv.waitKey()
    if key ==27:
       cv.destroyAllWindows()
    print(img.shape)
    adds=kernel//2
    rows, cols = img.shape
    if padding_way=='zero':
        a=[]
        b=[[]]
        for _ in range(cols):
           a.append(0)
        a = np.array(a)
        a = a.T
        for _ in range(adds):
            img=np.vstack([img,a])
            img = np.vstack([a,img])
        for _ in range(rows+adds*2):
             b[0].append(0)
        b = np.array(b)
        b=b.T
        for _ in range(adds):
            img=np.hstack([img,b])
            img=np.hstack([b,img])
    if padding_way=='REPLICA':
        a1=img[0]
        a2=img[rows-1]
        for _ in range(adds):
            img=np.vstack([a1,img])
            img = np.vstack([img,a2])
        b1=[img[:,0]]
        b2=[img[:,cols-1]]
        b1 = np.array(b1)
        b1 = b1.T
        b2 = np.array(b2)
        b2 = b2.T
        for _ in range(adds):
            img=np.hstack([b1,img])
            img=np.hstack([img,b2])

    newImage=np.zeros((rows,cols),dtype=np.float)
    for i in range(adds,adds+rows):
        for j in range(adds,adds+cols):
            ram=[]
            for n in range(-adds,adds+1):
                for m in range(-adds,adds+1):
                    ram.append(img[i+n][j+m])
            ram.sort()
            newImage[i-adds][j-adds]=ram[kernel**2//2]
    print(newImage)
    plt.imshow(newImage,cmap ='gray')
    pylab.show()
    key=cv.waitKey()
    if key==27:
        cv.destroyAllWindows()




MedianBlur(img,15,'REPLICA')
