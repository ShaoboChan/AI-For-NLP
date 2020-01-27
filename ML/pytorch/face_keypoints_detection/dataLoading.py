import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# # #############画人脸框和关键点##################
# file=open('label1.txt','r')
# list_name=[]
# lines=file.readline()
# while lines:
#     str_list=lines.split(' ')
#     str_list[0]='Image1'+'/'+ str_list[0]
#     list_name.append(str_list)
#     lines=file.readline()
# # for i in range(len(list_name)):
# #     print(list_name[i])
# # print(len(list_name))
# # print(list_name[0])
# for i in list_name:
#     path=i[0]
#     img=cv.imread(path)
#     x1,y1,x2,y2=round(float(i[1])),round(float(i[2])),round(float(i[3])),round(float(i[4]))
#     width=x2-x1
#     length=y2-y1
#     # print(x1)
#     # plt.figure(8)
#     # plt.imshow(img)
#     # currentAxis = plt.gca()
#     # rect = patches.Rectangle((x1, y1), width, length, linewidth=1, edgecolor='r', facecolor='none')
#     # currentAxis.add_patch(rect)
#     cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4)
#     point_list=[]
#     j=5
#     while j<len(i) :
#         x11=round(float(i[j]))
#         y11=round(float(i[j+1]))
#         # point_list.append('('+str(x11)+','+str(y11)+')')
#         cv.circle(img, (x11,y11), 1, (0, 0, 255), 4)
#         j=j+2
#
#     # for point in point_list:
#     #     print(point)
#         #cv.circle(img, point, 1, (0,0,255), 4)
#     cv.imshow('img',img)
#     key = cv.waitKey(0)
#     if key == 27:
#         cv.destroyAllWindows()
#
#
# # ###########crop########
# #
# # ########################################################
# file=open('label1.txt','r')
# list_name=[]
# lines=file.readline()
# while lines:
#     str_list=lines.split(' ')
#     str_list[0]='Image1'+'/'+ str_list[0]
#     list_name.append(str_list)
#     lines=file.readline()
# # for i in range(len(list_name)):
# #     print(list_name[i])
# # print(len(list_name))
# # print(list_name[0])
# for i in list_name:
#     path=i[0]
#     img=cv.imread(path)
#     # cv.imshow('image', img)
#     x1,y1,x2,y2=round(float(i[1])),round(float(i[2])),round(float(i[3])),round(float(i[4]))
#     width=x2-x1
#     length=y2-y1
#     img_cut=img[y1:y2,x1:x2]
#     base = os.path.basename(path)
#     jpgname=os.path.splitext(base)[0]
#     print(jpgname)
#     newpath='Image2_new/'+jpgname+'.jpg'
#     cv.imwrite(newpath,img_cut)
# # ################################################
# #
# # ############修改crop后的label.txt##############
# # file=open('label1.txt','r')
# # list_name=[]
# # lines=file.readline()
# # while lines:
# #     str_list=lines.split(' ')
# #     str_list[0]=str_list[0]
# #     list_name.append(str_list)
# #     lines=file.readline()
# # with open('Image2_new/label.txt','w') as file_crop:
# #
# #     for i in list_name:
# #         temp = []
# #         j = 5
# #         for idx in i[0:5]:
# #             temp.append(idx)
# #         while j<len(i):
# #             x1, y1 = float(i[1]), float(i[2])
# #             x11 = float(i[j])
# #             y11=float(i[j+1])
# #             xt=x11-x1
# #             yt=y11-y1
# #             temp.append(xt)
# #             temp.append(yt)
# #             j=j+2
# #         for k in range(len(temp)-1):
# #             file_crop.write(str(temp[k])+' ')
# #
# #         file_crop.write(str(temp[-1]))
# #         file_crop.write('\n')
# #
# # ########################################################
# #
# #
# # ##############test######################################
# # #############画人脸框和关键点##################
# # file=open('Image2_new/label.txt','r')
# # list_name=[]
# # lines=file.readline()
# # while lines:
# #     str_list=lines.split(' ')
# #     str_list[0]='Image2_new'+'/'+ str_list[0]
# #     list_name.append(str_list)
# #     lines=file.readline()
# # # for i in range(len(list_name)):
# # #     print(list_name[i])
# # # print(len(list_name))
# # # print(list_name[0])
# # print(len(list_name))
# # for i in list_name:
# #     path=i[0]
# #     img=cv.imread(path)
# #     cv.imshow('image', img)
# #     # x1,y1,x2,y2=round(float(i[1])),round(float(i[2])),round(float(i[3])),round(float(i[4]))
# #     # width=x2-x1
# #     # length=y2-y1
# #     # print(x1)
# #     # plt.figure(8)
# #     # plt.imshow(img)
# #     # currentAxis = plt.gca()
# #     # rect = patches.Rectangle((x1, y1), width, length, linewidth=1, edgecolor='r', facecolor='none')
# #     # currentAxis.add_patch(rect)
# #     # cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4)
# #     # point_list=[]
# #     j=5
# #     while j<len(i) :
# #         x11=round(float(i[j]))
# #         y11=round(float(i[j+1]))
# #         # point_list.append('('+str(x11)+','+str(y11)+')')
# #         cv.circle(img, (x11,y11), 1, (0, 0, 255), 4)
# #         j=j+2
# #
# #     # for point in point_list:
# #     #     print(point)
# #         #cv.circle(img, point, 1, (0,0,255), 4)
# #     cv.imshow('img',img)
# #     key = cv.waitKey(0)
# #     if key == 27:
# #         cv.destroyAllWindows()
# #
# #
# #
# ##############CROP#######################################################
# file=open('II/label.txt','r')
# list_name=[]
# lines=file.readline()
# while lines:
#     str_list=lines.split(' ')
#     str_list[0]='II'+'/'+ str_list[0]
#     list_name.append(str_list)
#     lines=file.readline()
# # for i in range(len(list_name)):
# #     print(list_name[i])
# # print(len(list_name))
# # print(list_name[0])
# count=100000
# with open('image4/test.txt','w') as file_crop:
#     for i in list_name:
#         path=i[0]
#         img=cv.imread(path)
#         # cv.imshow('image', img)
#         x1,y1,x2,y2=round(float(i[1])),round(float(i[2])),round(float(i[3])),round(float(i[4]))
#         width=x2-x1
#         length=y2-y1
#         img_cut=img[y1:y2,x1:x2]
#         newpath='image4/'+str(count)+'.jpg'
#         count=count+1
#         temp = []
#         j = 5
#         temp.append(newpath)
#         for idx in i[1:5]:
#             temp.append(idx)
#         while j < len(i):
#             x1, y1 = float(i[1]), float(i[2])
#             x11 = float(i[j])
#             y11 = float(i[j + 1])
#             xt = x11 - x1
#             yt = y11 - y1
#             temp.append(xt)
#             temp.append(yt)
#             j = j + 2
#         for k in range(len(temp) - 1):
#             file_crop.write(str(temp[k]) + ' ')
#         file_crop.write(str(temp[-1]))
#         file_crop.write('\n')
#         cv.imwrite(newpath,img_cut)
#
# # # #
# # #
# #
# #
# #
# #
# #
# # ###############################################
#
#
#
#
#
# #
# # ##############test######################################
#
file=open('image4/test.txt','r')
list_name=[]
lines=file.readline()
while lines:
    str_list=lines.split(' ')
    str_list[0]=str_list[0]
    list_name.append(str_list)
    lines=file.readline()
# for i in range(len(list_name)):
#     print(list_name[i])
# print(len(list_name))
# print(list_name[0])
print(len(list_name))
for i in list_name:
    path=i[0]
    img=cv.imread(path)
    # x1,y1,x2,y2=round(float(i[1])),round(float(i[2])),round(float(i[3])),round(float(i[4]))
    # width=x2-x1
    # length=y2-y1
    # print(x1)
    # plt.figure(8)
    # plt.imshow(img)
    # currentAxis = plt.gca()
    # rect = patches.Rectangle((x1, y1), width, length, linewidth=1, edgecolor='r', facecolor='none')
    # currentAxis.add_patch(rect)
    # cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),4)
    # point_list=[]
    j=5
    w, h = img.shape[0], img.shape[1]
    x_scale, y_scale = 112 / w, 112 / h
    img=cv.resize(img,(112,112))
    print(img.shape)
    while j<len(i) :
        x11=round(float(i[j])*x_scale)
        y11=round(float(i[j+1])*y_scale)
        # point_list.append('('+str(x11)+','+str(y11)+')')

        cv.circle(img, (x11,y11), 1, (0, 0, 255), 4)
        j=j+2

    # for point in point_list:
    #     print(point)
        #cv.circle(img, point, 1, (0,0,255), 4)
    cv.imshow('img',img)
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyAllWindows()

