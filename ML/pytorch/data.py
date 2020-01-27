import numpy as np
import cv2 as cv

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageDraw
import itertools

folder_list = ['I', 'II']
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)  # 均值
    std = np.std(img)  # 标准差
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)  # 扩大维度
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        try:
            img = Image.open(img_name).convert('L')
            img_crop = img.crop(tuple(rect))
            landmarks = np.array(landmarks).astype(np.float32)
            w, h = img.size[0], img.size[1]
            # print(w,h)
            x_scale, y_scale = train_boarder / w, train_boarder / h
            # print(x_scale, y_scale)
            for i in range(len(landmarks)):
                if i% 2 == 0:
                    # print(landmarks[i])
                    landmarks[i] = x_scale * landmarks[i]
                    # print(landmarks[i])
                else:
                    # print(landmarks[i])
                    landmarks[i] = y_scale * landmarks[i]
                    # print(landmarks[i])
            sample = {'image': img_crop, 'landmarks': landmarks}
            sample = self.transform(sample)

            return sample
        except(OSError, NameError):
            print('fail to open')
        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:




def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),  # do channel normalization
            ToTensor()]  # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines,  transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set,valid_set
        # , valid_set


# if __name__ == '__main__':
#     train_set = load_data('train')
#     for i in range(1, len(train_set)):
#         sample = train_set[i]
#         img = sample['image']
#         landmarks = sample['landmarks']
#         ## 请画出人脸crop以及对应的landmarks
#         # please complete your code under this blank
#         file = open('image3/train.txt', 'r')
#         list_name = []
#         lines = file.readline()
#         while lines:
#             str_list = lines.split(' ')
#             str_list[0] = str_list[0]
#             list_name.append(str_list)
#             lines = file.readline()
#         # for i in range(len(list_name)):
#         #     print(list_name[i])
#         # print(len(list_name))
#         # print(list_name[0])
#         print(len(list_name))
#         for i in list_name:
#             path = i[0]
#             img1 = cv.imread(path)
#
#             j=0
#
#             while j < len(landmarks):
#                 print(landmarks[j])
#                 print(landmarks[j+1])
#                 x11 = round(float(landmarks[j]))
#                 y11 = round(float(landmarks[j + 1]))
#                 # point_list.append('('+str(x11)+','+str(y11)+')')
#                 cv.circle(img1, (x11, y11), 1, (0, 0, 255), 4)
#                 j = j + 2
#             cv.imshow('image', img1)
#             key = cv.waitKey()
#             if key == 27:
#
#                 cv.destroyAllWindows()




