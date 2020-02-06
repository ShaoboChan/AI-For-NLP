from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
import pandas as pd
import numpy as np
df=pd.read_csv('train.csv',header=None,sep=',') #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
print (df.head(1))
print (df.tail(1))
print(df)
print(df.shape)
# print(df.info())
# df[:].astype('int')
print(df[562][100])#列；行
# df[:562][1:].astype('float')
df['label']=None
df['label'][0]='label'
for i in range(1,len(df['label'])):
    if df[562][i] == 'STANDING':
        df['label'][i]=0
    elif df[562][i] == 'SITTING':
        df['label'][i]=1
    elif df[562][i] == 'LAYING':
        df['label'][i]=2
    elif df[562][i] == 'WALKING':
        df['label'][i]=3
    elif df[562][i] == 'WALKING_DOWNSTAIRS':
        df['label'][i]=4
    elif df[562][i]== 'WALKING_UPSTAIRS':
        df['label'][i]=5

df.to_csv('./train_addLabel.csv')


# print(df.columns)
# class CustomDatasetFromImages(Dataset):
#     def __init__(self, csv_path):
#         """
#         Args:
#             csv_path (string): csv 文件路径
#             img_path (string): 图像文件所在路径
#             transform: transform 操作
#         """
#         # Transforms
#         self.to_tensor = transforms.ToTensor()
#         # 读取 csv 文件
#         self.data_info = pd.read_csv(csv_path, header=None)
#         for i in range(len(self.data_info.index)):
#             #     print(item[-1])
#             if self.data_info.iloc[i][-1] == 'STANDING':
#                 self.data_info.iloc[i].append(0)
#             elif self.data_info.iloc[i][-1] == 'SITTING':
#                 self.data_info.iloc[i].append(1)
#             elif self.data_info.iloc[i][-1] == 'LAYING':
#                 self.data_info.iloc[i].append(2)
#             elif self.data_info.iloc[i][-1] == 'WALKING':
#                 self.data_info.iloc[i].append(3)
#             elif self.data_info.iloc[i][-1] == 'WALKING_DOWNSTAIRS':
#                 self.data_info.iloc[i].append(4)
#             elif self.data_info.iloc[i][-1] == 'WALKING_UPSTAIRS':
#                 self.data_info.iloc[i].append(5)
#             else:
#                 self.data_info.iloc[i].append('label')
#         # 文件第一列包含图像文件的名称
#         self.features = np.asarray(self.data_info.iloc[1:, :-2])
#         # 第二列是图像的 label
#         self.label_arr = np.asarray(self.data_info.iloc[:, -1])
#         # 计算 length
#         self.data_len = len(self.data_info.index)
#
#     def __getitem__(self, index):
#         # 从 pandas df 中得到文件名
#         single_feature = self.features[index]
#         feature_tensor=torch.FloatTensor(single_feature)
#         # 得到图像的 label
#         single_label = self.label_arr[index]
#
#         return (feature_tensor, single_label)
#
#     def __len__(self):
#         return self.data_len
#
#
# if __name__ == "__main__":
#     custom_mnist_from_images =CustomDatasetFromImages('./train.csv')