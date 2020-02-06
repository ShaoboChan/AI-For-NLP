from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import types
#train:(7353,565) 最后三列：user,active,label
df=pd.read_csv('train_addLabel.csv',header=None,sep=',')
# print(df.shape)
# print(df[562]) subject
# def readCsv(phase):
#     dir='./'+phase+'_addLable.csv'
#     feat=pd.read_csv(dir)
# features=df.iloc[2:7353,0:562]
# labels=df.iloc[2:7353,564]
# print('features:{}'.format(features))  #(7351,562)
# print(features.shape)
# print('labels:{}'.format(labels))     #(7351,)
# print(labels.shape)
class FeatureDataset(Dataset):
    def __init__(self,csv_File,transform=None):
        self.df=pd.read_csv('train_addLabel.csv',header=None,sep=',')
        self.to_tensor=torch.FloatTensor()
        var_list=[]
        for i in range(2,7353):

            var_list.append(list(map(float,df.iloc[i,1:562])))

        self.features=var_list
        # self.features=df.iloc[2:7353,0:562]
        # self.features=np.array(self.features).astype(np.float32)
        labels=list(map(int,df.iloc[2:7353,564]))
        self.labels=np.array(labels)
        # self.root_dir=root_dir
        # self.labels=df.iloc[2:7353,564]

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        feature=self.features[idx]
        feature=np.array([feature])
        feature=feature.astype('float')
        label=self.labels[idx]

        # feature=list(map(float,feature))

        # self.feature=feature.astype('float')

        # feature=self.to_tensor(feature)
        # label=self.labels[idx]
        sample={'feature':feature,'label':label}
        return sample

    # def build_data(self):




if __name__ == "__main__":
    # custom_mnist_from_images =  \
    #     a=FeatureDataset('./train_addLabel.csv').features
    # print(a)
    # feature,label=FeatureDataset('./train_addLabel.csv').features,FeatureDataset('./train_addLabel.csv').labels
    # for i in range(len()):
    # print(feature)
    # print('???????????')
    # print(feature[0])
    # print(type(feature),type(label))
    var_f=FeatureDataset('./train_addLabel.csv')
    for i in range(7350):
        sample=var_f[i]
        # print(i, sample['feature'].shape, sample['label'])
    print(var_f.__len__())
    sample1=var_f[1]
    sample1['feature']
















