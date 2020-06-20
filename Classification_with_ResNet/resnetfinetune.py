import numpy as np
import torchvision
from torchvision import datasets,transforms,models
import torch
import torch.nn as nn
import torch.utils.data.dataloader as DL
import matplotlib.pyplot as plt
import time
import os
import  copy
print("version:",torchvision.__version__)
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "./hymenoptera_data"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 15
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
input_size = 224

##################预处理#####################
############################################



all_imgs=datasets.ImageFolder(os.path.join(data_dir,"train"),transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),#反转
    # transforms.RandomAffine(),
    transforms.ToTensor(),
]))

loader = torch.utils.data.DataLoader(all_imgs,batch_size=batch_size,shuffle=True,num_workers=4)


####################################分批###########################################################################
###################################################################################################################
data_transforms={
    'train':
    transforms.Compose([transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.255])
]),
    'val':
        transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

}
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
dataloaders_dict={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=2) for x in ['train','val']}
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
def set_parameter_requires_grad(model,feature_extract):########?????????????????
    if feature_extract:
        for param in model.parameters():
            param.requires_grad=False
    return

def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    if model_name=='resnet':
        model_ft=models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft,feature_extract)
        #
        #提取fc层中固定的参数
        num_ftrs=model_ft.fc.in_features#in_feature改变原有的fc结构，原来的为512
        model_ft.fc=nn.Linear(num_ftrs,num_classes)##分类
        input_size=224
    else:
        print('404 not found')
        return None,None
    return model_ft,input_size
def train(model,dataloaders,loss_fn,batch_size,optimizer,epoch_num=100):
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0
    val_acc_history=[]
    for epoch in range(epoch_num):
        for phase in ['train','val']:
            running_loss=0.
            running_corr=0.
            if phase=='train':
                model.train()
            if phase=='val':
                 model.eval()
            for inputs,labels in dataloaders[phase]:
                inputs, labels=inputs.to(device),labels.to(device)
                with torch.autograd.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    loss=loss_fn(outputs,labels)
                pres=outputs.argmax(dim=1)
                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss+=loss.item()*inputs.batch_size#size(0)
                running_corr+=torch.sum(pres.view(-1)==labels.view(-1)).item()
            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            epoch_acc=running_corr/len(dataloaders[phase].dataset)
            print("phase {} epoch  {} , acc {}".format(phase,epoch_loss,epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
if __name__ == '__main__':

    # img = next(iter(dataloaders_dict["val"]))[0]
    # # print(img.shape)
    #
    # unloader = transforms.ToPILImage()  # reconvert into PIL image
    #
    # plt.ion()

    #
    # model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)

    # plt.figure()
    # imshow(img[11], title='Image')
    model_ft, input_size = initialize_model(model_name,
                                            num_classes, feature_extract, use_pretrained=True)
    model_ft=model_ft.to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model_ft.parameters()), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    _, ohist = train(model_ft, dataloaders_dict, loss_fn, optimizer, epoch_num=100)




#
# model_scratch, _ = initialize_model(model_name,
#                     num_classes, feature_extract=False, use_pretrained=False)
# model_scratch = model_scratch.to(device)
# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
#                                    model_scratch.parameters()), lr=0.001, momentum=0.9)
# loss_fn = nn.CrossEntropyLoss()
# _, scratch_hist = train_model(model_scratch, dataloaders_dict, loss_fn, optimizer, num_epochs=num_epochs)


#
# plt.title("Validation Accuracy vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
# plt.plot(range(1,num_epochs+1),scratch_hist,label="Scratch")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1, num_epochs+1, 1.0))
# plt.legend()
# plt.show()






# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=2, bias=True)
# )