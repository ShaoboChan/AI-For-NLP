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

data_dir='./hymenoptera_data'
classes_num=2
model_name='resnet'
feature_extract=True
input_size=224
batch_size=64
data_transform={
    'train':
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
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
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transform[x]) for x in ['train','val']}
dataset_dict={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=1) for x in['train','val']}
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')


def imshow(tensor,title=None):
    imgage=tensor.cpu().clone()
    image=tensor.squeez(0)
    image=image.unloader
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)
    return
def set_parameter_requires_grad(model,features_grad):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad=False
    return
def initialization_model(model_name,num_classes, feature_extract, use_pretrained=True):
    model_fc=models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_fc,feature_extract)
    mff=model_fc.fc.in_features
    model_fc.fc=nn.Linear(mff,classes_num)
    input_size=224
    return model_fc,input_size
def train(model,dataloader,loss_fc,lr,optimizer,epochs):
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.
    val_acc_history=[]

    for epoch in range(epochs):
        for x in ['train','val']:
            loss=0.
            corr=0.
            if x=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.
            running_acc=0.
            for inputs,label in dataloader[x]:
                inputs,label=inputs.to(device),label.to(device)
                with torch.set_grad_enabled(x=='train'):
                    output=model(inputs)
                    loss=loss_fc(output,label)
                preds=output.argmax(dim=1)
                if x=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss +=loss.item()*inputs.size(0)
                running_acc +=torch.sum(preds.view(-1)==label.view(-1)).item()
            epoch_loss = running_loss / len(dataloader[x].dataset)
            epoch_acc = running_acc / len(dataloader[x].dataset)
            print("phase {} epoch  {} , acc {}".format(x, epoch_loss, epoch_acc))
            if x == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            if x == "val":
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
    model_ft, input_size = initialization_model(model_name,classes_num, feature_extract, use_pretrained=True)
    model_ft=model_ft.to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model_ft.parameters()), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    _, ohist = train(model_ft,dataset_dict,loss_fn,0.001,optimizer,20)




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





