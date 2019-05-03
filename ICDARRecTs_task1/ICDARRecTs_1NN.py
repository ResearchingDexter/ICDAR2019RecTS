import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from typing import Any,Tuple
__all__=['desnet','se_resnext_101','se_resnext_152']
def desnet(class_nums):
    model=models.densenet169(pretrained=True)
    infeatures=model.classifier.in_features
    model.classifier=nn.Linear(in_features=infeatures,out_features=class_nums)
    return model
def se_resnext_101(class_nums):#(3,4,6,3):50
    return SEResNeXt(class_nums,Bottleneck,(3,4,23,3))
def se_resnext_152(class_nums):
    return SEResNeXt(class_nums,Bottleneck,(3,8,36,3))
class SELayer(nn.Module):
    def __init__(self,channel:int,reduction:int=16,drop_rate:float=0):
        super(SELayer,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(in_features=channel,out_features=channel//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel//reduction,out_features=channel,bias=True),
            nn.Sigmoid()
        )
        self.drop_rate=drop_rate
    def forward(self, input:torch.Tensor):
        b,c,_,_=input.size()
        output=self.avg_pool(input).squeeze(-1).squeeze(-1)
        output=self.fc(output).reshape(b,c,1,1)
        if self.drop_rate>0 and self.drop_rate<1:
            output=F.dropout(output,p=self.drop_rate,training=self.training)
        return input*output.expand_as(input)
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes:int,planes:int,stride:int=1,group:int=32,downsample:nn.Sequential=None,SE:nn.Module=SELayer):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inplanes,planes,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(planes,planes,3,stride=stride,padding=1,groups=group,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(planes,planes*self.expansion,1,bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.SE=SE(channel=planes*self.expansion)
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.SE is not None:
            output=self.SE(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)
class SEResNeXt(nn.Module):
    def __init__(self,num_classes:int,block:nn.Module=Bottleneck,nums_block_tuple:Tuple=(3,4,23,3)):
        super(SEResNeXt,self).__init__()
        self.inplanes=64
        self.conv=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1=self._make_layer(block,64,nums_block_tuple[0],stride=1)
        self.layer2=self._make_layer(block,128,nums_block_tuple[1],stride=2)
        self.layer3=self._make_layer(block,256,nums_block_tuple[2],stride=2)
        self.layer4=self._make_layer(block,512,nums_block_tuple[3],stride=2)
        self.avg_pool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.linear=nn.Linear(self.inplanes,num_classes)
    def forward(self, input):
        output=self.conv(input)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.avg_pool(output).reshape(output.size(0),-1)
        output=self.linear(output)
        return output
    def _make_layer(self,block,inplanes,nums_block,kernel_size=1,stride=1,padding=1):
        downsample=None
        print('inplanes:{}'.format(self.inplanes))
        if stride!=1 or self.inplanes!=inplanes*block.expansion:
            print('expansion'.format(block.expansion))
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,inplanes*block.expansion,kernel_size=kernel_size,stride=stride,bias=False),
                                     nn.BatchNorm2d(inplanes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,inplanes,stride=stride,downsample=downsample))
        self.inplanes=inplanes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.inplanes,inplanes))
        return nn.Sequential(*layers)
if __name__=='__main__':
    desnet(1)