import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict,namedtuple
from typing import Union,List,Tuple
__all__=['DenseLSTM','VGGLSTM','DenseCNN','VGGFC','ResNetLSTM']
kernel_sizes=[(2,2),(2,1),(2,1),(2,1)]
strides=[(2,2),(2,1),(2,1),(2,1)]
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
vgg_kernel_sizes=((2,2),(2,2),(2,2),(2,1),(2,1))
vgg_strides=((2,2),(2,2),(2,2),(2,1),(2,1))
class ResNetLSTM(nn.Module):
    def __init__(self,num_classes=4134+1):
        super(ResNetLSTM,self).__init__()
        self.feature_extractor=ResNet()
        self.decoder=nn.Sequential(BLSTM(self.feature_extractor.inplanes,512,num_out=512,drop_rate=0),
                                   BLSTM(512,512,num_out=num_classes,drop_rate=0))
        self.logsoftmax=nn.LogSoftmax(-1)
    def forward(self, input):
        output=self.feature_extractor(input)
        b, c, h, w = output.size()
        assert h == 1, "b:{}|c:{}|h:{}|w:{}".format(b, c, h, w)
        #output=output.permute(0,1,3,2).reshape(b,c,w*h).permute(0,2,1)
        output=output.squeeze(2).permute(0,2,1)
        output=self.decoder(output)
        output=output.permute(1,0,2)
        return self.logsoftmax(output)
class VGGFC(nn.Module):
    def __init__(self,num_classes=4134+1):
        super(VGGFC,self).__init__()
        self.feature_extractor=vgg_13()
        self.decoder=nn.Sequential(nn.Linear(self.feature_extractor.num_features,self.feature_extractor.num_features*2),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0),
                                   nn.Linear(self.feature_extractor.num_features*2,self.feature_extractor.num_features*4),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.1),
                                   nn.Linear(self.feature_extractor.num_features*4,num_classes),
                                   nn.Dropout(p=0.1))
        self.logsoftmax=nn.LogSoftmax(-1)
    def forward(self, input):
        output=self.feature_extractor(input)
        b, c, h, w = output.size()
        assert h == 1, "b:{}|c:{}|h:{}|w:{}".format(b, c, h, w)
        output=output.squeeze(2).permute(0,2,1).reshape(b*w,c)
        output=self.decoder(output)
        output=output.reshape(b,w,-1)
        output=output.permute(1,0,2)
        return self.logsoftmax(output)
class DenseCNN(nn.Module):
    def __init__(self,num_classes=4134+1):
        super(DenseCNN,self).__init__()
        self.feature_extractor=DenseNet()
        self.decoder=nn.Sequential(nn.BatchNorm2d(self.feature_extractor.num_features),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(self.feature_extractor.num_features,num_classes,kernel_size=1,bias=False),
                                   nn.BatchNorm2d(num_classes),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(num_classes,num_classes,kernel_size=3,stride=1,padding=1))
        self.logsoftmax=nn.LogSoftmax(-1)
    def forward(self, input):
        output=self.feature_extractor(input)
        output=self.decoder(output)
        b, c, h, w = output.size()
        assert h == 1, "b:{}|c:{}|h:{}|w:{}".format(b, c, h, w)
        output=output.squeeze(2)
        output=output.permute(2,0,1)
        return self.logsoftmax(output)
class VGGLSTM(nn.Module):
    def __init__(self,num_classes=4134+1):
        super(VGGLSTM,self).__init__()
        self.feature_extractor=vgg_13()
        self.decoder=nn.Sequential(BLSTM(self.feature_extractor.num_features,num_hidden=512,num_out=512,drop_rate=0),
                                   BLSTM(512,512,num_out=num_classes,drop_rate=0))
        self.logsoftmax=nn.LogSoftmax(-1)
    def forward(self, input):
        output=self.feature_extractor(input)
        b,c,h,w=output.size()
        assert h==1,"b:{}|c:{}|h:{}|w:{}".format(b,c,h,w)
        output=output.squeeze(2)
        output=output.permute(0,2,1)
        output=self.decoder(output)
        output=self.logsoftmax(output)
        output=output.permute(1,0,2)
        return output
class DenseLSTM(nn.Module):
    def __init__(self,num_classes=4134+1):# +1 for the blank
        super(DenseLSTM,self).__init__()
        self.feature_extractor=DenseNet()
        self.decoder=BLSTM(self.feature_extractor.num_features,num_hidden=512,num_out=num_classes)
        self.logsoftmax=nn.LogSoftmax(-1)
    def forward(self, input):
        output=self.feature_extractor(input)
        b,c,h,w=output.size()
        assert h==1,"b:{}|c:{}|h:{}|w:{}".format(b,c,h,w)
        output=output.squeeze(2)
        output=output.permute(0,2,1)
        output=self.decoder(output)
        output=self.logsoftmax(output)
        output=output.permute(1,0,2)
        return output
class _DenseLayer(nn.Sequential):
    def __init__(self,num_in_features,grow_rate,bn_size,drop_rate):
        super(_DenseLayer,self).__init__()
        self.add_module('norm1',nn.BatchNorm2d(num_in_features))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(num_in_features,bn_size*grow_rate,kernel_size=1,stride=1,bias=False))
        self.add_module('norm2',nn.BatchNorm2d(bn_size*grow_rate))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(bn_size*grow_rate,grow_rate,kernel_size=3,stride=1,padding=1,bias=False))
        self.drop_rate=drop_rate
    def forward(self, input):
        new_features=super(_DenseLayer,self).forward(input)
        if self.drop_rate>0 and self.drop_rate<1:
            new_features=F.dropout(new_features,p=self.drop_rate,training=self.training)
        return torch.cat((input,new_features),1)
class _DenseBlock(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
        super(_DenseBlock,self).__init__()
        for i in range(num_layers):
            layer=_DenseLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module('denselayer%d'%(i+1),layer)
class _Transition(nn.Sequential):
    r"""
    Args: kernel_size (int or tuple): Size of the pooling kernel. Default:2
          stride (int or tuple, optional): Stride of the pool. Default: 2
    """
    def __init__(self,num_input_features:int,kernel_size:Union[int,tuple]=2,stride:Union[int,tuple]=2,theata:float=0.5):
        """
        :param num_input_features:
        :param num_output_features:
        :param theata: compression rate of the features map
        """
        super(_Transition,self).__init__()
        num_output_features=int(num_input_features*theata)
        self.add_module('norm',nn.BatchNorm2d(num_input_features))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,kernel_size=1,stride=1,bias=False))
        self.add_module('pool',nn.AvgPool2d(kernel_size=kernel_size,stride=stride))
class DenseNet(nn.Module):
    def __init__(self,growth_rate=32,block_config=(6,12,48,32),num_init_features=64,bn_size=4,drop_rate=0,theata=0.5):
        super(DenseNet,self).__init__()
        self.features=nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3,num_init_features,kernel_size=7,stride=2,padding=3,bias=False)),
            ('norm0',nn.BatchNorm2d(num_init_features)),
            ('relu0',nn.ReLU(inplace=True)),
            ('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        ]))
        num_features=num_init_features
        for i,num_layers in enumerate(block_config):
            block=_DenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,
                              growth_rate=growth_rate,drop_rate=drop_rate)
            self.features.add_module('denseblock%d' %(i+1),block)
            num_features+=num_layers*growth_rate
            if i!=len(block_config)-1:
                trans=_Transition(num_input_features=num_features,kernel_size=kernel_sizes[i],stride=strides[i],theata=theata)
                self.features.add_module('transition%d' %(i+1),trans)
                num_features=int(num_features*theata)
        self.features.add_module('norm5',nn.BatchNorm2d(num_features))
        self.num_features=num_features
        print('num_features:{}'.format(self.num_features))
    def forward(self, input):
        features=self.features(input)
        out=F.relu(features,inplace=True)
        return out
class BLSTM(nn.Module):
    def __init__(self,num_in,num_hidden,num_layers=1,num_out=4135,drop_rate=0.1,batch_normalization=True):
        super(BLSTM,self).__init__()
        self.batch_normalization=batch_normalization
        self.LSTM=nn.LSTM(num_in,num_hidden,num_layers,batch_first=True,bidirectional=True)
        if batch_normalization==True:
            self.bn1 = nn.BatchNorm1d(num_features=num_hidden * 2)
        self.embedding=nn.Linear(num_hidden*2,num_out)
        self.drop_rate=drop_rate
    def forward(self, input):
        out,_=self.LSTM(input)
        b,t,h=out.size()
        if self.batch_normalization==True:
            out = out.permute(0, 2, 1)
            out = self.bn1(out)
            out = out.permute(0, 2, 1)

        out=out.reshape(b*t,h)
        #out=out.view(b*t,h)
        out=self.embedding(out)
        if self.drop_rate>0 and self.drop_rate<1:
            out=F.dropout(out,p=self.drop_rate,training=self.training)
        output=out.view(b,t,-1)
        return output
class VGG(nn.Module):
    def __init__(self,cfg:List,batch_normal:bool=False,kernel_sizes=vgg_kernel_sizes,strides=vgg_strides,in_channel:int=3):
        super(VGG,self).__init__()
        self.num_features=in_channel
        self.features=self._make_layer(cfg,batch_normal,kernel_size=kernel_sizes,stride=strides)
    def forward(self, input):
        return self.features(input)
    def _make_layer(self,cfg:List,batch_normal:bool,kernel_size:Tuple,stride:Tuple):
        layers=[]
        i=0
        for v in cfg:
            if v=='M':
                layers.append(nn.MaxPool2d(kernel_size[i],stride[i]))
                i+=1
            else:
                #print('v',self.num_features,v)
                layers.append(nn.Conv2d(self.num_features,v,kernel_size=3,padding=1))
                if batch_normal==True:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                self.num_features=v
        return nn.Sequential(*layers)
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes:int,planes:int,kernel_size:Union[int,Tuple]=3,stride:Union[int,Tuple]=1,padding:Union[int,Tuple]=1,group:int=1,downsample:nn.Sequential=None):
        super(Bottleneck,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inplanes,planes,1,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(planes,planes,kernel_size,stride=stride,padding=padding,groups=group,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(planes,planes*self.expansion,1,bias=False),
            nn.BatchNorm2d(planes*self.expansion)
        )
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        #self.SE=SE(channel=planes*self.expansion)
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        #if self.SE is not None:
            #output=self.SE(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)
class ResNet(nn.Module):
    def __init__(self,block:nn.Module=Bottleneck,nums_block_tuple:Tuple=(3,4,23,3)):
        super(ResNet,self).__init__()
        self.inplanes=64
        self.conv=nn.Sequential(nn.Conv2d(3,64,5,1,2,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,2,1)
                                )
        self.layer1=self._make_layer(block,64,nums_block_tuple[0],stride=1)
        self.layer2=self._make_layer(block,128,nums_block_tuple[1],stride=2)
        self.layer3=self._make_layer(block,256,nums_block_tuple[2],stride=2)
        self.layer4=self._make_layer(block,512,nums_block_tuple[3],(3,3),(2,1),(1,1))
        self.maxpool=nn.MaxPool2d(3,(2,1),(1,1))
    def forward(self, input):
        output=self.conv(input)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.maxpool(output)
        return output
    def _make_layer(self,block,inplanes,nums_block,kernel_size:Union[int,Tuple]=3,stride:Union[int,Tuple]=1,padding:Union[int,Tuple]=1):
        downsample=None
        print('inplanes:{}'.format(self.inplanes))
        if stride!=1 or self.inplanes!=inplanes*block.expansion:
            print('expansion'.format(block.expansion))
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,inplanes*block.expansion,kernel_size=1,stride=stride,bias=False),
                                     nn.BatchNorm2d(inplanes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,inplanes,kernel_size=kernel_size,stride=stride,padding=padding,downsample=downsample))
        self.inplanes=inplanes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.inplanes,inplanes))
        return nn.Sequential(*layers)
def vgg_11(cfg:dict=cfg):
    return VGG(cfg['A'])
def vgg_13(cfg:dict=cfg):
    return VGG(cfg['B'])
def vgg_13bn(cfg:dict=cfg):
    return VGG(cfg['B'],batch_normal=True)
if __name__=='__main__':
    #point=namedtuple('point',['x','y'])
    #a=point([7,8],[9,0])
    temp=DenseNet()
    import torchvision
    a=torchvision.models.resnet101()
    print(temp)
    print(a)
    #print(a)


