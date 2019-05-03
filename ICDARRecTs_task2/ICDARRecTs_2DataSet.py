from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Any,Optional,List,Iterable
import random
import cv2
import numpy as np
import os
import math
import pdb
from ICDARRecTs_2Preprocessing import rotate_crop
def load_img(path:str)->Image.Image:
    return Image.open(path).convert('RGB')
class ICDARRecTs_2DataSet(Dataset):
    def __init__(self,img_path:str,dictionary:Optional[dict]=None,coordinates_path:Optional[str]=None,img_transform:Any=None,train=True,load_img=load_img):
        super(ICDARRecTs_2DataSet,self).__init__()
        self.img_path=img_path
        self.dictionary=dictionary
        self.coordinates_path=coordinates_path
        self.img_transform=img_transform
        self.train=train
        self.load_img=load_img
        if train==True:
            with open('label_less_30.txt','r',encoding='UTF -8') as f:
                self.labels=f.readlines()
        else:
            self.img_names=os.listdir(img_path)
    def __getitem__(self, item):
        if self.train==True:
            name_label=self.labels[item]
            name_label=name_label.split(',')
            #name_label=name_label.split(' ')
            name=name_label[0]
            label=''.join(name_label[1:])
            if len(label)==1:
                label=',\n'
            """
            figure_label=[]
            for s in label:
                s_=self.dictionary.get(s,default=-1)
                if s==-1:
                    print('name:{}|label:{}'.format(name,label))
                figure_label.append(int(s_)+1)#0 for blank"""
            img=self.load_img(self.img_path+name)
            w,h=img.size
            theata=random.random()#
            if theata>=0.5:
                #print('theata:{:.4f}'.format(theata))
                if h>=w:
                    img = img.transpose(Image.ROTATE_90)
            else:
                #print('theata:{:.4f}'.format(theata))
                if h>w:
                    img = img.transpose(Image.ROTATE_90)
            img=self._resize_img(img=img,w=400,h=32)#376=47*8
            w,h=img.size
            if w<400:
                img=self._pad_img(img,(400,32))
            if self.img_transform is not None:
                img=self.img_transform(img)
            return img,label[:-1],len(label[:-1]),name    #image,the label ,the length of the label
        else:
            img_name=self.img_names[item]
            with open(self.coordinates_path+img_name.split('.')[0]+'.txt','r') as f:
                coordinate=f.readlines()[0].split(',')
            coordinate=list(map(int,coordinate))
            img=self.load_img(self.img_path+img_name)
            print('before rotate crop:{}'.format(img.size))
            img=self._PIL2cv(img)
            img=rotate_crop(img,coordinate)
            img=self._cv2PIL(img)
            print(img.size)
            w,h=img.size
            if h>=w:
                img = img.transpose(Image.ROTATE_90)
            img=self._resize_img(img=img,w=600,h=32)
            w,h=img.size
            #print(img.size)
            #if w<240:
                #img=self._pad_img(img,(240,32))
            if self.img_transform is not None:
                img=self.img_transform(img)
            return img,img_name
    def __len__(self):
        if self.train==True:
            return len(self.labels)
        else:
            return len(self.img_names)
    def collate(self,batch_data:List):
        elem_type=batch_data[0]
        if isinstance(batch_data[0],str):
            return batch_data
        elif isinstance(batch_data[0],Image.Image):
            """
            max_w,max_h=batch_data[0].size
            for img in batch_data[1:]:
                w,h=img.size
                if w>max_w:
                    max_w=w
                if h>max_h:
                    max_h=h
            """
            batch_img=[]
            for img in batch_data:
                img=self._pad_img(img,(self.max_w,self.max_h))
                if self.img_transform is not None:
                    img=self.img_transform(img)
                    batch_img.append(img)
                else:
                    print('please input tansform')
                    pdb.set_trace()
            return torch.stack(batch_img,dim=0)
        elif isinstance(batch_data[0],int):
            return torch.LongTensor(batch_data)
        elif isinstance(batch_data[0],Iterable):
            transposed=zip(*batch_data)
            return [self.collate(samples) for samples in transposed]
        else:
            raise TypeError('Expected get Image or string but get the type:{}'.format(elem_type))
    def _cv2PIL(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        return img
    def _PIL2cv(self,img):
        img=np.asarray(img)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img
    def _resize_img(self,img:Image.Image,w=376,h=32):
        img_w,img_h=img.size
        if img_h<h:
            theata=math.ceil(h/img_h)
            img_w*=theata
        elif img_h>h:
            theata=h/img_h
            img_w=math.ceil(img_w*theata)
        #print('img_w:{}'.format(img_w))
        img=img.resize((img_w,h))
        if img_w>w:
            return img.resize((w,h))
        return img
    def _pad_img(self,img,img_size_padded:tuple):#img_size_padded=(w,h)
        w,h=img.size
        print(img_size_padded)
        img_array=np.array(img)
        img_array=img_array.transpose((2,0,1))
        img_w_left=img_size_padded[0]-w
        img_w_right=img_w_left//2
        img_w_left-=img_w_right
        img_array=np.pad(img_array,((0,0),(0,img_size_padded[1]-h),(img_w_left,img_w_right)),mode='constant',constant_values=0)
        #img_array=np.pad(img_array,((0,0),(0,img_size_padded[1]-h),(0,img_size_padded[0]-w)),mode='constant',constant_values=0)
        img_array=img_array.transpose((1,2,0))
        img=Image.fromarray(img_array)
        #img1.show()
        return img
    def transform_label(self,batch_name):
        batch_label=[]
        one_dim_label_figure=[]
        for name in batch_name:
            label=[]
            for s in name:
                s_=self.dictionary.get(s,-1)
                if s_==-1:
                    print('name:{}|'.format(s))
                label.append(int(s_))#0 for blank
                one_dim_label_figure.append(int(s_))
            batch_label.append(torch.LongTensor(label))
        """
        one_dim_label=''.join(batch_name)
        one_dim_label_figure=[]
        for s in one_dim_label:
            s_ = self.dictionary.get(s, default=-1)
            if s_ == -1:
                print('name:{}|'.format(s))
            one_dim_label_figure.append(int(s_) + 1)  # 0 for blank"""
        return torch.LongTensor(one_dim_label_figure),batch_label
if __name__=='__main__':
    a=ICDARRecTs_2DataSet('SS')
    print(a)
    """
    print(type(load_img))
    with open('label.txt','r',encoding='UTF-8') as f:
        temp=f.readlines()
        print(temp[0].split(',')[-1][:-1])"""
