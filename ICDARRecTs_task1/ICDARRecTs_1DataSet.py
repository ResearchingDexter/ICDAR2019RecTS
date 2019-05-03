from torch.utils.data import Dataset
from PIL import Image
import os
def load_img(path):
    return Image.open(path).convert('RGB')
class ICDARRecTs_1DataSet(Dataset):
    def __init__(self,path,img_transform=None,train=True):
        super(ICDARRecTs_1DataSet,self).__init__()
        self.path=path
        self.name_list=os.listdir(path)
        self.train=train
        self.img_transform=img_transform
    def __getitem__(self, item):
        name=self.name_list[item]
        img=load_img(self.path+name)
        if self.img_transform is not None:
            img=self.img_transform(img)
        if self.train==True:
            label=name.split('.')[-2]
            print('name:{}|label:{}'.format(name,label))
            label=int(label)
            return img,label
        return img,name
    def __len__(self):
        return len(self.name_list)