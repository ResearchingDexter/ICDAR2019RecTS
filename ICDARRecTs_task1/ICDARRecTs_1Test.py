import torch
import os
import json
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from ICDARRecTs_1DataSet import ICDARRecTs_1DataSet
from ICDARRecTs_1NN import desnet,se_resnext_101
import sys
sys.path.append('../')
from Logging import *
DEVICE='cuda'
BATCH_SIZE=8
PATH=r'E:\Files\ICDAR2019RecTs\ReCTS'
IMAGE_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS_test_part1\ReCTS_test_part1\Task1\img\\'#r'E:\Files\ICDAR2019RecTs\ReCTS\cropped_img\\'
MODEL_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
MODEL_NAME='optimal0.9687443188015853SEResNeXt101.pkl'#'optimal0.9926917063593063desnet169.pkl'#'desnet161.pkl'
NUM_CLASS=4135
def test():
    #model=desnet(NUM_CLASS)
    model=se_resnext_101(NUM_CLASS)
    model.load_state_dict(torch.load(MODEL_PATH+MODEL_NAME,map_location=DEVICE))
    model.to(DEVICE).eval()
    dataset=DataLoader(ICDARRecTs_1DataSet(IMAGE_PATH,img_transform=transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
                                                                          transforms.Resize((224,224)),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),train=False),
                       batch_size=BATCH_SIZE,shuffle=False,num_workers=4,drop_last=False)
    f=open(MODEL_NAME.split('.')[1]+'_task1.txt','w',encoding='utf-8')
    with open('E:\\Files\\ICDAR2019RecTs\\ReCTSdictionary_inv.json','r') as f1:
        dictionary_inv=json.load(f1)
    for step,data in enumerate(dataset):
        img,name=data
        img=Variable(img).to(DEVICE)
        pred=model(img)
        batch_size=pred.size(0)
        y_=pred.max(1)[1]
        for i in range(batch_size):
            y=y_[i].item()
            label=dictionary_inv[str(y)]
            logging.info('name:{}|pred:{}|label:{}'.format(name[i],y,label))
            #f.write('{},{}\n'.format(name[i],label))
            name_i=name[i].split('_')
            f.write(name_i[0]+'_'+name_i[-1]+','+label+'\n')
    logging.info('ended')
    f.close()
if __name__=='__main__':
    test()


