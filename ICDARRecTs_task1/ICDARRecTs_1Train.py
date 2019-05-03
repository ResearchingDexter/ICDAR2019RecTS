import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import sys
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from ICDARRecTs_1DataSet import ICDARRecTs_1DataSet
from ICDARRecTs_1NN import desnet,se_resnext_101
from IPython.display import clear_output
sys.path.append('../')
from Logging import *
import pdb
#from ICDARRecTs_1Configure import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="0"
DEVICE='cuda'
BATCH_SIZE=8
EPOCH=10000
PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
IMAGE_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\cropped_img\\'
MODEL_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
MODEL_NAME='SEResNeXt101.pkl'#'desnet161.pkl'
PRETRAIN=False
NUM_CLASS=4135
LR=0.001
MAX_ACCURACY=0
def train(pretrain=PRETRAIN):
    #model=desnet(NUM_CLASS)
    model=se_resnext_101(NUM_CLASS)
    if(DEVICE=='cuda'):
        if torch.cuda.is_available()==False:
            logging.debug('cuda is not available')
            pdb.set_trace()
    if pretrain==True:
        model.load_state_dict(torch.load(MODEL_PATH+MODEL_NAME,map_location=DEVICE))
    model.to(DEVICE).train()
    optimizer=optim.Adam(model.parameters(),lr=LR,betas=(0.9,0.999))
    #optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.99)
    criterion=CrossEntropyLoss()
    dataset=DataLoader(ICDARRecTs_1DataSet(IMAGE_PATH,img_transform=transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
                                                                          transforms.Resize((224,224)),
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])),
                       batch_size=BATCH_SIZE,shuffle=False,num_workers=4,drop_last=False)
    length=len(dataset)*BATCH_SIZE
    max_accuracy = MAX_ACCURACY
    for epoch in range(EPOCH):
        epoch_time=datetime.now()
        correct=0
        min_loss=100
        sum_loss=0
        for step,data in enumerate(dataset):
            step_time=datetime.now()
            img,label=data
            img=Variable(img,requires_grad=True).to(DEVICE)
            label=Variable(label.long()).to(DEVICE)
            pred=model(img)
            y=pred.max(1)[1]
            batch_correct=(y==label).sum().item()
            correct+=batch_correct
            optimizer.zero_grad()
            loss=criterion(pred,label)
            loss.backward()
            optimizer.step()
            sum_loss+=loss.item()
            if loss.item()<min_loss:
                min_loss=loss.item()
                torch.save(model.state_dict(),MODEL_PATH+MODEL_NAME)
            logging.debug('length:{}|epoch:{}|step:{}|loss:{}|batch_correct:{}'.format(length,epoch,step,loss.item(),batch_correct))
            logging.debug('the cost of time:{}'.format(datetime.now()-step_time))
            if step%100==0:
                clear_output(wait=True)
        accuray=correct/length
        if(accuray>max_accuracy):
            max_accuracy=accuray
            torch.save(model.state_dict(),MODEL_NAME+'optimal'+str(max_accuracy)+MODEL_NAME)
            torch.save(model.state_dict(), MODEL_PATH + MODEL_NAME)
        logging.debug('accuracy:{}|the cost of time in one epoch:{}'.format(accuray,datetime.now()-epoch_time))
        with open('accuracy.txt','r') as f:
            f.write('epoch:{}|accuracy:{}|mean_loss:{}|MAX_ACCURACY:{}'.format(epoch,accuray,sum_loss/length,max_accuracy))
if __name__=='__main__':
    train()


