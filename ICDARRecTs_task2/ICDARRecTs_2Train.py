import torch
from torch.autograd import Variable
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim import Adam,Adadelta
import json
from torchvision import transforms
from IPython.display import clear_output
from ICDARRecTs_2DataSet import ICDARRecTs_2DataSet
from ICDARRecTs_2NN import DenseLSTM,VGGLSTM,DenseCNN,VGGFC,ResNetLSTM
import pdb
import os
import sys
sys.path.append('../')
from Logging import *
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES']='0'
DEVICE='cuda'
BATCH_SIZE=8
EPOCH=10000
PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
DICTIONARY_NAME='RecTs2dictionary.json'
IMAGE_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\task2_cropped_img_less_30\\'
MODEL_PATH=r'E:\Files\ICDAR2019RecTs\ReCTS\\'
MODEL_NAME='DenseCNN.pkl'
PRETRAIN=False
NUM_CLASS=4134+1
LR=0.001
MAX_ACCURACY=0
def train(pretrain=PRETRAIN):
    logging.debug('pretrain:{}'.format(pretrain))
    if DEVICE=='cuda':
        if torch.cuda.is_available()==False:
            logging.error("can't find a GPU device")
            pdb.set_trace()
    #model=DenseLSTM(NUM_CLASS)
    #model=VGGLSTM(NUM_CLASS)
    #model=DenseCNN(NUM_CLASS)
    #model=VGGFC(NUM_CLASS)
    model=ResNetLSTM(NUM_CLASS)
    if os.path.exists(MODEL_PATH)==False:
        os.makedirs(MODEL_PATH)
    if os.path.exists(PATH+DICTIONARY_NAME)==False:
        logging.error("can't find the dictionary")
        pdb.set_trace()
    with open(PATH+DICTIONARY_NAME,'r') as f:
        dictionary=json.load(f)
    if pretrain==True:
        model.load_state_dict(torch.load(MODEL_PATH+MODEL_NAME,map_location=DEVICE))
    model.to(DEVICE).train()
    model.register_backward_hook(backward_hook)#transforms.Resize((32,400))
    dataset=ICDARRecTs_2DataSet(IMAGE_PATH,dictionary,BATCH_SIZE,img_transform=transforms.Compose([transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.3),
                                                                                        transforms.ToTensor(),
                                                                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,drop_last=False)#collate_fn=dataset.collate
    #optimizer=Adam(model.parameters(),lr=LR,betas=(0.9,0.999),weight_decay=0)
    optimizer=Adadelta(model.parameters(),lr=0.01,rho=0.9,weight_decay=0)
    criterion=CTCLoss(blank=0)
    length=len(dataloader)
    max_accuracy=0
    if os.path.exists('max_accuracy.txt')==True:
        with open('max_accuracy.txt','r') as f:
            max_accuracy=float(f.read())
    for epoch in range(EPOCH):
        epoch_time=datetime.now()
        epoch_correct=0
        epoch_loss=0
        min_loss=100
        for step,data in enumerate(dataloader):
            step_time=datetime.now()
            imgs,names,label_size,img_name=data
            #print(names,label_size)
            logging.debug("imgs' size:{}".format(imgs.size()))
            imgs=Variable(imgs,requires_grad=True).to(DEVICE)
            label,batch_label=dataset.transform_label(batch_name=names)
            label=Variable(label).to(DEVICE)
            label_size=Variable(label_size).to(DEVICE)
            preds=model(imgs)
            logging.debug("preds size:{}".format(preds.size()))
            preds_size=Variable(torch.LongTensor([preds.size(0)]*BATCH_SIZE)).to(DEVICE)
            loss=criterion(preds,label,preds_size,label_size)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            if min_loss>loss.item():
                min_loss=loss.item()
                torch.save(model.state_dict(),MODEL_PATH+MODEL_NAME)
            num_same=if_same(preds.cpu().data,batch_label)
            epoch_correct+=num_same
            logging.debug("Epoch:{}|length:{}|step:{}|num_same:{}|loss:{:.4f}|min loss:{:.4f}".format(epoch,length,step,num_same,loss.item(),min_loss))
            logging.debug("the time of one step:{}".format(datetime.now()-step_time))
            if step%100==0:
                clear_output(wait=True)
        accuracy=epoch_correct/(length)*BATCH_SIZE
        if accuracy>max_accuracy:
            max_accuracy=accuracy
            with open('max_accuracy.txt','w') as f:
                f.write(str(max_accuracy))
            torch.save(model.state_dict(),MODEL_PATH+MODEL_NAME)
            torch.save(model.state_dict(),MODEL_PATH+'optimal'+str(max_accuracy)+MODEL_NAME)
        mean_loss=epoch_loss/length
        logging.info('Epoch:{}|accuracy:{}|mean loss:{}|the time of one epoch:{}|max accuracy:{}'.format(epoch,accuracy,mean_loss,datetime.now()-epoch_time,max_accuracy))
        with open('accuracy.txt','a+') as f:
            f.write('Epoch:{}|accuracy:{}|mean loss:{}|the time of one epoch:{}|max accuracy:{}\n'.format(epoch,accuracy,mean_loss,datetime.now()-epoch_time,max_accuracy))
def backward_hook(module,grad_input,grad_output):
    for g in grad_input:
        #print('g:{}'.format(g))
        g[g!=g]=0#replace all nan or inf in gradients to zero
def if_same(preds,batch_label):
    #print(batch_label)
    t,b,n_class=preds.size()
    preds=preds.permute(1,0,2)
    _,preds=preds.max(2)
    count=0
    def condense(pred):
        result=[]
        original_pred=[]
        for i,p in enumerate(pred):
            original_pred.append(p.item())
            if p!=0 and (not(i>0 and pred[i-1]==pred[i])):
                result.append(p.item())
        return result,original_pred
    for pred,label in zip(preds,batch_label):
        flag=0
        pred,original_pred=condense(pred)
        label,_=condense(label)
        if(len(pred)==len(label)):
            for i,p in enumerate(pred):
                if(p!=label[i]):
                    flag=1
                    break
        if(flag==0 and len(pred)==len(label)):
            count+=1
        """if(count==1):
            print('label:{}'.format(label))
            print('pred:{}'.format(pred))
            print('original pred:{}'.format(original_pred))"""
        print('label:{}'.format(label))
        print('pred:{}'.format(pred))
        if(len(pred)==0):
            pass
            #return (0,1)
    return count
if __name__=='__main__':
    train(PRETRAIN)
    """
    temp=PATH + DICTIONARY_NAME
    with open(temp,'r') as f:#train_ReCTS_019633.12.jpg,¡
        a=json.load(f)
    i=len(a)
    a['¡']=i
    print(len(a))
    with open(PATH+'1'+DICTIONARY_NAME,'w') as f:
        json.dump(a,f)
    #print(a.get('¡'))"""
