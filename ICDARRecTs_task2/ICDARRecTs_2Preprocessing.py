import json
import os
from PIL import Image
import sys
import cv2
import math
import numpy as np
import pysnooper
import pdb
sys.path.insert(0,'../')
from Logging import *
__all__=['rotate_crop']
def load_img(path):
    return Image.open(path).convert('RGB')
def rotate_crop(img,coordinate):
    if coordinate[2]!=coordinate[0] and coordinate[4]!=coordinate[6]:
        radian1=(coordinate[3]-coordinate[1])/(coordinate[2]-coordinate[0])#radian
        radian2=(coordinate[5]-coordinate[7])/(coordinate[4]-coordinate[6])
        logging.debug('radian1:{}|radian2:{}'.format(radian1,radian2))
        angle=(radian1+radian2)/2
        h,w=img.shape[0],img.shape[1]
    else:
        angle=0
    #print(h,w,img.shape)#(88, 93, 3)
    if angle>0:#anti-clockwise
        new_h=w*math.sin(angle)+h*math.cos(angle)
        new_w=w*math.cos(angle)+h*math.sin(angle)
    elif angle<0:#clockwise
        new_h=math.fabs(w*math.sin(-angle))+math.fabs(h*math.cos(angle))
        new_w=math.fabs(w*math.cos(angle))+math.fabs(h*math.sin(-angle))
    else:
        coordinate_0 = min(coordinate[0], coordinate[6])
        coordinate_1 = min(coordinate[1], coordinate[3])
        coordinate_4 = max(coordinate[4], coordinate[2])
        coordinate_5 = max(coordinate[5], coordinate[7])
        return img[coordinate_1:coordinate_5,coordinate_0:coordinate_4]
        #pass
    angle = angle * (180 / math.pi)
    rotate_matrix=cv2.getRotationMatrix2D((w/2,h/2),angle=angle,scale=1)
    logging.debug('rotate_matrix:{}'.format(rotate_matrix))
    rotate_matrix[0, 2] += (new_w - w) / 2
    rotate_matrix[1, 2] += (new_h - h) / 2
    img_rotated=cv2.warpAffine(img,M=rotate_matrix,dsize=(int(new_w),int(new_h)),borderValue=(0,0,0))
    [[coordinate[0]],[coordinate[1]]]=np.dot(rotate_matrix,np.array([[coordinate[0]],[coordinate[1]],[1]]))
    [[coordinate[2]],[coordinate[3]]]=np.dot(rotate_matrix,np.array([[coordinate[2]],[coordinate[3]],[1]]))
    [[coordinate[4]],[coordinate[5]]]=np.dot(rotate_matrix,np.array([[coordinate[4]],[coordinate[5]],[1]]))
    [[coordinate[6]],[coordinate[7]]]=np.dot(rotate_matrix,np.array([[coordinate[6]],[coordinate[7]],[1]]))
    logging.debug('coordinate:{}'.format(coordinate))
    #get max rectangle box
    coordinate_0=min(coordinate[0],coordinate[6])
    coordinate_1=min(coordinate[1],coordinate[3])
    coordinate_4=max(coordinate[4],coordinate[2])
    coordinate_5=max(coordinate[5],coordinate[7])
    img_crop_rotated=img_rotated[int(coordinate_1):int(coordinate_5),int(coordinate_0):int(coordinate_4)]
    #"""
    cv2.imshow('original',img)
    cv2.imshow('rotated',img_rotated)
    cv2.imshow('crop_rotated',img_crop_rotated)
    cv2.waitKey(0)
    #"""
    return img_crop_rotated
def crop_img():
    path = r'E:\Files\ICDAR2019RecTs\ReCTS'
    label_path = '\\gt\\'
    img_path = '\\img\\'
    label_list = os.listdir(path + label_path)
    save_path = path + '\\task2_cropped_img_less_30\\'
    save_path_unlabeled=path+'\\task2_cropped_img_unlabeled\\'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    if os.path.exists(save_path_unlabeled)==False:
        os.makedirs(save_path_unlabeled)
    length=len(label_list)
    file=open('label.txt','w',encoding='UTF-8')
    max_length=0
    for j,label in enumerate(label_list):
        img=load_img(path+img_path+label.split('.')[0]+'.jpg')
        with open(path+label_path+label,'r',encoding='UTF-8') as f:
            label_dict=json.load(f)
        k = 0
        for mess in label_dict['lines']:
            transcription=mess['transcription']
            logging.debug('label:{}|length:{}|j:{}|k:{}|transcription:{}'.format(label,length,j,k,transcription))
            coordinate = []
            if mess['points'][0]>mess['points'][2]:
                mess['points'][0],mess['points'][2]=mess['points'][2],mess['points'][0]
            if mess['points'][1]>mess['points'][7]:
                mess['points'][1],mess['points'][7]=mess['points'][7],mess['points'][1]
            coordinate.append(min(mess['points'][0],mess['points'][6]))
            if coordinate[0]<=0:
                continue
            coordinate.append(min(mess['points'][1],mess['points'][3]))
            coordinate.append(max(mess['points'][4],mess['points'][2]))
            coordinate.append(max(mess['points'][5],mess['points'][7]))
            #coordinate.extend(mess['points'][0:2])
            #coordinate.extend(mess['points'][4:6])
            logging.debug('coordinate:{}'.format(coordinate))
            #img.crop(coordinate).show()
            if len(transcription) > max_length:
                max_length=len(transcription)
            if transcription=='###':
                pass
                #img.crop(coordinate).save(save_path_unlabeled+label.split('.')[0]+'.'+str(k)+'.'+transcription+'.jpg')
            else:
                filename=label.split('.')[0]+'.'+str(k)+'.jpg'
                img.crop(coordinate).save(save_path+filename)
                if len(transcription)<30:
                    file.write(filename+','+transcription+'\n')
                #the longest of transcription:47
                logging.debug('filename:{}|transcription:{}|the length of transcription:{}|the longest of transcription:{}'.format(filename,transcription,len(transcription),max_length))
            k+=1
        logging.info('one step finished')
    file.close()
    logging.info('finished')
def get_dictionary():#len(dictionary)=4134
    path = r'E:\Files\ICDAR2019RecTs\ReCTS'
    label_path = '\\gt\\'
    label_list = os.listdir(path + label_path)
    length = len(label_list)
    max_length = 0
    dictionary={}
    file=open('label.txt','w',encoding='UTF-8')
    i=len(dictionary)+1
    length_dict={}
    #file_max_length=open('max_length.txt','a+',encoding='UTF-8')
    for j,label in enumerate(label_list):
        with open(path+label_path+label,'r',encoding='UTF-8') as f:
            label_dict=json.load(f)
        k=0
        for mess in label_dict['lines']:
            transcription = mess['transcription']
            if length_dict.__contains__(len(transcription))==False:
                length_dict[len(transcription)]=1
            else:
                length_dict[len(transcription)]+=1
            if len(transcription)>=30:
                pass
                #file_max_length.write(label+' '+transcription+'\n')
            if max_length<len(transcription):
                max_length=len(transcription)
                #file_max_length.write(label+' '+transcription+'\n')
            logging.debug('label:{}|length:{}|j:{}|k:{}|transcription:{}|max_length:{}'.format(label, length, j, k, transcription,max_length))
            if transcription!='###':
                if len(transcription)<30:
                    filename = label.split('.')[0] + '.' + str(k) + '.jpg'
                    file.write(filename + ',' + transcription + '\n')
                for s in transcription:
                    if dictionary.__contains__(s)==False:
                        dictionary[s]=i
                        i+=1
            k+=1
    #file_max_length.close()
    #with open('length_dict.txt','w') as f:
        #json.dump(length_dict,f)
    file.close()
    #dictionary_inv={v:k for k,v in dictionary.items()}
    #with open(path+'\\'+'RecTs2dictionary.json','w',encoding='UTF-8') as f:
        #json.dump(dictionary,f)
    #with open(path+'\\'+'RecTs2dictionary_inv.json','w',encoding='UTF-8') as f:
        #json.dump(dictionary_inv,f)
    logging.debug('finished the length of dictionary:{}'.format(len(dictionary)))#4134
def remove_labels(remove_length=30):
    print(remove_length)
    with open('label.txt','r',encoding='UTF-8') as f:
        labels=f.readlines()
    print(labels)
    for i in range(len(labels)):
        name_label = labels[i]
        #name_label = name_label.split(',')
        name=name_label[:23]
        label=name_label[24:]
        print(1)
        logging.warning('i:{}|name:{}|label:{}'.format(i,name,label))

        #name = name_label[0]
        #label = ''.join(name_label[1:])
        if len(label) == 1:
            label = ',\n'
if __name__=='__main__':
    #crop_img()
    from PIL import ImageDraw
    #get_dictionary()
    #remove_labels()
                                                                                                            #[1,66,66,1,68,92,4,149]
    path=r'E:\Files\ICDAR2019RecTs\ReCTS_test_part1\ReCTS_test_part1\Task2\img\test_ReCTS_task2_000002.jpg'#test_ReCTS_task2_000120.jpg'#test_ReCTS_task2_000700
    img=cv2.imread(path)
    #print(type(img))
    rotate_crop(img,[1,1,528,1,528,68,1,68])
    """
    import numpy as np
    img=load_img(path)
    img.show()
    #np.asarray()
    img_np=np.asarray(img)
    cv2_img=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)
    cv2.imshow('ss',cv2_img)
    cv2.waitKey(0)
    print(img_np.shape)
    print(img.size)
    #draw=ImageDraw.Draw(img)
    #draw.polygon([1,66,66,1,68,92,4,149],fill=None,outline=(255,0,0))
    print(type(img))
    """
    """
    @pysnooper.snoop()
    def te(a=1):
        b=a
        c=b*b
        return c
    te()"""
    #img.show()
    #temp=load_img(r'E:\Files\ICDAR2019RecTs\ReCTS\img\train_ReCTS_000432.jpg')
    #print(type(temp))
    #temp.crop([73, 200, 131, 454]).show()
