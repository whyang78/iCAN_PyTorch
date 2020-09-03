import sys
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\process')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\networks')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\tools')

import torch
import numpy as np
import pickle
import argparse
import os
import random
from torchvision import transforms
from PIL import Image
import cv2
import glob
import shutil
import scipy.io as sio
from config import cfg
from networks import HICO_model

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on HICO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.8, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.6, type=float)

    args = parser.parse_args()
    return args

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def save_pkl(pkl_path,data):
    with open(pkl_path,'wb') as f:
        pickle.dump(data,f)

def process_image(image_path):
    # image=Image.open(image_path).convert('RGB')
    # tfs=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # image=tfs(image).float().unsqueeze(0)

    # type2
    image=cv2.imread(image_path).astype(np.float32)
    image-=cfg.PIXEL_MEANS
    image=torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
    return image

def get_sp_pattern(human_box,object_box):
    sp_size=64
    new_x1=min(human_box[0],object_box[0])
    new_y1=min(human_box[1],object_box[1])

    hbox=np.array([human_box[0]-new_x1,human_box[1]-new_y1,human_box[2]-new_x1,human_box[3]-new_y1]).astype(np.float64)
    obox=np.array([object_box[0]-new_x1,object_box[1]-new_y1,object_box[2]-new_x1,object_box[3]-new_y1]).astype(np.float64)
    sp_h=max(hbox[3],obox[3])-min(hbox[1],obox[1])+1
    sp_w=max(hbox[2],obox[2])-min(hbox[0],obox[0])+1

    factor=min(sp_size/sp_h,sp_size/sp_w)
    h_pad=(sp_size-sp_h*factor)/2.0
    w_pad=(sp_size-sp_w*factor)/2.0

    sp=np.zeros((1,2,sp_size,sp_size))
    sp[0, 0, int(hbox[1]*factor+h_pad):min(int(hbox[3]*factor+h_pad+factor),sp_size),
             int(hbox[0]*factor+w_pad):min(int(hbox[2]*factor+w_pad+factor),sp_size)]=1
    sp[0, 1, int(obox[1]*factor+h_pad):min(int(obox[3]*factor+h_pad+factor),sp_size),
             int(obox[0]*factor+w_pad):min(int(obox[2]*factor+w_pad+factor),sp_size)]=1
    return sp

def save_HICO(HICO, HICO_dir, classid, begin, finish):
    all_boxes = []
    for i in range(finish - begin + 1):
        total = []
        score = []
        for key, value in HICO.items():
            for element in value:
                if element[2] == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))  # action id (0-599)
                    temp.append(element[3][begin - 1 + i] * element[4] * element[5])
                    total.append(temp)
                    score.append(element[3][begin - 1 + i] * element[4] * element[5])

        idx = np.argsort(score)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])
    savefile = os.path.join(HICO_dir ,'detections_' + str(classid).zfill(2) + '.mat')
    sio.savemat(savefile, {'all_boxes': all_boxes})

def im_detect(net,image_id,detection_result,object_thres,human_thres,detection,device):
    all_info=[]
    test_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\images\test2015'
    image_path = os.path.join(test_path, 'HICO_test2015_' + str(image_id).zfill(8) + '.jpg')
    image = process_image(image_path)

    num_pos=1
    for human in detection_result[image_id]:
        if np.max(human[5])>human_thres and human[1]=='Human':
            h_box=np.array([ 0,human[2][0],human[2][1],human[2][2],human[2][3] ]).reshape(1,5)
            for object in detection_result[image_id]:
                if np.max(object[5])>object_thres and not np.all(human[2]==object[2]):
                    o_box=np.array([0,object[2][0],object[2][1],object[2][2],object[2][3]]).reshape(1,5)
                    sp=get_sp_pattern(human[2],object[2])
                    h_box_new = torch.from_numpy(h_box).float()
                    o_box_new = torch.from_numpy(o_box).float()
                    sp_new = torch.from_numpy(sp).float()

                    with torch.no_grad():
                        h_score, o_score, sp_score ,a , b= net(image.to(device), sp_new.to(device),h_box_new.to(device),
                                                            o_box_new.to(device), num_pos )
                        score = sp_score * ( h_score + o_score )
                        score = score.cpu().data.float().numpy()

                    temp = []
                    temp.append(human[2])  # Human box
                    temp.append(object[2])  # Object box
                    temp.append(object[4])  # Object class
                    temp.append(score[0])  # Score
                    temp.append(human[5])  # Human score
                    temp.append(object[5])  # Object score
                    all_info.append(temp)

    detection[image_id]=all_info

def get_prediction(net,detection_result,output_dir,object_thres,human_thres,device):
    # detection={}

    test_image_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\images\test2015\*.jpg'
    path=list(sorted(glob.glob(test_image_path)))
    for image in path:
        detection = {}
        image_id=int(image[-9:-4])
        print('current_image', image_id)
        im_detect(net,image_id,detection_result,object_thres,human_thres,detection,device)
        save_pkl('./temp/{}.pkl'.format(str(image_id)), detection)
    #save_pkl(output_dir,detection)


def generate_hico_detection(output_file,HICO_dir):
    with open(output_file,'rb') as f:
        HICO=pickle.load(f)

    save_HICO(HICO, HICO_dir, 1, 161, 170)  # 1 person
    save_HICO(HICO, HICO_dir, 2, 11, 24)  # 2 bicycle
    save_HICO(HICO, HICO_dir, 3, 66, 76)  # 3 car
    save_HICO(HICO, HICO_dir, 4, 147, 160)  # 4 motorcycle
    save_HICO(HICO, HICO_dir, 5, 1, 10)  # 5 airplane
    save_HICO(HICO, HICO_dir, 6, 55, 65)  # 6 bus
    save_HICO(HICO, HICO_dir, 7, 187, 194)  # 7 train
    save_HICO(HICO, HICO_dir, 8, 568, 576)  # 8 truck
    save_HICO(HICO, HICO_dir, 9, 32, 46)  # 9 boat
    save_HICO(HICO, HICO_dir, 10, 563, 567)  # 10 traffic light
    save_HICO(HICO, HICO_dir, 11, 326, 330)  # 11 fire_hydrant
    save_HICO(HICO, HICO_dir, 12, 503, 506)  # 12 stop_sign
    save_HICO(HICO, HICO_dir, 13, 415, 418)  # 13 parking_meter
    save_HICO(HICO, HICO_dir, 14, 244, 247)  # 14 bench
    save_HICO(HICO, HICO_dir, 15, 25, 31)  # 15 bird
    save_HICO(HICO, HICO_dir, 16, 77, 86)  # 16 cat
    save_HICO(HICO, HICO_dir, 17, 112, 129)  # 17 dog
    save_HICO(HICO, HICO_dir, 18, 130, 146)  # 18 horse
    save_HICO(HICO, HICO_dir, 19, 175, 186)  # 19 sheep
    save_HICO(HICO, HICO_dir, 20, 97, 107)  # 20 cow
    save_HICO(HICO, HICO_dir, 21, 314, 325)  # 21 elephant
    save_HICO(HICO, HICO_dir, 22, 236, 239)  # 22 bear
    save_HICO(HICO, HICO_dir, 23, 596, 600)  # 23 zebra
    save_HICO(HICO, HICO_dir, 24, 343, 348)  # 24 giraffe
    save_HICO(HICO, HICO_dir, 25, 209, 214)  # 25 backpack
    save_HICO(HICO, HICO_dir, 26, 577, 584)  # 26 umbrella
    save_HICO(HICO, HICO_dir, 27, 353, 356)  # 27 handbag
    save_HICO(HICO, HICO_dir, 28, 539, 546)  # 28 tie
    save_HICO(HICO, HICO_dir, 29, 507, 516)  # 29 suitcase
    save_HICO(HICO, HICO_dir, 30, 337, 342)  # 30 Frisbee
    save_HICO(HICO, HICO_dir, 31, 464, 474)  # 31 skis
    save_HICO(HICO, HICO_dir, 32, 475, 483)  # 32 snowboard
    save_HICO(HICO, HICO_dir, 33, 489, 502)  # 33 sports_ball
    save_HICO(HICO, HICO_dir, 34, 369, 376)  # 34 kite
    save_HICO(HICO, HICO_dir, 35, 225, 232)  # 35 baseball_bat
    save_HICO(HICO, HICO_dir, 36, 233, 235)  # 36 baseball_glove
    save_HICO(HICO, HICO_dir, 37, 454, 463)  # 37 skateboard
    save_HICO(HICO, HICO_dir, 38, 517, 528)  # 38 surfboard
    save_HICO(HICO, HICO_dir, 39, 534, 538)  # 39 tennis_racket
    save_HICO(HICO, HICO_dir, 40, 47, 54)  # 40 bottle
    save_HICO(HICO, HICO_dir, 41, 589, 595)  # 41 wine_glass
    save_HICO(HICO, HICO_dir, 42, 296, 305)  # 42 cup
    save_HICO(HICO, HICO_dir, 43, 331, 336)  # 43 fork
    save_HICO(HICO, HICO_dir, 44, 377, 383)  # 44 knife
    save_HICO(HICO, HICO_dir, 45, 484, 488)  # 45 spoon
    save_HICO(HICO, HICO_dir, 46, 253, 257)  # 46 bowl
    save_HICO(HICO, HICO_dir, 47, 215, 224)  # 47 banana
    save_HICO(HICO, HICO_dir, 48, 199, 208)  # 48 apple
    save_HICO(HICO, HICO_dir, 49, 439, 445)  # 49 sandwich
    save_HICO(HICO, HICO_dir, 50, 398, 407)  # 50 orange
    save_HICO(HICO, HICO_dir, 51, 258, 264)  # 51 broccoli
    save_HICO(HICO, HICO_dir, 52, 274, 283)  # 52 carrot
    save_HICO(HICO, HICO_dir, 53, 357, 363)  # 53 hot_dog
    save_HICO(HICO, HICO_dir, 54, 419, 429)  # 54 pizza
    save_HICO(HICO, HICO_dir, 55, 306, 313)  # 55 donut
    save_HICO(HICO, HICO_dir, 56, 265, 273)  # 56 cake
    save_HICO(HICO, HICO_dir, 57, 87, 92)  # 57 chair
    save_HICO(HICO, HICO_dir, 58, 93, 96)  # 58 couch
    save_HICO(HICO, HICO_dir, 59, 171, 174)  # 59 potted_plant
    save_HICO(HICO, HICO_dir, 60, 240, 243)  # 60 bed
    save_HICO(HICO, HICO_dir, 61, 108, 111)  # 61 dining_table
    save_HICO(HICO, HICO_dir, 62, 551, 558)  # 62 toilet
    save_HICO(HICO, HICO_dir, 63, 195, 198)  # 63 TV
    save_HICO(HICO, HICO_dir, 64, 384, 389)  # 64 laptop
    save_HICO(HICO, HICO_dir, 65, 394, 397)  # 65 mouse
    save_HICO(HICO, HICO_dir, 66, 435, 438)  # 66 remote
    save_HICO(HICO, HICO_dir, 67, 364, 368)  # 67 keyboard
    save_HICO(HICO, HICO_dir, 68, 284, 290)  # 68 cell_phone
    save_HICO(HICO, HICO_dir, 69, 390, 393)  # 69 microwave
    save_HICO(HICO, HICO_dir, 70, 408, 414)  # 70 oven
    save_HICO(HICO, HICO_dir, 71, 547, 550)  # 71 toaster
    save_HICO(HICO, HICO_dir, 72, 450, 453)  # 72 sink
    save_HICO(HICO, HICO_dir, 73, 430, 434)  # 73 refrigerator
    save_HICO(HICO, HICO_dir, 74, 248, 252)  # 74 book
    save_HICO(HICO, HICO_dir, 75, 291, 295)  # 75 clock
    save_HICO(HICO, HICO_dir, 76, 585, 588)  # 76 vase
    save_HICO(HICO, HICO_dir, 77, 446, 449)  # 77 scissors
    save_HICO(HICO, HICO_dir, 78, 529, 533)  # 78 teddy_bear
    save_HICO(HICO, HICO_dir, 79, 349, 352)  # 79 hair_drier
    save_HICO(HICO, HICO_dir, 80, 559, 562)  # 80 toothbrush

def pred_demo(net,image_id):
    test_path = r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\images\test2015'
    image_path = os.path.join(test_path, 'HICO_test2015_' + str(image_id).zfill(8) + '.jpg')
    image = process_image(image_path)

    h_box= np.array([0, 226, 18,340,  210]).reshape(1, 5)
    o_box = np.array([0, 174, 65,393,  440]).reshape(1, 5)
    sp = get_sp_pattern([226, 18,340,  210],[174, 65,393,  440])

    h_box_new = torch.from_numpy(h_box).float()
    o_box_new = torch.from_numpy(o_box).float()
    sp_new = torch.from_numpy(sp).float()
    with torch.no_grad():
        h_score, o_score, sp_score ,a,b= net(image.to(device), sp_new.to(device), h_box_new.to(device),
                                     o_box_new.to(device),1)
    score = sp_score * (h_score + o_score)
    print(score)
    print(score[0][129:146])
    print(torch.topk(score[0][129:146],5))
    return score


if __name__ == '__main__':
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    arg=parse_args()

    # detection_path=r'C:\Users\asus\Desktop\iCAN\Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl'
    # detection_result=load_pkl(detection_path)

    # output_file='./'+str(arg.iteration)+'_'+arg.model+'.pkl'
    output_file='./iter_0.pkl'
    hico_dir='./detection_result'
    if os.path.exists(hico_dir):
        shutil.rmtree(hico_dir)
    os.makedirs(hico_dir)

    # device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device=torch.device('cpu')
    # weight_path=r'C:\Users\asus\Desktop\iCAN\test_result\4\iter_1180000.pth'
    # weight=torch.load(weight_path,map_location=torch.device('cpu'))
    # net = HICO_model()
    # net.load_state_dict(weight['model_state_dicts'])
    # net.to(device)
    # net.eval()
    # get_prediction(net, detection_result, output_file, arg.object_thres, arg.human_thres, device)
    generate_hico_detection(output_file,hico_dir)

