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
import json
import scipy.io as sio
from config import cfg
from networks import VCOCO_model,VCOCO_model_early
from ult import apply_prior

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=300000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_ResNet50_VCOCO', type=str)
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag',
            default=3, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.4, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)

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

def load_json(json_path):
    with open(json_path,'r') as f:
        q1=json.load(f)
    return q1

def process_image(image_path):
    # image=Image.open(image_path).convert('RGB')
    # tfs=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # image=tfs(image).float().unsqueeze(0)

    # type2
    image = cv2.imread(image_path).astype(np.float32)
    image -= cfg.PIXEL_MEANS
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def bbox_trans(human_box_ori, object_box_ori, ratio, size=64):
    human_box = human_box_ori.copy()
    object_box = object_box_ori.copy()

    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1

    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'

    # shift the top-left corner to (0,0)

    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1]

    if ratio == 'height':  # height is larger than width

        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
        human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
        object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

        # Need to shift horizontally
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        shift = size / 2 - (InteractionPattern[2] + 1) / 2
        human_box += [shift, 0, shift, 0]
        object_box += [shift, 0, shift, 0]

    else:  # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

        # Need to shift vertically
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2

        human_box = human_box + [0, shift, 0, shift]
        object_box = object_box + [0, shift, 0, shift]

    return np.round(human_box), np.round(object_box)


def get_sp_pattern(human_box, object_box):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((1, 2, 64, 64))
    Pattern[0, 0, int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1] = 1
    Pattern[0, 1, int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1] = 1

    return Pattern

def im_detect(net, image_id, detection_result, prior_mask,action_dict_inv,object_thres, human_thres, prior_flag,detection, device):
    test_path='vcoco_test'
    image_path = os.path.join(test_path, 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg')
    image = process_image(image_path)

    num_pos=1
    for human in detection_result[image_id]:
        if np.max(human[5])>human_thres and human[1]=='Human':
            h_box=np.array([ 0,human[2][0],human[2][1],human[2][2],human[2][3] ]).reshape(1,5)
            temp = {}
            temp['image_id'] = image_id
            temp['person_box'] = human[2]
            score_obj=np.empty((0,4+29),dtype=np.float32)
            for object in detection_result[image_id]:
                if np.max(object[5])>object_thres and not np.all(human[2]==object[2]):
                    o_box=np.array([0,object[2][0],object[2][1],object[2][2],object[2][3]]).reshape(1,5)
                    sp=get_sp_pattern(human[2],object[2])
                    h_box_new = torch.from_numpy(h_box).float()
                    o_box_new = torch.from_numpy(o_box).float()
                    sp_new = torch.from_numpy(sp).float()

                    with torch.no_grad():
                        if args.model == 'iCAN_ResNet50_VCOCO':
                            h_score, o_score, sp_score ,a , b= net(image.to(device), sp_new.to(device),h_box_new.to(device),
                                                                o_box_new.to(device), num_pos )
                            score = sp_score * ( h_score + o_score )
                            score = score.cpu().data.float().numpy()
                            h_score=h_score.cpu().data.float().numpy()
                        if args.model == 'iCAN_ResNet50_VCOCO_Early':
                            h_score, all_score, atten_h, atten_o=net(image.to(device), sp_new.to(device),h_box_new.to(device),
                                                                o_box_new.to(device), num_pos)
                            score = all_score.cpu().data.float().numpy()
                            h_score = h_score.cpu().data.float().numpy()

                    if prior_flag==1:
                        score=apply_prior(object,score)
                    if prior_flag==2:
                        score=score*prior_mask[:,object[4]].reshape(1,29)
                    if prior_flag==3:
                        score = apply_prior(object, score)
                        score = score * prior_mask[:, object[4]].reshape(1, 29)

                    temp_score_obj=np.concatenate((object[2].reshape(1,4),score.reshape(1,29)*np.max(object[5])),axis=1)
                    score_obj=np.concatenate((score_obj,temp_score_obj),axis=0)

            if score_obj.shape[0]==0:
                continue

            max_idx = np.argmax(score_obj, 0)[4:]
            # agent mAP
            for i in range(29):
                # walk, smile, run, stand
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    agent_name = action_dict_inv[i] + '_agent'
                    temp[agent_name] = np.max(human[5]) * h_score[0][i]
                    continue

                # cut
                if i == 2:
                    agent_name = 'cut_agent'
                    temp[agent_name] = np.max(human[5]) * max(score_obj[max_idx[2]][4 + 2],
                                                                 score_obj[max_idx[4]][4 + 4])
                    continue
                if i == 4:
                    continue

                # eat
                if i == 9:
                    agent_name = 'eat_agent'
                    temp[agent_name] = np.max(human[5]) * max(score_obj[max_idx[9]][4 + 9],
                                                                 score_obj[max_idx[16]][4 + 16])
                    continue
                if i == 16:
                    continue

                # hit
                if i == 19:
                    agent_name = 'hit_agent'
                    temp[agent_name] = np.max(human[5]) * max(score_obj[max_idx[19]][4 + 19],
                                                                 score_obj[max_idx[20]][4 + 20])
                    continue
                if i == 20:
                    continue

                # These 2 classes need to save manually because there is '_' in action name
                if i == 6:
                    agent_name = 'talk_on_phone_agent'
                    temp[agent_name] = np.max(human[5]) * score_obj[max_idx[i]][4 + i]
                    continue

                if i == 8:
                    agent_name = 'work_on_computer_agent'
                    temp[agent_name] = np.max(human[5]) * score_obj[max_idx[i]][4 + i]
                    continue

                # all the rest
                agent_name = action_dict_inv[i].split("_")[0] + '_agent'
                temp[agent_name] = np.max(human[5]) * score_obj[max_idx[i]][4 + i]

            # role mAP
            for i in range(29):
                # walk, smile, run, stand. Won't contribute to role mAP
                if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                    temp[action_dict_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(human[5]) * h_score[0][i])
                    continue

                # Impossible to perform this action
                if np.max(human[5]) * score_obj[max_idx[i]][4 + i] == 0:
                    temp[action_dict_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(human[5]) * score_obj[max_idx[i]][4 + i])

                # Action with >0 score
                else:
                    temp[action_dict_inv[i]] = np.append(score_obj[max_idx[i]][:4],
                                                       np.max(human[5]) * score_obj[max_idx[i]][4 + i])

            detection.append(temp)


def get_prediction(net,detection_result,output_file,prior_mask,action_dict_inv,object_thres, human_thres, prior_flag,device):
    # detection=[]

    test_image_path = r'C:\Users\asus\Desktop\iCAN\vcoco_test.ids'
    with open(test_image_path,'r') as f:
        all_test=f.readlines()
    net.eval()
    for line in all_test:
        detection = []
        image_id = int(line.strip())
        print('current_image', image_id)
        im_detect(net, image_id, detection_result, prior_mask,action_dict_inv,object_thres, human_thres, prior_flag,detection, device)
        save_pkl('./temp/{}.pkl'.format(str(image_id)), detection)
    # save_pkl(output_dir,detection)

if __name__ == '__main__':
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    args = parse_args()

    detection_path=r'C:\Users\asus\Desktop\iCAN\Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl'
    detection_result=load_pkl(detection_path)
    mask_path=r'C:\Users\asus\Desktop\iCAN\prior_mask.pkl'
    prior_mask=load_pkl(mask_path)
    action_path=r'C:\Users\asus\Desktop\iCAN\action_index.json'
    action_dict=load_json(action_path)
    action_dict_inv={v:k for k,v in action_dict.items()}

    output_file = './' + str(args.iteration) + '_' + args.model + '.pkl'

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_path='../iter_180000.pth'
    weight=torch.load(weight_path,map_location=device)
    if args.model == 'iCAN_ResNet50_VCOCO':
        net = VCOCO_model()
    if args.model == 'iCAN_ResNet50_VCOCO_Early':
        net=VCOCO_model_early()
    net.load_state_dict(weight)
    net.to(device)

    get_prediction(net,detection_result,output_file,prior_mask,action_dict_inv,args.object_thres, args.human_thres, args.prior_flag,device)