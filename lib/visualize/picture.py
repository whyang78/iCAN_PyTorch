import os
import sys
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\process')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\networks')
sys.path.append(r'C:\Users\asus\Desktop\iCAN\lib\tools')
import cv2
import torchvision
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
import pickle
import glob
import argparse
import shutil
import json
import scipy.io as sio
from config import cfg
from networks import VCOCO_model,VCOCO_model_early,HICO_model
from ult import apply_prior

#80
COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign','parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow','elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella','handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball','kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket','bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza','donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book','clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
#81
CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush']
#91
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign','parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow','elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A','handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball','kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket','bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza','donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table','N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book','clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def parse_args():
    parser = argparse.ArgumentParser(description='Test an iCAN ')
    parser.add_argument('--model', dest='model',
            help='Select model:iCAN_ResNet50_HICO,iCAN_ResNet50_VCOCO',
            default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag',
            default=3, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.4, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)
    # parser.add_argument('--img_dir', dest='img_dir',
    #         help='Please specify the img folder',
    #         default='./demo_images', type=str)
    # parser.add_argument('--Demo_RCNN', dest='Demo_RCNN',
    #         help='The object detection .pkl file',
    #         default='./detection_result/', type=str)
    # parser.add_argument('--HOI_Detection', dest='HOI_Detection',
    #         help='Where to save the final HOI_Detection',
    #         default='./detection_result/', type=str)
    args = parser.parse_args()
    return args

def nms(bboxes,labels,thresh=0.3):
    x1=bboxes[:,0]
    y1=bboxes[:,1]
    x2=bboxes[:,2]
    y2=bboxes[:,3]
    areas=(x2-x1+1)*(y2-y1+1)

    keep=[]
    order=np.arange(len(bboxes))
    while len(order)>0:
        i=order[0]
        keep.append(i)

        x1_max=np.maximum(x1[i],x1[order[1:]])
        y1_max=np.maximum(y1[i],y1[order[1:]])
        x2_min=np.minimum(x2[i],x2[order[1:]])
        y2_min=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0,x2_min-x1_max+1)
        h=np.maximum(0,y2_min-y1_max+1)
        overlap=w*h
        ious=overlap/(areas[i]+areas[order[1:]]-overlap)

        keep_ind=np.where((ious<=thresh) | (labels[i]!=labels[order[1:]]))[0]
        order=order[keep_ind+1]
    return keep

def c91_to_c81(labels):
    label_id=[]
    for l in labels:
        label_id.append(CLASSES.index(l))
    return label_id

def save_pkl(pkl_path,data):
    with open(pkl_path,'wb') as f:
        pickle.dump(data,f)

def get_predection(image_path,model,threshold):
    img=Image.open(image_path)
    tf=transforms.ToTensor()
    img=tf(img).to(device)
    pred=model([img])
    bboxes=pred[0]['boxes'].detach().cpu().numpy()
    labels=pred[0]['labels'].cpu().numpy()
    scores=pred[0]['scores'].detach().cpu().numpy()
    keep=nms(bboxes,labels)
    nms_bboxes=bboxes[keep]
    nms_labels=labels[keep]
    nms_scores=scores[keep]

    pred_labels=[COCO_INSTANCE_CATEGORY_NAMES[label] for label in list(nms_labels)]

    pred_scores=nms_scores[nms_scores>threshold]
    pred_bboxes=nms_bboxes[:len(pred_scores)]
    pred_labels=pred_labels[:len(pred_scores)]
    labels_id=c91_to_c81(pred_labels)

    return pred_bboxes,labels_id,pred_scores

def object_detection(image_path,model,detection,threshold=0.5):
    pred_bboxes, pred_labels,pred_scores=get_predection(image_path,model,threshold)

    image_name=int(image_path.split(os.sep)[-1].split('.')[0].split('_')[-1])
    temp=[]
    for i,bbox in enumerate(pred_bboxes):
        inst_temp=[image_name]
        if pred_labels[i]==1:
            inst_temp.append('Human')
        else:
            inst_temp.append('Object')
        inst_temp.append(np.array(bbox))
        inst_temp.append(np.nan)
        inst_temp.append(pred_labels[i])
        inst_temp.append(np.array(pred_scores[i]))
        temp.append(inst_temp)
    detection[image_name]=temp

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def load_json(json_path):
    with open(json_path,'r') as f:
        q1=json.load(f)
    return q1

def process_image(image_path):

    image=cv2.imread(image_path).astype(np.float32)
    image-=cfg.PIXEL_MEANS
    image=torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
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

def im_detect(net, image_id,image_path, detection_result, prior_mask,action_dict_inv,object_thres, human_thres, prior_flag,detection, device):
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

def im_detect_hico(net,image_id,image_path,detection_result,object_thres,human_thres,detection,device):
    all_info=[]
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

def get_prediction(net,detection_result,img_path,prior_mask,action_dict_inv,model,object_thres, human_thres, prior_flag,device):
    net.eval()
    if model=='iCAN_ResNet50_VCOCO':
        print(1)
        detection = []
        image_id=int(img_path.split(os.sep)[-1].split('.')[0].split('_')[-1])
        im_detect(net, image_id,img_path, detection_result, prior_mask,action_dict_inv,object_thres, human_thres, prior_flag,detection, device)
    elif model=='iCAN_ResNet50_HICO':
        print(2)
        detection = {}
        image_id=int(img_path[-9:-4])
        im_detect_hico(net, image_id, img_path,detection_result, object_thres, human_thres, detection, device)
    return detection


def _mask(start,finish):
    mask=np.zeros(600)
    for i in range(start,finish+1):
        mask[i]=1
    return mask


def end_pt(start_pt, text, font_size, count):
    y2 = start_pt[1] + 30
    x2 = start_pt[0] + 100

    return (int(x2), int(y2))

def create_text(img, ho_dic,text, shape, count, H_box,O_box):
    font_size = 0.5
    x1, y1, x2, y2 = H_box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    start_pt = (x1 + 5, y1 + 5 + 25 * (count - 1))

    # cv2.rectangle(img, start_pt, end_pt(start_pt, text, font_size, count), tuple(map(int,cc[ho_dic[tuple(O_box)]])), thickness=-1)
    cv2.putText(
        img, text, (x1 + 10, y1 - 5 + (count * 25)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,tuple(map(int,cc[ho_dic[tuple(O_box)]])) , 1,
        cv2.LINE_AA
    )

def colormap(n=256):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])
    return cmap

def create_bbox(img, ho_dic,box, count):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color=tuple(map(int,cc[ho_dic[tuple(box)]]))

    cv2.line(img, (x1, y1), (x1, y2),color , 1)
    cv2.line(img, (x1, y2), (x2, y2), color, 1)
    cv2.line(img, (x2, y2), (x2, y1), color, 1)
    cv2.line(img, (x2, y1), (x1, y1),color, 1)


def print_image_vcoco(Detection, im_data):
    img_shape = list(im_data.shape)
    new_shape = list(im_data.shape)
    new_img = np.zeros(tuple(new_shape), np.uint8)
    new_img.fill(255)

    new_img[:img_shape[0], :img_shape[1]] = im_data

    HO_dic = {}
    HO_set = set()
    count = 0
    action_count = -1
    for ele in Detection:
        H_box = ele['person_box']

        if tuple(H_box) not in HO_set:
            HO_dic[tuple(H_box)] = count
            HO_set.add(tuple(H_box))
            count += 1
            action_count = -1

        show_H_flag = 0

        if ele['smile'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'smile, ' + "%.2f" % ele['smile'][4]
            create_text(new_img, HO_dic, text, img_shape, action_count, H_box, H_box)

        if ele['stand'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'stand, ' + "%.2f" % ele['stand'][4]
            create_text(new_img, HO_dic, text, img_shape, action_count, H_box, H_box)

        if ele['run'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'run, ' + "%.2f" % ele['run'][4]
            create_text(new_img, HO_dic, text, img_shape, action_count, H_box, H_box)

        if ele['walk'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'walk, ' + "%.2f" % ele['walk'][4]
            create_text(new_img, HO_dic, text, img_shape, action_count, H_box, H_box)

        if show_H_flag == 1:
            create_bbox(new_img,HO_dic,H_box, action_count)

        for action_key, action_value in ele.items():
            if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                if (not np.isnan(action_value[0])) and (action_value[4] > 0.05):
                    O_box = action_value[:4]

                    action_count += 1

                    if tuple(O_box) not in HO_set:
                        HO_dic[tuple(O_box)] = count
                        HO_set.add(tuple(O_box))
                        count += 1

                    create_bbox(new_img,HO_dic,H_box, action_count)
                    text = '_'.join(action_key.split('_')[:-1]) + ', ' + "%.2f" % action_value[4]

                    create_text(new_img, HO_dic,text, img_shape, action_count, H_box,O_box)
                    create_bbox(new_img,HO_dic, O_box, action_count)
    return new_img

def print_image_hico(Detection, im_data,img_id):
    img_shape = list(im_data.shape)
    new_shape = list(im_data.shape)
    new_img = np.zeros(tuple(new_shape), np.uint8)
    new_img.fill(255)

    new_img[:img_shape[0], :img_shape[1]] = im_data

    HO_dic = {}
    HO_set = set()
    count = 0
    action_count = -1
    for ele in Detection[img_id]:
        score, h_score, o_score = ele[3], ele[4], ele[5]
        obj_class = int(ele[2]) - 1
        start, finish = hoi_loc[obj_class]
        mask = _mask(start, finish)
        score = score * h_score * o_score * mask

        if np.any(score > 0.1):
            H_box = ele[0]
            if tuple(H_box) not in HO_set:
                HO_dic[tuple(H_box)] = count
                HO_set.add(tuple(H_box))
                count += 1
                action_count = -1

            O_box = ele[1]
            if tuple(O_box) not in HO_set:
                HO_dic[tuple(O_box)] = count
                HO_set.add(tuple(O_box))
                count += 1

            action_ind = np.where(score > 0.1)[0]
            for action in action_ind:
                action_count += 1

                create_bbox(new_img,HO_dic,H_box, action_count)
                Verb, Obj = hois.iloc[action, 2], hois.iloc[action, 1]
                text = str(Verb) + '_' + str(Obj) + ', ' + "%.2f" % score[action]

                create_text(new_img, HO_dic,text, img_shape, action_count, H_box,O_box)
                create_bbox(new_img,HO_dic, O_box, action_count)
    return new_img

if __name__ == '__main__':
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    args = parse_args()

    mask_path=r'C:\Users\asus\Desktop\iCAN\prior_mask.pkl'
    prior_mask=load_pkl(mask_path)
    action_path=r'C:\Users\asus\Desktop\iCAN\action_index.json'
    action_dict=load_json(action_path)
    action_dict_inv={v:k for k,v in action_dict.items()}

    device=torch.device('cpu')
    if args.model == 'iCAN_ResNet50_VCOCO':
        net = VCOCO_model()
        weight_path = './models/iter_10.pth'
    # if args.model == 'iCAN_ResNet50_VCOCO_Early':
    #     net=VCOCO_model_early()
    if args.model ==  'iCAN_ResNet50_HICO':
        net = HICO_model()
        weight_path = './models/iter_10_hico.pth'
    weight = torch.load(weight_path, map_location=device)
    net.load_state_dict(weight['model_state_dicts'])
    net.to(device)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    hoi_path = r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\action_list.csv'
    hois = pd.read_csv(hoi_path)

    objs = list(hois['object'].unique())
    hoi_loc = np.zeros((len(objs), 2)).astype(np.int32)
    for obj in objs:
        locs = hois[hois.iloc[:, 1] == obj].index
        start, finish = min(locs), max(locs)
        hoi_loc[COCO_NAMES.index(obj)] = [start, finish]

    images_folder='./hico_images'
    cc = colormap().astype('uint8')
    for p in glob.iglob(images_folder+'/*.jpg'):
        print(p)
        obj_detection={}
        object_detection(p,model,obj_detection)
        hoi_detection=get_prediction(net, obj_detection, p, prior_mask, action_dict_inv, args.model,
                       args.object_thres, args.human_thres, args.prior_flag, device)

        img_data=cv2.imread(p)
        if args.model == 'iCAN_ResNet50_VCOCO':
            frame=print_image_vcoco(hoi_detection,img_data)
        if args.model == 'iCAN_ResNet50_HICO':
            image_id = int(p[-9:-4])
            frame=print_image_hico(hoi_detection,img_data,image_id)
        cv2.imwrite('./1.jpg',frame)
        break