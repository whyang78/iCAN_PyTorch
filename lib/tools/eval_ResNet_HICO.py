import numpy as np
from scipy.io import loadmat
import pickle
import os
import argparse
import pandas as pd
import shutil
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='eval on HICO-DET dataset')
    parser.add_argument('--type', dest='type',
            help='Select dataset: train or test',
            default='test', type=str)
    parser.add_argument('--eval_mode', dest='eval_mode',
                        help='eval mode: default or known_object ',
                        default='known_object', type=str)
    parser.add_argument('--min_overlap', dest='min_ov',
            help='min overlap',
            default=0.5, type=float)

    args = parser.parse_args()
    return args

def get_gt_all(matdata,gt_all):
    data=matdata.copy()

    for data_id in range(len(data)):
        temp_data=data[data_id]
        img_name=temp_data['filename']
        img_id=int(img_name[-9:-4])

        data_hois=temp_data['hoi']
        if data_hois.ndim==0:
            data_hois=data_hois[np.newaxis]

        for hoi_id in range(len(data_hois)):
            temp_hoi=data_hois[hoi_id]
            img_action=temp_hoi['id'] - 1
            img_invis=temp_hoi['invis']
            if img_invis==1:   #不可见则直接跳过此hoi
                continue

            if temp_hoi['connection'].ndim==1:
                temp_hoi['connection']=temp_hoi['connection'][np.newaxis]
            if temp_hoi['bboxhuman'].ndim==0:
                temp_hoi['bboxhuman']=temp_hoi['bboxhuman'][np.newaxis]
            if temp_hoi['bboxobject'].ndim==0:
                temp_hoi['bboxobject']=temp_hoi['bboxobject'][np.newaxis]

            for connect_id in range(len(temp_hoi['connection'])):
                if len(temp_hoi['connection'][connect_id])<2:
                    continue

                human_bbox_id=temp_hoi['connection'][connect_id][0]-1
                object_bbox_id=temp_hoi['connection'][connect_id][1]-1
                human_bbox=temp_hoi['bboxhuman'][human_bbox_id]
                object_bbox=temp_hoi['bboxobject'][object_bbox_id]

                human_bbox = [int(human_bbox['x1'])-1,
                              int(human_bbox['y1'])-1,
                              int(human_bbox['x2'])-1,
                              int(human_bbox['y2'])-1]

                object_bbox = [int(object_bbox['x1'])-1,
                            int(object_bbox['y1'])-1,
                            int(object_bbox['x2'])-1,
                            int(object_bbox['y2'])-1]
                gt_all[img_action].setdefault(img_id,[]).append(human_bbox+object_bbox)


def get_ious(det,gt):
    x1 = det[0]
    y1 = det[1]
    x2 = det[2]
    y2 = det[3]
    det_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    gt_area=(gt[:,2]-gt[:,0]+1)*(gt[:,3]-gt[:,1]+1)

    x1_max = np.maximum(x1, gt[:,0])
    y1_max = np.maximum(y1, gt[:,1])
    x2_min = np.minimum(x2, gt[:,2])
    y2_min = np.minimum(y2, gt[:,3])

    w = np.maximum(0, x2_min - x1_max + 1)
    h = np.maximum(0, y2_min - y1_max + 1)
    overlap = w * h
    ious = overlap / (det_area+gt_area-overlap)
    return ious


def cal_ap_rec_prec(detection_id,detection_bbox,detection_conf,gt,min_overlap):
    gt_={}
    npos=0
    for key,value in gt.items():
        value=np.array(value)
        npos += float(value.shape[0])

        det=np.zeros((value.shape[0],1))
        value=np.concatenate((value,det),axis=1)
        gt_[key]=value

    si=np.argsort(-1*detection_conf,axis=0).squeeze(axis=1)
    detection_id=detection_id[si,:]
    detection_bbox=detection_bbox[si,:]

    total=len(detection_conf)
    tp=np.zeros((total,1))
    fp=np.zeros((total,1))
    for i in range(total):
        img_id=int(detection_id[i][0])
        h_box=detection_bbox[i,0:4]
        o_box=detection_bbox[i,4:]

        iou_max=-np.inf
        if img_id in gt_.keys():
            gt_det=gt_[img_id][:,-1]
            gt_h_box=gt_[img_id][:,0:4]
            gt_o_box=gt_[img_id][:,4:-1]
            h_ious=get_ious(h_box,gt_h_box)
            o_ious=get_ious(o_box,gt_o_box)
            min_ious=np.minimum(h_ious,o_ious)

            iou_max_ind=np.argmax(min_ious)
            iou_max =min_ious[iou_max_ind]
        if iou_max>=min_overlap:
            if gt_det[iou_max_ind]==0:
                tp[i]=1
                gt_[img_id][iou_max_ind,-1]=1
            else:
                fp[i]=1  # multi detect
        else:
            fp[i]=1  # can't need min_ov or detect not in gt

    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    prec=tp/(fp+tp)
    rec=tp/npos

    ap = 0
    for t in np.linspace(0,1,11):
        try:
            p = max(prec[rec >= t])
        except ValueError:
            p=0
        ap = ap + p / 11.0

    return prec,rec,ap

#'__background__', 0
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
    'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe',  'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball',
    'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed',  'dining_table',
    'toilet',  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
]

if __name__ == '__main__':
    args=parse_args()
    anno_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\anno.mat'
    anno_box_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\anno_bbox.mat'
    tranform_detection = './detection_result/new'

    anno=loadmat(anno_path, squeeze_me=True)
    anno_box=loadmat(anno_box_path, squeeze_me=True)
    if args.type=='train':
        gt_bbox = anno_box['bbox_train']
        list_im = anno['list_train']
        anno_im = anno['anno_train']
    elif args.type=='test':
        gt_bbox=anno_box['bbox_test']
        list_im=anno['list_test']
        anno_im=anno['anno_test']

    list_im=list_im.tolist()
    images_id = list(map(lambda x: int(x[-9:-4]), list_im))

    hoi_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\action_list.csv'
    hois=pd.read_csv(hoi_path)
    objs=list(hois['object'].unique())
    hoi_loc=np.zeros((len(objs),2)).astype(np.int32)
    for obj in objs:
        locs=hois[hois.iloc[:,1]==obj].index
        start,finish=min(locs),max(locs)
        hoi_loc[COCO_NAMES.index(obj)]=[start,finish]

    hoi_num=600
    gt_all={i:{} for i in range(hoi_num)}
    get_gt_all(gt_bbox,gt_all)
    with open('./gt_all.pkl','wb') as f:
        pickle.dump(gt_all,f)

    AP=np.zeros((hoi_num,1))
    REC=np.zeros((hoi_num,1))
    for hoi in range(hoi_num):
        obj=hois.iloc[hoi,1]
        obj_index=COCO_NAMES.index(obj)

        detection_path=os.path.join(tranform_detection,'detections_{:02d}.pkl'.format(obj_index+1))
        with open(detection_path,'rb') as f:
            detection=pickle.load(f)

        hoi_id=hoi-hoi_loc[obj_index][0]
        detection=detection[hoi_id]

        detection_id=np.zeros((0,1))
        detection_bbox=np.zeros((0,8))
        detection_conf=np.zeros((0,1))
        for key,value in detection.items():
            val=np.array(value)
            bboxs=val[:,0:-1].reshape(-1,8)
            confs=val[:,-1].reshape(-1,1)
            image_id=np.array([key]).repeat(val.shape[0]).reshape(val.shape[0],1)

            detection_id=np.concatenate((detection_id,image_id),axis=0)
            detection_bbox=np.concatenate((detection_bbox,bboxs),axis=0)
            detection_conf=np.concatenate((detection_conf,confs),axis=0)

        gt=gt_all[hoi]
        if args.eval_mode=='default':
            pass
        elif args.eval_mode=='known_object':
            start,finish=hoi_loc[obj_index]
            keep_index=np.where(np.any(anno_im[start:finish+1,:]==1,0))[0]
            keep_id=[images_id[i] for i in list(keep_index)]

            keep_id=np.array(keep_id).reshape(-1,1)
            keep=np.where(np.in1d(detection_id,keep_id))[0]
            detection_id=detection_id[keep,:]
            detection_bbox=detection_bbox[keep,:]
            detection_conf=detection_conf[keep,:]

        prec,rec,ap=cal_ap_rec_prec(detection_id,detection_bbox,detection_conf,gt,args.min_ov)
        AP[hoi]=ap
        # if not rec:
        #     REC[hoi]=rec[-1]

    eval_result_path = './eval_result'
    if os.path.exists(eval_result_path):
        shutil.rmtree(eval_result_path)
    os.makedirs(eval_result_path)
    ap_save_path=os.path.join(eval_result_path,'det_ap_{}.pkl'.format(args.eval_mode))
    with open(ap_save_path,'wb') as f:
        pickle.dump(AP,f)

    inst_total=np.zeros((600,1))
    train=pd.read_csv(r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\anno_box_train.csv')
    for i in range(600):
        inst_total[i]+=list(train['action_id']).count(i+1)
    rare_ind=np.where(inst_total<10)[0]
    non_rare_ind=np.where(inst_total>=10)[0]

    map_full=np.mean(AP)
    map_rare=np.mean(AP[rare_ind])
    map_non_rare=np.mean(AP[non_rare_ind])

