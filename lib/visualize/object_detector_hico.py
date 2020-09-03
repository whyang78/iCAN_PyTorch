import os
import sys
import cv2
import torchvision
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import random
import numpy as np
import pickle
import glob

#81
CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush']
#91
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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

def get_predection(image_path,threshold):
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

def object_detection(image_path,detection,threshold=0.5):
    pred_bboxes, pred_labels,pred_scores=get_predection(image_path,threshold)

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


if __name__ == '__main__':
    img_folder='./hico_images'
    output_file='./detection_result/hico_detection.pkl'
    # with open(output_file, 'rb') as f:
    #     p = pickle.load(f)
    device=torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    detection={}
    for p in glob.iglob(img_folder+'/*.jpg'):

        object_detection(p,detection)

    save_pkl(output_file,detection)
