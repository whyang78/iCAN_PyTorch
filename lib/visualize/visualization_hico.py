from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负
import pandas as pd
import pickle
import json
import numpy as np
import cv2
import os
import sys

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        p = pickle.load(f)
    return p

def _mask(start,finish):
    mask=np.zeros(600)
    for i in range(start,finish+1):
        mask[i]=1
    return mask

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
hoi_path = r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\action_list.csv'
hois = pd.read_csv(hoi_path)

objs = list(hois['object'].unique())
hoi_loc = np.zeros((len(objs), 2)).astype(np.int32)
for obj in objs:
    locs = hois[hois.iloc[:, 1] == obj].index
    start, finish = min(locs), max(locs)
    hoi_loc[COCO_NAMES.index(obj)] = [start, finish]

image_id = 239  #1,239,774,798
# detection_path='./detection_result/hoi_hico_detection.pkl'
detection_path='./hico_temp/{}.pkl'.format(str(image_id)) #所有结果
Detection = load_pkl(detection_path)

cc = plt.get_cmap('hsv', lut=6)
dpi = 80

im_file = r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\images\test2015'+'\HICO_test2015_' + str(image_id).zfill(8) + '.jpg'
im_data = plt.imread(im_file)
height, width, nbands = im_data.shape
figsize = width / float(dpi), height / float(dpi)
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(im_data, interpolation='nearest')

HO_dic = {}
HO_set = set()
count = 0
action_count = -1
a=Detection[image_id]
ch={'hold_baseball_bat':'手持棒球棒','swing_baseball_bat':'挥舞棒球棒','wield_baseball_bat':'挥棒球棒',
    'hold_baseball_glove':'手握棒球手套','wear_baseball_glove':'戴着棒球手套'}
for ele in Detection[image_id]:

    score,h_score,o_score=ele[3],ele[4],ele[5]
    obj_class=int(ele[2])-1
    start,finish=hoi_loc[obj_class]
    mask=_mask(start,finish)
    score=score*h_score*o_score*mask
    print(obj_class,np.max(score))
    if np.any(score>0.1):
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

        action_ind=np.where(score>0.1)[0]
        for action in action_ind:
            action_count += 1

            ax.add_patch(
                plt.Rectangle((H_box[0], H_box[1]),
                              H_box[2] - H_box[0],
                              H_box[3] - H_box[1], fill=False,
                              edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
            )

            Verb,Obj=hois.iloc[action,2],hois.iloc[action,1]
            text = str(Verb)+'_'+str(Obj) + ', ' + "%.2f" % score[action]
            # v_o=str(Verb)+'_'+str(Obj)
            # if v_o in ch:
            #     text=ch[v_o]+', ' + "%.2f" % score[action]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 25,
                    text,
                    bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                    fontsize=10, color='white')

            ax.add_patch(
                plt.Rectangle((O_box[0], O_box[1]),
                              O_box[2] - O_box[0],
                              O_box[3] - O_box[1], fill=False,
                              edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=3)
            )
            ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

plt.show()



