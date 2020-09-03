from PIL import Image
import matplotlib.pyplot as plt
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


def plt_hico(Detection):
    HO_dic = {}
    HO_set = set()
    count = 0
    action_count = -1
    for ele in Detection[image_id]:

        score, h_score, o_score = ele[3], ele[4], ele[5]
        obj_class = int(ele[2]) - 1
        start, finish = hoi_loc[obj_class]
        mask = _mask(start, finish)
        score = score * h_score * o_score * mask
        print(obj_class, np.max(score))
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

                ax.add_patch(
                    plt.Rectangle((H_box[0], H_box[1]),
                                  H_box[2] - H_box[0],
                                  H_box[3] - H_box[1], fill=False,
                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=15)
                )

                Verb, Obj = hois.iloc[action, 2], hois.iloc[action, 1]
                text = str(Verb) + '_' + str(Obj) + ', ' + "%.2f" % score[action]

                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                        text,
                        bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                        fontsize=100, color='white')

                ax.add_patch(
                    plt.Rectangle((O_box[0], O_box[1]),
                                  O_box[2] - O_box[0],
                                  O_box[3] - O_box[1], fill=False,
                                  edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=15)
                )
                ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    plt.show()


def plt_vcoco(Detection):
    HO_dic = {}
    HO_set = set()
    count = 0
    action_count = -1
    for ele in Detection:
        if (ele['image_id'] == image_id):

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
                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                        text,
                        bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                        fontsize=100, color='white')

            if ele['stand'][4] > 0.5:
                action_count += 1
                show_H_flag = 1
                text = 'stand, ' + "%.2f" % ele['stand'][4]
                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                        text,
                        bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                        fontsize=100, color='white')

            if ele['run'][4] > 0.5:
                action_count += 1
                show_H_flag = 1
                text = 'run, ' + "%.2f" % ele['run'][4]
                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                        text,
                        bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                        fontsize=100, color='white')

            if ele['walk'][4] > 0.5:
                action_count += 1
                show_H_flag = 1
                text = 'walk, ' + "%.2f" % ele['walk'][4]
                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                        text,
                        bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                        fontsize=100, color='white')

            if show_H_flag == 1:
                ax.add_patch(
                    plt.Rectangle((H_box[0], H_box[1]),
                                  H_box[2] - H_box[0],
                                  H_box[3] - H_box[1], fill=False,
                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=15)
                )

            for action_key, action_value in ele.items():
                if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                    if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                        O_box = action_value[:4]

                        action_count += 1

                        if tuple(O_box) not in HO_set:
                            HO_dic[tuple(O_box)] = count
                            HO_set.add(tuple(O_box))
                            count += 1

                        ax.add_patch(
                            plt.Rectangle((H_box[0], H_box[1]),
                                          H_box[2] - H_box[0],
                                          H_box[3] - H_box[1], fill=False,
                                          edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=15)
                        )
                        text = '_'.join(action_key.split('_')[:-1]) + ', ' + "%.2f" % action_value[4]

                        ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 150,
                                text,
                                bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                                fontsize=100, color='white')

                        ax.add_patch(
                            plt.Rectangle((O_box[0], O_box[1]),
                                          O_box[2] - O_box[0],
                                          O_box[3] - O_box[1], fill=False,
                                          edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=15)
                        )
                        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    plt.show()

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

image_id = 1  #1,239,774,798
model='hico'  #hico or vcoco
detection_path='./detection_result/hoi_detection.pkl'
Detection = load_pkl(detection_path)

cc = plt.get_cmap('hsv', lut=6)
dpi = 80

im_file ='./demo_images/'+ str(image_id) + '.jpg'
im_data = plt.imread(im_file)
height, width, nbands = im_data.shape
figsize = width / float(dpi), height / float(dpi)
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(im_data, interpolation='nearest')


if model=='hico':
    plt_hico(Detection)
elif model=='vcoco':
    plt_vcoco(Detection)







