
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负
import pickle
import json
import numpy as np
import cv2
import os
import sys

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

detection_path='./detection_result/hoi_vcoco_detection.pkl'
# detection_path=r'C:\Users\asus\Desktop\iCAN\300000_iCAN_ResNet50_VCOCO.pkl'  #官方结果
Detection = load_pkl(detection_path)

image_id = 2890
cc = plt.get_cmap('hsv', lut=6)
dpi = 80

im_file = './vcoco_images/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
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

for ele in Detection:
    if (ele['image_id'] == image_id):
        action_count = -1

        H_box = ele['person_box']
        if tuple(H_box) not in HO_set:
            HO_dic[tuple(H_box)] = count
            HO_set.add(tuple(H_box))
            count += 1
        show_H_flag = 0

        if ele['smile'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'smile, ' + "%.2f" % ele['smile'][4]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                    text,
                    bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                    fontsize=10, color='white')

        if ele['stand'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'stand, ' + "%.2f" % ele['stand'][4]
            # text = '站立, ' + "%.2f" % ele['stand'][4]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                    text,
                    bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                    fontsize=10, color='white')

        if ele['run'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'run, ' + "%.2f" % ele['run'][4]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                    text,
                    bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                    fontsize=10, color='white')

        if ele['walk'][4] > 0.5:
            action_count += 1
            show_H_flag = 1
            text = 'walk, ' + "%.2f" % ele['walk'][4]
            ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                    text,
                    bbox=dict(facecolor=cc(HO_dic[tuple(H_box)])[:3], alpha=0.5),
                    fontsize=10, color='white')

        if show_H_flag == 1:
            ax.add_patch(
                plt.Rectangle((H_box[0], H_box[1]),
                              H_box[2] - H_box[0],
                              H_box[3] - H_box[1], fill=False,
                              edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
            )

        for action_key, action_value in ele.items():
            if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                if (not np.isnan(action_value[0])) and (action_value[4] > 0.05):
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
                                      edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
                    )
                    text = action_key.split('_')[0] + ', ' + "%.2f" % action_value[4]
                    # if action_key.split('_')[0]=='ski':
                    #     text = '滑雪' + ', ' + "%.2f" % action_value[4]

                    ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
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




