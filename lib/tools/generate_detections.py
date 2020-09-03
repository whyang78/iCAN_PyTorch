from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
import shutil
import sys
import pickle
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
    anno_bbox = loadmat(r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\anno.mat', squeeze_me=True)
    test_images=anno_bbox['list_test'].tolist()
    test_images_id=list(map(lambda x:int(x[-9:-4]),test_images))

    hoi_path=r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\action_list.csv'
    hois=pd.read_csv(hoi_path)
    objs=list(hois['object'].unique())
    hoi_loc=np.zeros((len(objs),2)).astype(np.int32)
    for obj in objs:
        locs=hois[hois.iloc[:,1]==obj].index
        start,finish=min(locs),max(locs)
        hoi_loc[COCO_NAMES.index(obj)]=[start,finish]

    detection_result='./detection_result'
    tranform_detection='./detection_result/new'
    if os.path.exists(tranform_detection):
        shutil.rmtree(tranform_detection)
    os.makedirs(tranform_detection)
    for i in range(len(objs)):
        detection_path=os.path.join(detection_result,'detections_{:02d}.mat'.format(i+1))
        detection=loadmat(detection_path,squeeze_me=True)['all_boxes']

        start,finish=hoi_loc[i]
        all_boxes = {i:{} for i in range(finish-start+1)}
        for detect in detection:
            h_box=list(detect[0])
            o_box=list(detect[1])
            image_id=detect[2]
            hoi_id=detect[3]
            score=list([detect[4]])

            data=h_box+o_box+score
            all_boxes[hoi_id].setdefault(image_id,[]).append(data)

        save_path=os.path.join(tranform_detection,'detections_{:02d}.pkl'.format(i+1))
        with open(save_path,'wb') as f:
            pickle.dump(all_boxes,f)



