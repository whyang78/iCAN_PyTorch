from scipy.io import loadmat
import numpy as np
import pandas as pd
import pickle

def get_hoi_class_verb(vb_path):
    '''
    v-o  -->  v
    :return:
    '''

    vb_list=[]
    with open(vb_path,'r+') as f:
        hois=f.readlines()
        for line in hois:
            vb_list.append(line.strip().split(' ')[-1])
    del vb_list[:2]
    vb_list=list(set(vb_list))
    vb_list.sort()
    return vb_list

#nms操作  有些差异
def nms(bbox_list, thresh=0.7):
    bbox_array = np.array(bbox_list)
    x1 = bbox_array[:, 0]
    y1 = bbox_array[:, 1]
    x2 = bbox_array[:, 2]
    y2 = bbox_array[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    index = np.arange(len(bbox_array)) #原应根据score排序，因全部是GT样本(全部score=1)，所以就按照读取顺序即可
    while len(index) > 0:
        x1_max = np.maximum(x1[index[0]], x1[index[1:]])
        y1_max = np.maximum(y1[index[0]], y1[index[1:]])
        x2_min = np.minimum(x2[index[0]], x2[index[1:]])
        y2_min = np.minimum(y2[index[0]], y2[index[1:]])

        w = np.maximum(0, x2_min - x1_max + 1)
        h = np.maximum(0, y2_min - y1_max + 1)
        overlap = w * h
        ious = overlap / (areas[index[0]] + areas[index[1:]] - overlap)

        idx_g=np.where(ious>thresh)[0]
        bbox_array[index[idx_g+1]]=bbox_array[index[0]]

        idx_l=np.where(ious<=thresh)[0]
        index=index[idx_l+1]
    return bbox_array

#得到以此object为对象的所有hoi类别
def get_object_all_hoi(hois, hoi):
    hoi_object = hois.iloc[hoi, 1]
    all_hoi = list(hois[hois.iloc[:, 1] == hoi_object].index)
    return all_hoi

#得到Trainval_GT_HICO.pkl  产生的数目与官方有些小差异，通过调节nms的thresh控制数目
def make_trainVal_GT_HICO(matdata,action_list_path):
    data = matdata.copy()
    hois = pd.read_csv(action_list_path)

    all_list = []
    for data_id in range(len(data)):
        images_human_bbox = []
        images_object_bbox = []
        images_action_id = []

        temp_data = data[data_id]
        img_name = temp_data['filename']

        data_hois = temp_data['hoi']
        if data_hois.ndim == 0:
            data_hois = data_hois[np.newaxis]

        for hoi_id in range(len(data_hois)):
            temp_hoi = data_hois[hoi_id]
            img_action = temp_hoi['id'] - 1
            img_invis = temp_hoi['invis']
            if img_invis == 1:  # 不可见则直接跳过此hoi
                continue

            if temp_hoi['connection'].ndim == 1:
                temp_hoi['connection'] = temp_hoi['connection'][np.newaxis]
            if temp_hoi['bboxhuman'].ndim == 0:
                temp_hoi['bboxhuman'] = temp_hoi['bboxhuman'][np.newaxis]
            if temp_hoi['bboxobject'].ndim == 0:
                temp_hoi['bboxobject'] = temp_hoi['bboxobject'][np.newaxis]

            for connect_id in range(len(temp_hoi['connection'])):
                if len(temp_hoi['connection'][connect_id]) < 2:
                    continue

                human_bbox_id = temp_hoi['connection'][connect_id][0] - 1
                object_bbox_id = temp_hoi['connection'][connect_id][1] - 1
                human_bbox = temp_hoi['bboxhuman'][human_bbox_id]
                object_bbox = temp_hoi['bboxobject'][object_bbox_id]

                human_bbox = [int(human_bbox['x1']) - 1,
                              int(human_bbox['y1']) - 1,
                              int(human_bbox['x2']) - 1,
                              int(human_bbox['y2']) - 1]

                object_bbox = [int(object_bbox['x1']) - 1,
                               int(object_bbox['y1']) - 1,
                               int(object_bbox['x2']) - 1,
                               int(object_bbox['y2']) - 1]

                images_human_bbox.append(human_bbox)
                images_object_bbox.append(object_bbox)
                images_action_id.append(img_action)
        if len(images_human_bbox) == 0:
            continue
        action_array = np.array(images_action_id)[:, np.newaxis]
        human_bbox_array = nms(images_human_bbox, 0.7)
        object_bbox_array = nms(images_object_bbox, 0.65)
        human_object_pair = np.concatenate((human_bbox_array, object_bbox_array, action_array), axis=1)
        human_object_pair_unique = np.unique(human_object_pair[:, :-1], axis=0)

        for pair in human_object_pair_unique:
            idx = [i for i, p in enumerate(human_object_pair) if np.all(p[:-1] == pair)]
            hoi_list = human_object_pair[idx, -1].tolist()
            all_hoi_object = get_object_all_hoi(hois, hoi_list[0])
            all_list.append([img_name,
                             hoi_list,
                             pair[:4].tolist(),
                             pair[4:].tolist(),
                             all_hoi_object
                             ])
    return all_list


if __name__ == '__main__':
    vb_path='../../../dataset/HICO-DET/hico_list_vb.txt'
    vb_list=get_hoi_class_verb(vb_path)

    anno_bbox = loadmat('../../../dataset/HICO-DET/anno_bbox.mat', squeeze_me=True)  # 此处已进行squeeze操作 注意:(1,4)->(4,) (1,1)->()
    bbox_train = anno_bbox['bbox_train']
    trainval_gt= make_trainVal_GT_HICO(bbox_train, '../../../dataset/HICO-DET/action_list.csv')
    with open('./my_trainval_gt_HICO.pkl','wb') as p:
        pickle.dump(trainval_gt,p)
