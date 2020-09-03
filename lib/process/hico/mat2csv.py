from scipy.io import loadmat
import numpy as np
import pandas as pd

def mat2csv(matdata,csvfile_path):
    data=matdata.copy()

    images_name=[]
    images_size=[]
    images_human_bbox=[]
    images_object_bbox=[]
    images_action_id=[]

    for data_id in range(len(data)):
        temp_data=data[data_id]
        img_name=temp_data['filename']
        img_size=[int(temp_data['size']['width']),
                  int(temp_data['size']['height']),
                  int(temp_data['size']['depth'])]

        data_hois=temp_data['hoi']
        if data_hois.ndim==0:
            data_hois=data_hois[np.newaxis]

        for hoi_id in range(len(data_hois)):
            temp_hoi=data_hois[hoi_id]
            img_action=temp_hoi['id']
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

                human_bbox = [int(human_bbox['x1']),
                              int(human_bbox['x2']),
                              int(human_bbox['y1']),
                              int(human_bbox['y2'])]

                object_bbox = [int(object_bbox['x1']),
                            int(object_bbox['x2']),
                            int(object_bbox['y1']),
                            int(object_bbox['y2'])]

                images_name.append(img_name)
                images_size.append(img_size)
                images_human_bbox.append(human_bbox)
                images_object_bbox.append(object_bbox)
                images_action_id.append(img_action)

    df=pd.DataFrame({
        'name': images_name,
        'action_id': images_action_id,
        'human_bbox': images_human_bbox,
        'object_bbox': images_object_bbox,
        'img_size_w_h': images_size
    })
    print(df.shape)
    df.to_csv(csvfile_path)

if __name__ == '__main__':
    anno_bbox = loadmat('../../../dataset/HICO-DET/anno_bbox.mat', squeeze_me=True) #此处已进行squeeze操作 注意:(1,4)->(4,) (1,1)->()
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']

    train_file_path='./anno_box_train.csv'
    test_file_path='./anno_box_test.csv'
    mat2csv(bbox_train,train_file_path)  # 117871,5
    mat2csv(bbox_test,test_file_path)  #33405,5





