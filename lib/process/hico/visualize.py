import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def hoi_bbox_visualize(bbox_file_path,action_path,image_path,num=5):
    data=pd.read_csv(bbox_file_path)
    data['human_bbox'] = data['human_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))
    data['object_bbox'] = data['object_bbox'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

    action_list=pd.read_csv(action_path)

    for i in range(num):
        temp_data=data.iloc[i]

        filename=temp_data['name']
        path=os.path.join(image_path,filename)
        img=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

        human_bbox=temp_data['human_bbox']
        object_bbox=temp_data['object_bbox']
        action=action_list[action_list['id']==temp_data['action_id']].iloc[0]

        cv2.rectangle(img,(human_bbox[0],human_bbox[2]),(human_bbox[1],human_bbox[3]),(0,0,255),2)
        cv2.rectangle(img,(object_bbox[0],object_bbox[2]),(object_bbox[1],object_bbox[3]),(0,255,0),2)
        fig=plt.figure(figsize=(5,5))
        plt.title('verb:{},object:{}'.format(action['verb'],action['object']))
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    # train_file_path='./anno_box_train.csv'
    # test_file_path='./anno_box_test.csv'
    # action_csv='./action_list.csv'
    #
    # train_images_path='../../images/train2015'
    # test_images_path='../../images/test2015'
    # hoi_bbox_visualize(train_file_path,action_csv,train_images_path,num=20)
    # # 2, 53, array(, dtype=float32), array(

    path = r'C:\Users\asus\Desktop\iCAN\dataset\HICO-DET\images\train2015\HICO_train2015_00000004.jpg'
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    human_bbox =list(map(int,[222, 28, 483, 367]))
    object_bbox = list(map(int,[268, 88, 362, 121]))
    action = 355

    cv2.rectangle(img, (human_bbox[0], human_bbox[1]), (human_bbox[2], human_bbox[3]), (0, 0, 255), 2)
    cv2.rectangle(img, (object_bbox[0], object_bbox[1]), (object_bbox[2], object_bbox[3]), (0, 255, 0), 2)
    fig = plt.figure(figsize=(5, 5))
    # plt.title('verb:{},object:{}'.format('no_interaction','handbag'))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(fname="hico.svg", format="svg")
    plt.show()
