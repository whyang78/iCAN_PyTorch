import sys
sys.path.append(r'/home/cgy/yang/iCAN/lib')
sys.path.append(r'/home/cgy/yang/iCAN/lib/process')
sys.path.append(r'/home/cgy/yang/iCAN/lib/networks')
sys.path.append(r'/home/cgy/yang/iCAN/lib/tools')
sys.path.append(r'/home/cgy/yang/iCAN/lib/networks/RoIAlign')
sys.path.append(r'/home/cgy/yang/iCAN/lib/tools/new_optimizer')

import argparse
import pickle
import torch
from torch import optim,nn
import numpy as np
from torchvision import transforms,models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import math
import shutil
import random
import time
import cv2
from config import cfg
from networks import VCOCO_model
from checkpoint import Checkpoint
from ult import get_vcoco_weight

def parse_args():
    parser = argparse.ArgumentParser(description='Train an iCAN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=300000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='iCAN_ResNet50_HICO', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=30, type=int)
    # parser.add_argument('--Restore_flag', dest='Restore_flag',
    #         help='How many ResNet blocks are there?',
    #         default=5, type=int)
    args = parser.parse_args()
    return args

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def restore_pretrained_model(pretrained_model):
    pretrained_model_state=pretrained_model.state_dict()
    key_word = 'backbone.body.'
    pretrained_model_state_dict={ k[len(key_word):] : pretrained_model_state[k]
                                 for k in pretrained_model_state if str(k).startswith(key_word)}

    net_state_dict={k: net.state_dict()[k] for k in net.state_dict()}
    for k,p in pretrained_model_state_dict.items():
        if not str(k).startswith('layer4'):
            new_k='buildbone.'+str(k)
            net_state_dict[new_k]=p
        else:
            for i in ['human_stream.res5','object_stream.res5']:
                new_k=str(k).replace('layer4',i)
                net_state_dict[new_k]=p
    return net_state_dict

def process_image(image_path):
    # image=Image.open(image_path).convert('RGB')
    # tfs=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # image=tfs(image).float().unsqueeze(0)

    # type2
    image=cv2.imread(image_path).astype(np.float32)
    image-=cfg.PIXEL_MEANS
    image=torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0)
    return image

def generate_gt_hoi(action_list):
    gt_hoi=np.zeros(29)
    for i in action_list:
        gt_hoi[i]=1
    gt_hoi=gt_hoi.reshape(1,29)
    return gt_hoi

def bbox_iou(box1,box2):
    x1_max=np.maximum(box1[0],box2[0])
    y1_max=np.maximum(box1[1],box2[1])
    x2_min=np.minimum(box1[2],box2[2])
    y2_min=np.minimum(box1[3],box2[3])
    w=max(0,x2_min-x1_max+1)
    h=max(0,y2_min-y1_max+1)

    overlap=w*h
    union=(box1[3]-box1[1]+1)*(box1[2]-box1[0]+1)+(box2[3]-box2[1]+1)*(box2[2]-box2[0]+1)-overlap
    iou=overlap/union
    return float(iou)

def augment_bbox(bbox,shape,pos_augment,thresh,max_count=150):
    # shape->[h,w]
    box=np.array([[0,bbox[0],bbox[1],bbox[2],bbox[3]]]).astype(np.float64)

    time_count=0
    count=0
    while count<pos_augment:
        time_count+=1

        box_width=bbox[2]-bbox[0]+1
        box_height=bbox[3]-bbox[1]+1
        width_center=(bbox[2]+bbox[0])/2
        height_center=(bbox[3]+bbox[1])/2

        ratio=1+random.randint(-10,10)*0.01
        width_shift=random.randint(-math.floor(box_width),math.floor(box_width))*0.1
        height_shift=random.randint(-math.floor(box_height),math.floor(box_height))*0.1

        x1=max(0,width_center+width_shift-ratio*box_width/2)
        y1=max(0,height_center+height_shift-ratio*box_height/2)
        x2=min(shape[1]-1,width_center+width_shift+ratio*box_width/2)
        y2=min(shape[0]-1,height_center+height_shift+ratio*box_height/2)

        if bbox_iou(bbox,np.array([x1,y1,x2,y2]))>thresh:
            count+=1
            box_=np.array([[0,x1,y1,x2,y2]])
            box=np.concatenate((box,box_),axis=0)
        if time_count>max_count:
            return box
    return box


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

# def get_sp_pattern(human_box,object_box):
#     sp_size=64
#     new_x1=min(human_box[0],object_box[0])
#     new_y1=min(human_box[1],object_box[1])
#
#     hbox=np.array([human_box[0]-new_x1,human_box[1]-new_y1,human_box[2]-new_x1,human_box[3]-new_y1]).astype(np.float64)
#     obox=np.array([object_box[0]-new_x1,object_box[1]-new_y1,object_box[2]-new_x1,object_box[3]-new_y1]).astype(np.float64)
#     sp_h=max(hbox[3],obox[3])-min(hbox[1],obox[1])+1
#     sp_w=max(hbox[2],obox[2])-min(hbox[0],obox[0])+1
#
#     factor=min(sp_size/sp_h,sp_size/sp_w)
#     h_pad=(sp_size-sp_h*factor)/2.0
#     w_pad=(sp_size-sp_w*factor)/2.0
#
#     sp=np.zeros((1,2,sp_size,sp_size))
#     sp[0, 0, int(hbox[1]*factor+h_pad):min(int(hbox[3]*factor+h_pad+factor),sp_size),
#              int(hbox[0]*factor+w_pad):min(int(hbox[2]*factor+w_pad+factor),sp_size)]=1
#     sp[0, 1, int(obox[1]*factor+h_pad):min(int(obox[3]*factor+h_pad+factor),sp_size),
#              int(obox[0]*factor+w_pad):min(int(obox[2]*factor+w_pad+factor),sp_size)]=1
#     return sp


def get_pos_augment_and_neg_select(gt, gt_shape,Trainval_GT, Trainval_N,Pos_augment, Neg_select, thresh=0.7):
    image_id = gt[0]
    human_bbox = gt[2]
    object_bbox = gt[3]

    gt_sp_ = generate_gt_hoi(gt[1])
    gt_obj_ = generate_gt_hoi(gt[1])
    gt_h_ = generate_gt_hoi(gt[4])
    mask_sp_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29).astype(
        np.float)
    mask_obj_ = np.asarray(
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]).reshape(1, 29).astype(
        np.float)
    mask_h_ = np.asarray(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 29).astype(
        np.float)

    augment_human_bbox = augment_bbox(human_bbox, gt_shape, Pos_augment, thresh, max_count=150)
    augment_object_bbox = augment_bbox(object_bbox, gt_shape, Pos_augment, thresh, max_count=150)

    num_pos = min(len(augment_human_bbox), len(augment_object_bbox))
    augment_human_bbox = augment_human_bbox[:num_pos]
    augment_object_bbox = augment_object_bbox[:num_pos]

    if image_id in Trainval_N:
        neg_select = min(Neg_select, len(Trainval_N[image_id]))
        select_list = random.sample(range(len(Trainval_N[image_id])), neg_select)
        for index in select_list:
            neg = Trainval_N[image_id][index]
            neg_human_bbox = np.array([[0, neg[2][0], neg[2][1], neg[2][2], neg[2][3]]]).astype(np.float32)
            neg_object_bbox = np.array([[0, neg[3][0], neg[3][1], neg[3][2], neg[3][3]]]).astype(np.float32)

            augment_human_bbox = np.concatenate((augment_human_bbox, neg_human_bbox), axis=0)
            augment_object_bbox = np.concatenate((augment_object_bbox, neg_object_bbox), axis=0)

    num_pos_neg = len(augment_human_bbox)

    pattern = np.empty((0, 2, 64, 64), dtype=np.float32)
    for i in range(num_pos_neg):
        sp = get_sp_pattern(augment_human_bbox[i, 1:], augment_object_bbox[i, 1:])
        pattern = np.concatenate((pattern, sp), axis=0)

    gt_sp = gt_sp_
    gt_obj = gt_obj_
    gt_h = gt_h_
    mask_h = mask_h_
    mask_obj = mask_obj_
    mask_sp = mask_sp_
    for i in range(num_pos - 1):
        gt_sp = np.concatenate((gt_sp, gt_sp_), axis=0)
        gt_obj = np.concatenate((gt_obj, gt_obj_), axis=0)
        gt_h = np.concatenate((gt_h, gt_h_), axis=0)
        mask_obj = np.concatenate((mask_obj, mask_obj_), axis=0)
        mask_h = np.concatenate((mask_h, mask_h_), axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp = np.concatenate((mask_sp, mask_sp_), axis=0)

    for i in range(num_pos_neg - num_pos):
        gt_sp = np.concatenate((gt_sp, np.zeros(29).reshape(1, 29)), axis=0)

    augment_human_bbox_all = augment_human_bbox.reshape(num_pos_neg, 5)
    augment_human_bbox = augment_human_bbox[:num_pos].reshape(num_pos, 5)
    augment_object_bbox = augment_object_bbox[:num_pos].reshape(num_pos, 5)
    pattern = pattern.reshape(num_pos_neg, 2, 64, 64)
    gt_sp = gt_sp.reshape(num_pos_neg, 29)
    gt_h = gt_h.reshape(num_pos, 29)
    gt_obj = gt_obj.reshape(num_pos, 29)
    mask_sp = mask_sp.reshape(num_pos_neg, 29)
    mask_obj = mask_obj.reshape(num_pos, 29)
    mask_h = mask_h.reshape(num_pos, 29)

    return augment_human_bbox_all, augment_human_bbox, augment_object_bbox, pattern, num_pos, gt_sp, gt_h, gt_obj, mask_sp, mask_h, mask_obj


def get_instance(Trainval_GT, Trainval_N,Pos_augment, Neg_select, datasample):
    gt=Trainval_GT[datasample]
    image_id=gt[0]
    train_path=r'/home/cgy/yang/train2014'
    image_path=os.path.join(train_path,'COCO_train2014_'+str(image_id).zfill(12)+'.jpg')
    image=process_image(image_path)
    image_shape=[image.shape[2],image.shape[3]]

    augment_human_bbox_all,augment_human_bbox,augment_object_bbox,pattern,num_pos,\
    gt_sp,gt_h,gt_obj,mask_sp,mask_h,mask_obj=\
        get_pos_augment_and_neg_select(gt,image_shape,Trainval_GT,Trainval_N,Pos_augment,Neg_select,thresh=0.7)

    batchsample={}
    batchsample['image']=image
    batchsample['h_boxes']=torch.from_numpy(augment_human_bbox_all).float()
    batchsample['o_boxes']=torch.from_numpy(augment_object_bbox).float()
    batchsample['sp'] = torch.from_numpy(pattern).float()
    batchsample['num_pos'] = int(num_pos)
    batchsample['gt_h']=torch.from_numpy(gt_h).float()
    batchsample['gt_sp'] = torch.from_numpy(gt_sp).float()
    batchsample['gt_obj'] = torch.from_numpy(gt_obj).float()
    batchsample['mask_sp'] = torch.from_numpy(mask_sp).float()
    batchsample['mask_h'] = torch.from_numpy(mask_h).float()
    batchsample['mask_obj']=torch.from_numpy(mask_obj).float()

    return batchsample


def train_net(network, Trainval_GT, Trainval_N,  Pos_augment, Neg_select,
              output_dir,tb_dir,pretrained_model,device,max_iters=300000):
    #若存在文件夹，则删除重建
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)
    os.makedirs(tb_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    #载入预训练模型
    # net_state_dict=restore_pretrained_model(pretrained_model)
    # network.load_state_dict(net_state_dict)

    #设置优化器 损失函数
    # network.to(device)
    ignored_params = list(map(id, network.identify_block_phi.parameters())) + list(
        map(id, network.identify_block_g.parameters())) + list(map(id, network.head_phi[1].parameters())) + list(
        map(id, network.head_g[1].parameters())) + list(map(id, network.conv[1].parameters()))
    base_params = filter(lambda p: p.requires_grad and id(p) not in ignored_params, network.parameters())
    optimizer = optim.SGD([{'params': base_params, 'lr': 0.00001},
                           {'params': network.identify_block_phi.parameters()},
                           {'params': network.identify_block_g.parameters()},
                           {'params': network.head_phi[1].parameters()},
                           {'params': network.head_g[1].parameters()},
                           {'params': network.conv[1].parameters()}], lr=0.001,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.9)
    # criterion=nn.BCELoss(reduction='none').to(device)
    obj_weight,h_weight=get_vcoco_weight()
    obj_criterion=nn.BCELoss(weight=obj_weight.to(device),reduction='none').to(device)
    h_criterion=nn.BCELoss(weight=h_weight.to(device),reduction='none').to(device)
    # focal_criterion=BCEFocalLoss()

    # tensorboard记录 checkpoints初始化
    writer = SummaryWriter(tb_dir)
    checkpoint=Checkpoint(network,optimizer,output_dir,max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
    ckpt_name = 'iter_0.pth'
    checkpoint.save(ckpt_name)

    #训练
    dataLength=len(Trainval_GT)
    cur_iter=0
    total_time=0
    while cur_iter<max_iters+1:
        network.train()
        start=time.time()
        datasample=cur_iter%dataLength  #当前取的样本的索引
        batchsample=get_instance(Trainval_GT,Trainval_N,Pos_augment,Neg_select,datasample)
        if batchsample['num_pos']<=1:
            print(cur_iter)
            cur_iter+=1
            continue
        h_score,o_score,sp_score,atten_h,atten_o=network(batchsample['image'].to(device),batchsample['sp'].to(device),
                                           batchsample['h_boxes'].to(device),batchsample['o_boxes'].to(device),
                                           batchsample['num_pos'])

        num_pos= batchsample['num_pos']
        gt_h = batchsample['gt_h'].to(device)
        mask_h=batchsample['mask_h'].to(device)
        h_loss_mat=h_criterion(h_score[:num_pos,:],gt_h[:num_pos,:])
        h_loss=torch.mean(torch.mul(h_loss_mat,mask_h))

        gt_obj = batchsample['gt_obj'].to(device)
        mask_obj = batchsample['mask_obj'].to(device)
        o_loss_mat = obj_criterion(o_score[:num_pos, :], gt_obj[:num_pos, :])
        o_loss=torch.mean(torch.mul(o_loss_mat,mask_obj))

        gt_sp = batchsample['gt_sp'].to(device)
        mask_sp = batchsample['mask_sp'].to(device)
        sp_loss_mat = obj_criterion(sp_score, gt_sp)
        sp_loss=torch.mean(torch.mul(sp_loss_mat,mask_sp))
        loss=2*h_loss+o_loss+sp_loss

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(network.parameters(),max_norm=1.0) #梯度裁剪
        optimizer.step()
        scheduler.step(epoch=cur_iter)
        cur_iter+=1

        finish=time.time()
        total_time+=finish-start

        if (cur_iter % 5000 == 0) or (cur_iter < 20):
            writer.add_scalar('train/loss',loss.cpu().item(),cur_iter)
            writer.add_text('train/text', 'epoch:{},loss:{:.4f}'.format(cur_iter, loss.cpu().item()),cur_iter)
            atten_h,atten_o=atten_h.cpu().float().data,atten_o.cpu().float().data
            atten_h=(atten_h-torch.min(atten_h[0]))/(torch.max(atten_h[0])-torch.min(atten_h[0]))
            writer.add_images('train/attention_human',atten_h,global_step=cur_iter)
            atten_o=(atten_o-torch.min(atten_o[0]))/(torch.max(atten_o[0])-torch.min(atten_o[0]))
            writer.add_images('train/attention_object',atten_o,global_step=cur_iter)

        if (cur_iter % 10000 == 0 and cur_iter != 0) or (cur_iter == 10):
            ckpt_name=f'iter_{cur_iter}.pth'
            checkpoint.save(ckpt_name)


    print('vcoco_average_time',float(total_time)/float(max_iters))
    writer.close()

if __name__ == '__main__':
    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    arg=parse_args()

    gt_path=r'/home/cgy/yang/iCAN/Trainval_GT_VCOCO.pkl'
    neg_path=r'/home/cgy/yang/iCAN/Trainval_Neg_VCOCO.pkl'
    trainval_gt=load_pkl(gt_path)
    trainval_neg=load_pkl(neg_path)

    tb_dir=r'/home/cgy/yang/iCAN/logs1'
    output_dir=r'/home/cgy/yang/iCAN/output1'

    #加载并提取预训练res50参数
    pretrained_model=models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_path = './vcoco_test_model.pth'
    weight = torch.load(weight_path, map_location=torch.device('cpu'))

    net = VCOCO_model()
    net_state = net.state_dict()
    new_state_dict = {k: v for k, v in weight.items() if k in net_state}
    net_state.update(new_state_dict)
    net.load_state_dict(net_state)
    net.to(device)

    train_net(net,trainval_gt,trainval_neg,arg.Pos_augment,arg.Neg_select,output_dir,tb_dir,
              pretrained_model,device,arg.max_iters)


