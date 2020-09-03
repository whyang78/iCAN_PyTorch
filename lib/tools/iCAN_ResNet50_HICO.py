
import torch
from torch import nn
from torchvision import models,ops
from torch.nn import functional as F
from iCAN_HICO import iCAN
from config import cfg


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self,inChannel,outChannel):
        super(ResidualBlock, self).__init__()

        self.left=nn.Sequential(
            nn.Conv2d(inChannel,outChannel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel,outChannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannel, inChannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inChannel),
           
        )


    def forward(self, x):
        out=self.left(x)
        out=out+x
        out=F.relu(out)
        return out

class HICO_model(nn.Module):
    def __init__(self,num_classes=600):
        super(HICO_model, self).__init__()
        self.num_classes=num_classes
        resnet50 = models.resnet50(pretrained=True)
        self.buildbone=nn.Sequential()
        for i in range(7):
            b = list(resnet50.named_children())[i]
            self.buildbone.add_module(b[0], b[1])

        self.human_stream=iCAN()
        self.object_stream=iCAN()
        self.identify_block_phi = ResidualBlock(512, 128)
        self.identify_block_g = ResidualBlock(512, 128)
        self.head_phi=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.head_g=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.interaction_pattern=nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=64,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Flatten(),
        )
        self.fc_h=nn.Sequential(
            nn.Linear(3072, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        self.fc_o=nn.Sequential(
            nn.Linear(3072,1024,bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,num_classes),
            nn.Sigmoid()
        )
        self.fc_sp = nn.Sequential(
            nn.Linear(7456, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )
        self.initialize_param()

    def initialize_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for layer in [self.buildbone,self.fc_h,self.fc_o,self.fc_sp,self.human_stream.res5,self.object_stream.res5,
        self.identify_block_phi,self.identify_block_g]:
            for m in layer.modules():
                if isinstance(m,nn.BatchNorm1d):
                    m.affine=True
                    m.momentum=0.997
                    m.eps=1e-5
                elif isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias,0.0)
                    if isinstance(m,nn.Linear) and m.out_features==self.num_classes:
                        nn.init.normal_(m.weight, mean=0, std=0.01)

        for name,param in self.buildbone.named_parameters():
            if not name.startswith('layer'):
                param.requires_grad=False
            else:
                if int(name[len('layer')]) <= 3:
                    param.requires_grad = False

        # for param in self.parameters():
        #     param.requires_grad=False
        #
        # for m in [self.identify_block_phi,self.identify_block_g]:
        #     for k,v in m.named_parameters():
        #         v.requires_grad=True

    def forward(self, image, spatial, h_boxes, o_boxes, num_pos):
        head=self.buildbone(image)
        h_inst,atten_h,h_all=self.human_stream(head,h_boxes,self.head_phi,self.head_g,self.conv,self.identify_block_phi,self.identify_block_g)
        h_score = self.fc_h(h_all)
        _,atten_o,o_all=self.object_stream(head,o_boxes[:num_pos,:],self.head_phi,self.head_g,self.conv,self.identify_block_phi,self.identify_block_g)
        o_score=self.fc_o(o_all)
        sp = self.interaction_pattern(spatial)
        sp=torch.cat((h_inst,sp),dim=1)
        sp_score=self.fc_sp(sp)

        return h_score,o_score,sp_score,atten_h,atten_o

if __name__ == '__main__':
    net=HICO_model()
    print(net)
    # net_state=net.state_dict()

    # import pandas as pd
    # import pickle
    # data=pd.read_csv('./param.csv',encoding='utf-8',header=None)
    # tf_param=list(data.iloc[:,0])
    # pt_param=list(data.iloc[:,1])
    # with open('./model.pkl','rb') as f:
    #     tf=pickle.load(f)
    #
    # save_path='./pytorch_model.pth'
    #
    # count=0
    # for tf_p,pt_p in zip(tf_param,pt_param):
    #     if 'Momentum' not in tf_p:
    #         param=tf[str(tf_p).strip()]
    #         if str(pt_p).strip() in net_state and net_state[str(pt_p).strip()].shape==param.shape:
    #             net_state[str(pt_p).strip()]=torch.from_numpy(param).float()
    #             count+=1
    # torch.save(net_state,save_path)

    # aa = torch.load(save_path)
    #
    # path='./test_model.pth'
    # for k,v in aa.items():
    #     if 'num_batches_tracked' in k:
    #         aa[k]=torch.tensor([1800001],dtype=torch.int64)
    # torch.save(aa,path)
    # bb=torch.load(path)




