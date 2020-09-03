
import torch
from torch import nn
from torchvision import ops,models
from torch.nn import functional as F
from config import cfg

class iCAN(nn.Module):
    def __init__(self):
        super(iCAN, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.res5=resnet50.layer4
        self.fc=nn.Sequential(
            nn.Linear(in_features=2048,out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, base_feature, rois, head_phi_f, head_g_f, conv_f, identify_block_phi,identify_block_g,output_size=(7,7), spatial_scale=1.0/16.0):
        '''
        :param base_feature: [1,H,W,1024]  #注意feature_bs=1
        :param rois: [None,5]
        :param h_num: positive num
        '''
        if cfg.RESNET.MAX_POOL:
            output_size=(output_size[0]*2,output_size[1]*2)
            pool=self.roi_align(base_feature,rois,output_size,spatial_scale)
            pool=F.max_pool2d(pool,2)
        else:
            pool = self.roi_align(base_feature, rois, output_size, spatial_scale)
        pool=self.res5(pool)
        pool=F.adaptive_avg_pool2d(pool,output_size=(1,1))
        fc_inst=pool.view(pool.size(0),-1)
        fc_1=self.fc(fc_inst)
        fc_1=fc_1.reshape(fc_1.size(0),fc_1.size(1),1,1)

        head_phi=head_phi_f(base_feature)
        head_phi=identify_block_phi(head_phi)
        head_g=head_g_f(base_feature)
        head_g=identify_block_g(head_g)

        attention=torch.mean(torch.mul(head_phi,fc_1),keepdim=True,dim=1)
        attention_shape=attention.size()
        attention=attention.reshape(attention.size(0),attention.size(1),-1)
        attention=F.softmax(attention,dim=-1)
        attention=attention.reshape(attention_shape)

        context=torch.mul(attention,head_g)
        context=conv_f(context)
        context=F.adaptive_avg_pool2d(context,output_size=(1,1))
        fc_context=context.view(context.size(0),-1)

        stream=torch.cat((fc_inst,fc_context),dim=1)
        return fc_inst,attention,stream

    def roi_align(self,base_feature,rois,output_size,spatial_scale):
        rois=rois.detach()

        # type_1 使用torchvision自带的封装函数
        roi_align = ops.RoIAlign(output_size=output_size, spatial_scale=spatial_scale,sampling_ratio=-1)
        crops = roi_align(base_feature, rois)

        assert len(crops)==len(rois)

        # type_2 使用pytorch的仿射变换函数
        # depth=base_feature.size(1)
        # height=base_feature.size(2) - 1
        # width=base_feature.size(3) - 1
        #
        # rois_bs=rois.size(0)
        # x1=rois[:,1::4] * spatial_scale
        # y1=rois[:,2::4] * spatial_scale
        # x2=rois[:,3::4] * spatial_scale
        # y2=rois[:,4::4] * spatial_scale
        #
        # #仿射变换矩阵
        # zero=torch.zeros((rois_bs,1)).type_as(rois)
        # theta=torch.cat([
        #     (x2-x1)/width, zero, -1+(x1+x2)/width,
        #     zero, (y2-y1)/height, -1+(y2+y1)/height
        # ],dim=1).view(-1,2,3)
        #
        # grid=F.affine_grid(theta,torch.Size((rois_bs,depth,output_size[0],output_size[1])))
        # feature=base_feature.expand(rois_bs,-1,-1,-1) # 与grid一一对应
        # grid=grid.type_as(feature)
        # crops=F.grid_sample(feature,grid)
        return crops







