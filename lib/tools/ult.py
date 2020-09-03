import torch
import pandas as pd
import numpy as np
import pickle

def get_hico_det_weight(gt_path,neg_path):
    # path = './Trainval_GT_HICO.pkl'
    # with open(path, 'rb') as f:
    #     u = pickle._Unpickler(f)
    #     u.encoding = 'latin1'
    #     p = u.load()
    # path = './Trainval_Neg_HICO.pkl'
    # with open(path, 'rb') as f:
    #     u = pickle._Unpickler(f)
    #     u.encoding = 'latin1'
    #     q = u.load()
    with open(gt_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    with open(neg_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        q = u.load()

    total = np.zeros(600)
    for elem in p:
        for hoi in elem[1]:
            total[hoi] += 1

    for key, val in dict(q).items():
        for elem in val:
            total[elem[1]] += 1

    weight=np.log(sum(total)/total)
    weight=torch.from_numpy(weight).float()
    return weight

#这是上面计算出来的结果，计算太慢，省略计算
def get_weight():
    weight=torch.tensor([
                9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423,
                11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699,
                6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912,
                5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048,
                8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585,
                12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731,
                7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116,
                14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678,
                12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933,
                14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973,
                12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967,
                4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679,
                9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917,
                10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057,
                8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799,
                9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912,
                9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591,
                12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264,
                11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264,
                7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304,
                10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384,
                11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143,
                11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264,
                7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264,
                8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909,
                7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748,
                14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135,
                11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411,
                14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533,
                10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361,
                9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222,
                9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963,
                8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324,
                9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948,
                5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264,
                11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248,
                10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862,
                8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224,
                12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973,
                12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493,
                14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973,
                11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892,
                10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825,
                12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799,
                9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505,
                12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368,
                7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925,
                7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015,
                7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551,
                9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584,
                5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826,
                11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799,
                10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059,
                13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571,
                11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501,
                14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411,
                7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305,
                11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186,
                12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264,
                10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
            ]).float()
    return weight

def get_vcoco_weight():
    obj_weight = np.array([3.3510249, 3.4552405, 4.0257854, 0.0, 4.088436,
                           3.4370995, 3.85842, 4.637334, 3.5487218, 3.536237,
                           2.5578923, 3.342811, 3.8897269, 4.70686, 3.3952892,
                           3.9706533, 4.504736, 0.0, 1.4873443, 3.700363,
                           4.1058283, 3.6298118, 0.0, 6.490651, 5.0808263,
                           1.520838, 3.3888445, 0.0, 3.9899964], dtype='float32')
    h_weight = np.array([4.0984106, 4.102459, 4.0414762, 4.060745, 4.0414762,
                              3.9768186, 4.23686, 5.3542085, 3.723717, 3.4699364,
                              2.4587274, 3.7167964, 4.08836, 5.050695, 3.9077065,
                              4.534647, 3.4699364, 2.9466882, 1.8585607, 3.9433942,
                              3.9433942, 4.3523254, 3.8368235, 6.4963055, 5.138182,
                              1.7807873, 4.080392, 1.9544303, 4.5761204], dtype='float32')
    obj_weight=torch.from_numpy(obj_weight).float()
    h_weight=torch.from_numpy(h_weight).float()
    return obj_weight,h_weight

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pt, target):
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def apply_prior(Object, prediction):
    if Object[4] != 32:  # not a snowboard, then the action is impossible to be snowboard
        prediction[0][21] = 0

    if Object[4] != 74:  # not a book, then the action is impossible to be read
        prediction[0][24] = 0

    if Object[4] != 33:  # not a sports ball, then the action is impossible to be kick
        prediction[0][7] = 0

    if (Object[4] != 41) and (Object[4] != 40) and (Object[4] != 42) and (
            Object[4] != 46):  # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[0][13] = 0

    if Object[4] != 37:  # not a skateboard, then the action is impossible to be skateboard
        prediction[0][26] = 0

    if Object[4] != 38:  # not a surfboard, then the action is impossible to be surfboard
        prediction[0][0] = 0

    if Object[4] != 31:  # not a ski, then the action is impossible to be ski
        prediction[0][1] = 0

    if Object[4] != 64:  # not a laptop, then the action is impossible to be work on computer
        prediction[0][8] = 0

    if (Object[4] != 77) and (Object[4] != 43) and (
            Object[4] != 44):  # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        prediction[0][2] = 0

    if (Object[4] != 33) and (
            Object[4] != 30):  # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        prediction[0][15] = 0
        prediction[0][28] = 0

    if Object[4] != 68:  # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[0][6] = 0

    if (Object[4] != 14) and (Object[4] != 61) and (Object[4] != 62) and (Object[4] != 60) and (Object[4] != 58) and (
            Object[
                4] != 57):  # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[0][12] = 0

    if (Object[4] != 32) and (Object[4] != 31) and (Object[4] != 37) and (Object[
                                                                              4] != 38):  # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        prediction[0][11] = 0

    if (Object[4] != 47) and (Object[4] != 48) and (Object[4] != 49) and (Object[4] != 50) and (Object[4] != 51) and (
            Object[4] != 52) and (Object[4] != 53) and (Object[4] != 54) and (Object[4] != 55) and (Object[
                                                                                                        4] != 56):  # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[0][9] = 0

    if (Object[4] != 43) and (Object[4] != 44) and (
            Object[4] != 45):  # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[0][16] = 0

    if (Object[4] != 39) and (
            Object[4] != 35):  # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        prediction[0][19] = 0

    if (Object[4] != 33):  # not 'sports ball, then the action is impossible to be hit_obj
        prediction[0][20] = 0

    if (Object[4] != 2) and (Object[4] != 4) and (Object[4] != 6) and (Object[4] != 8) and (Object[4] != 9) and (
            Object[4] != 7) and (Object[4] != 5) and (Object[4] != 3) and (Object[4] != 18) and (Object[
                                                                                                     4] != 21):  # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        prediction[0][5] = 0

    if (Object[4] != 2) and (Object[4] != 4) and (Object[4] != 18) and (Object[4] != 21) and (Object[4] != 14) and (
            Object[4] != 57) and (Object[4] != 58) and (Object[4] != 60) and (Object[4] != 62) and (
            Object[4] != 61) and (Object[4] != 29) and (Object[4] != 27) and (Object[
                                                                                  4] != 25):  # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[0][10] = 0

    if (Object[4] == 1):
        prediction[0][4] = 0

    return prediction

if __name__ == '__main__':
    # from torch import nn
    # data=torch.tensor([[0.2,0.5,0.9],[0.7,0.1,0.8]]).float()
    # target=torch.tensor([[0,1,1],[1,0,0]]).float()
    # print('*'*5)
    # criter=nn.BCELoss(reduction='none')
    # loss=criter(data,target)
    # a=torch.tensor([[0,0,1],[0,0,1]]).float()
    #
    # print(loss)
    # loss = loss * a
    # print(torch.mean(loss))
    # print('*'*10)
    # criter=nn.BCELoss(weight=torch.tensor([0,0,1]).float())
    # loss=criter(data,target)
    # print(loss)

    import pickle
    import glob
    path='./temp/*.pkl'
    new_data={}
    for i in glob.iglob(path):
        with open(i,'rb') as f:
            data=pickle.load(f)
        new_data.update(data)
    with open('./iter_0.pkl','wb') as f:
        pickle.dump(new_data,f)
